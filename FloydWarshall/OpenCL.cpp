#include "Intestazione.h"
#include <vector>
#include <iostream>
#include <chrono>
#include <random>
#include <CL\cl.hpp>
#include <fstream>

#define DEBUG false

std::string simple_block_fw =
"   void kernel simple_block_fw(global int* G, const int k, const int n, const int pr, const int pc){"
" int mr = get_global_id(0), mc = get_global_id(1);"
"int sizer = n/pr, sizec = n/pc; int i, j;"
"for(i=mr*sizer; i < mr*sizer + sizer; i++)"
"for(j=mc*sizec; j < mc*sizec + sizec; j++){"
"int my_data_old = G[i*n + j], my_data_new = G[i*n + k] + G[k*n + j];"
" G[i*n + j] = (my_data_old < my_data_new)? my_data_old : my_data_new;}                 "
"   } ";

std::string tiled_fw_stage_1 =
"void kernel tiled_fw_stage_1(global int* G, const int k, const int n, const int tile_dim, local int* GL){"
" int i = k*tile_dim + get_local_id(0), j = k*tile_dim + get_local_id(1);"
"int i_equal = get_local_id(0), j_equal = get_local_id(1);"
"int index = i*n + j;"
"GL[i_equal*tile_dim + j_equal] = G[index];"
"barrier(CLK_LOCAL_MEM_FENCE);"
"for(int inner_k = 0; inner_k < tile_dim; inner_k++){"
"if(GL[i_equal*tile_dim + j_equal] > GL[i_equal*tile_dim + inner_k] + GL[inner_k*tile_dim + j_equal])"
"GL[i_equal*tile_dim + j_equal] = GL[i_equal*tile_dim + inner_k] + GL[inner_k*tile_dim + j_equal];"
"barrier(CLK_LOCAL_MEM_FENCE);"
"}"
"G[index] = GL[i_equal*tile_dim + j_equal];"
"}";

std::string tiled_fw_stage_2 =
"void kernel tiled_fw_stage_2(global int* G, const int k, const int n, const int ngroups, const int tile_dim, local int* GLKK, local int* GLIK, local int* GLKJ){"
"	int nblocks = n/tile_dim;"
"	int blocks_per_group = nblocks/ngroups;"
"	if(get_group_id(1) == 0){"
"		int group_index = get_group_id(0);"
"		int index = get_local_id(0)*tile_dim + get_local_id(1);"
"		GLKK[index] = G[(k*tile_dim + get_local_id(0))*n + (k*tile_dim + get_local_id(1))];"
"		barrier(CLK_LOCAL_MEM_FENCE);"
"		int base_index = group_index*blocks_per_group;"
"		for(int i = base_index; i < base_index + blocks_per_group; i++){"
"			int block_r = i;"
"			if(block_r != k){"
"				GLIK[index] = G[(block_r*tile_dim + get_local_id(0))*n + (k*tile_dim + get_local_id(1))];"
"				barrier(CLK_LOCAL_MEM_FENCE);"
"				GLKJ[index] = G[(k*tile_dim + get_local_id(0))*n + (block_r*tile_dim + get_local_id(1))];"
"				barrier(CLK_LOCAL_MEM_FENCE);"
"				for(int inner_k = 0; inner_k < tile_dim; inner_k++){"
"					if(GLIK[get_local_id(0)*tile_dim + get_local_id(1)] > GLIK[get_local_id(0)*tile_dim + inner_k] + GLKK[inner_k*tile_dim + get_local_id(1)])"
"						GLIK[get_local_id(0)*tile_dim + get_local_id(1)] = GLIK[get_local_id(0)*tile_dim + inner_k] + GLKK[inner_k*tile_dim + get_local_id(1)];"
"					if(GLKJ[get_local_id(0)*tile_dim + get_local_id(1)] > GLKK[get_local_id(0)*tile_dim + inner_k] + GLKJ[inner_k*tile_dim + get_local_id(1)])"
"						GLKJ[get_local_id(0)*tile_dim + get_local_id(1)] = GLKK[get_local_id(0)*tile_dim + inner_k] + GLKJ[inner_k*tile_dim + get_local_id(1)];"
"					barrier(CLK_LOCAL_MEM_FENCE);"
"				}"
"				G[(block_r*tile_dim + get_local_id(0))*n + (k*tile_dim + get_local_id(1))] = GLIK[index];"
"				barrier(CLK_LOCAL_MEM_FENCE);"
"				G[(k*tile_dim + get_local_id(0))*n + (block_r*tile_dim + get_local_id(1))] = GLKJ[index];"
"				barrier(CLK_LOCAL_MEM_FENCE);"
"			}"
"		}"
"	}"
"}";

std::string tiled_fw_stage_3 =
"void kernel tiled_fw_stage_3(global int* G, const int k, const int n, const int ngroups, const int tile_dim, local int* GLIK, local int* GLKJ, local int* GLIJ){"
"	int nblocks = n*n/(tile_dim*tile_dim);"
"	int blocks_per_side = n/tile_dim;"
"	int blocks_per_group = nblocks/ngroups;"
"	int group_index = get_group_id(0);"
"	int index = get_local_id(0)*tile_dim + get_local_id(1);"
"	int base_index = group_index*blocks_per_group;"
"	for(int i = base_index; i < base_index + blocks_per_group; i++){"
"		int block_r = i/blocks_per_side;"
"		int block_c = i%blocks_per_side;"
"			GLIK[index] = G[(block_r*tile_dim + get_local_id(0))*n + (k*tile_dim + get_local_id(1))];"
"			barrier(CLK_LOCAL_MEM_FENCE);"
"			GLKJ[index] = G[(k*tile_dim + get_local_id(0))*n + (block_c*tile_dim + get_local_id(1))];"
"			barrier(CLK_LOCAL_MEM_FENCE);"
"			GLIJ[index] = G[(block_r*tile_dim + get_local_id(0))*n + (block_c*tile_dim + get_local_id(1))];"
"			barrier(CLK_LOCAL_MEM_FENCE);"
"			for(int inner_k = 0; inner_k < tile_dim; inner_k++){"
"				if(GLIJ[get_local_id(0)*tile_dim + get_local_id(1)] > GLIK[get_local_id(0)*tile_dim + inner_k] + GLKJ[inner_k*tile_dim + get_local_id(1)])"
"					GLIJ[get_local_id(0)*tile_dim + get_local_id(1)] = GLIK[get_local_id(0)*tile_dim + inner_k] + GLKJ[inner_k*tile_dim + get_local_id(1)];"
"				barrier(CLK_LOCAL_MEM_FENCE);"
"			}"
"			G[(block_r*tile_dim + get_local_id(0))*n + (block_c*tile_dim + get_local_id(1))] = GLIJ[index];"
"			barrier(CLK_LOCAL_MEM_FENCE);"

"	}"
"}";

void tiled_fw(int* G, int n, int nthreads, int tile_dim, cl::Device default_device, cl::Context context, std::ofstream& myfile) {
	cl::Program::Sources sources;
	sources.push_back({ tiled_fw_stage_1.c_str(), tiled_fw_stage_1.length() });
	sources.push_back({ tiled_fw_stage_2.c_str(), tiled_fw_stage_2.length() });
	sources.push_back({ tiled_fw_stage_3.c_str(), tiled_fw_stage_3.length() });

	cl::Program program(context, sources);
	if (program.build({ default_device }) != CL_SUCCESS) {
		std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
		int c;
		std::cin >> c;
		exit(1);
	}

	cl::Buffer buffer_G(context, CL_MEM_READ_WRITE, sizeof(int) * n*n);

	cl::CommandQueue queue(context, default_device);

	cl::Kernel tiled_fw_stage_1_kernel(program, "tiled_fw_stage_1");
	cl::Kernel tiled_fw_stage_2_kernel(program, "tiled_fw_stage_2");
	cl::Kernel tiled_fw_stage_3_kernel(program, "tiled_fw_stage_3");

	int blocks_per_side = n / tile_dim;
	int ngroups = nthreads / (tile_dim*tile_dim);

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	queue.enqueueWriteBuffer(buffer_G, CL_TRUE, 0, sizeof(int) * n*n, G);
	for (int k = 0; k < blocks_per_side; k++) {
		tiled_fw_stage_1_kernel.setArg(0, buffer_G);
		tiled_fw_stage_1_kernel.setArg(1, sizeof(int), &k);
		tiled_fw_stage_1_kernel.setArg(2, sizeof(int), &n);
		tiled_fw_stage_1_kernel.setArg(3, sizeof(int), &tile_dim);
		tiled_fw_stage_1_kernel.setArg(4, sizeof(int)*tile_dim*tile_dim, NULL);

		queue.enqueueNDRangeKernel(tiled_fw_stage_1_kernel, cl::NullRange, cl::NDRange(tile_dim, tile_dim), cl::NDRange(tile_dim, tile_dim));

		int nlgroups = (ngroups > blocks_per_side) ? blocks_per_side : ngroups;
		tiled_fw_stage_2_kernel.setArg(0, buffer_G);
		tiled_fw_stage_2_kernel.setArg(1, sizeof(int), &k);
		tiled_fw_stage_2_kernel.setArg(2, sizeof(int), &n);
		tiled_fw_stage_2_kernel.setArg(3, sizeof(int), &nlgroups);
		tiled_fw_stage_2_kernel.setArg(4, sizeof(int), &tile_dim);
		tiled_fw_stage_2_kernel.setArg(5, sizeof(int)*tile_dim*tile_dim, NULL);
		tiled_fw_stage_2_kernel.setArg(6, sizeof(int)*tile_dim*tile_dim, NULL);
		tiled_fw_stage_2_kernel.setArg(7, sizeof(int)*tile_dim*tile_dim, NULL);

		queue.enqueueNDRangeKernel(tiled_fw_stage_2_kernel, cl::NullRange, cl::NDRange(nlgroups*tile_dim, tile_dim), cl::NDRange(tile_dim, tile_dim));

		tiled_fw_stage_3_kernel.setArg(0, buffer_G);
		tiled_fw_stage_3_kernel.setArg(1, sizeof(int), &k);
		tiled_fw_stage_3_kernel.setArg(2, sizeof(int), &n);
		tiled_fw_stage_3_kernel.setArg(3, sizeof(int), &ngroups);
		tiled_fw_stage_3_kernel.setArg(4, sizeof(int), &tile_dim);
		tiled_fw_stage_3_kernel.setArg(5, sizeof(int)*tile_dim*tile_dim, NULL);
		tiled_fw_stage_3_kernel.setArg(6, sizeof(int)*tile_dim*tile_dim, NULL);
		tiled_fw_stage_3_kernel.setArg(7, sizeof(int)*tile_dim*tile_dim, NULL);

		queue.enqueueNDRangeKernel(tiled_fw_stage_3_kernel, cl::NullRange, cl::NDRange(ngroups * tile_dim, tile_dim), cl::NDRange(tile_dim, tile_dim));
	}
	queue.enqueueReadBuffer(buffer_G, CL_TRUE, 0, sizeof(int) * n*n, G);

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	myfile << "Time difference(" << n << ", " << nthreads << ", " << tile_dim << ") = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
	queue.flush();
	if (DEBUG) {
		std::cout << " result: \n";
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++)
				std::cout << G[i * n + j] << " ";
			std::cout << "\n";
		}
	}
}

void simple_fw(int* G, int n, int threads_r, int threads_c, cl::Device default_device, cl::Context context, std::ofstream& myfile) {
	

	cl::Program::Sources sources;
	sources.push_back({ simple_block_fw.c_str(), simple_block_fw.length() });

	cl::Program program(context, sources);
	if (program.build({ default_device }) != CL_SUCCESS) {
		std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
		int c;
		std::cin >> c;
		exit(1);
	}

	cl::Buffer buffer_G(context, CL_MEM_READ_WRITE, sizeof(int) * n*n);

	cl::CommandQueue queue(context, default_device);

	cl::Kernel simple_block_fw_kernel(program, "simple_block_fw");

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	queue.enqueueWriteBuffer(buffer_G, CL_TRUE, 0, sizeof(int) * n*n, G);
	for (int k = 0; k < n; k++) {
		simple_block_fw_kernel.setArg(0, buffer_G);
		simple_block_fw_kernel.setArg(1, sizeof(int), &k);
		simple_block_fw_kernel.setArg(2, sizeof(int), &n);
		simple_block_fw_kernel.setArg(3, sizeof(int), &threads_r);
		simple_block_fw_kernel.setArg(4, sizeof(int), &threads_c);

		queue.enqueueNDRangeKernel(simple_block_fw_kernel, cl::NullRange, cl::NDRange(threads_r, threads_c), cl::NullRange);
	}
	queue.enqueueReadBuffer(buffer_G, CL_TRUE, 0, sizeof(int) * n*n, G);

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	myfile << "Time difference(" << n << ", " << threads_r << ", " << threads_c << ") = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
	
	queue.flush();
	if (DEBUG) {
		std::cout << " result: \n";
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++)
				std::cout << G[i * n + j] << " ";
			std::cout << "\n";
		}
	}
}

void main_opencl() {
	//get all platforms (drivers)
	std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);
	if (all_platforms.size() == 0) {
		std::cout << " No platforms found. Check OpenCL installation!\n";
		exit(1);
	}
	cl::Platform default_platform = all_platforms[0];
	std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

	std::vector<cl::Device> all_devices;
	default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
	if (all_devices.size() == 0) {
		std::cout << " No devices found. Check OpenCL installation!\n";
		exit(1);
	}
	cl::Device default_device = all_devices[0];
	std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

	cl::Context context({ default_device });

	std::ofstream myfile;
	myfile.open("OpenCLTiledSquare.txt");
	for (int n = 1024; n <= 1024; n *= 2) {
		for (int tile_dim = 32; tile_dim <= 32; tile_dim+=2) {
			//int max_tile = (sqrt(threads) < 32) ? sqrt(threads) : 32;
			for (int threads = 1; threads <= 1; threads*=2) {
					//std::cout << n << " " << threads << " " << tile_dim << std::endl;
					std::mt19937 rng;
					rng.seed(std::random_device()());
					std::uniform_int_distribution<std::mt19937::result_type> dist(1, n);
					int* graph = new int[n*n];
					for (int i = 0; i < n; i++)
						for (int j = 0; j < n; j++)
							if (i == j) graph[i*n + j] = 0;
							else graph[i*n + j] = dist(rng);

							tiled_fw(graph, n, tile_dim*tile_dim, tile_dim, default_device, context, myfile);
			}
		}
	}
	myfile.close();
}