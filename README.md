#Parallel Computing Project
##Description
This is a simple project that was done by me for my exam in Parallel Computing.
It consists of various solutions to the All-Pairs Shorthest Path problem, implemented using various 
Parallel Computing frameworks/paradigms:

* OpenMP
* OpenCL
* MPI

For each paradigm multiple implementations have been produced, these implementations are set up such
that is simple to conduct simple performance test in order to measure the scaling of the parallel solutions.

##Requirements
* As regards OpenMP you only need a compiler which supports OpenMP 2.0+
* As regards MPI, I used the Microsoft MPI implementation; however there should be no problem in using
MPICH;
* As regards OpenCL, I tested the program with implementations by both AMD and Intel and you need to
install the SDK for your specific GPU.
