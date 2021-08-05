#if NCNN_MPI
#include <mpi.h>
#include <memory>
#include <iostream>
#endif
#include <omp.h>

#include "net.h"
#include "cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <sched.h>


#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include <stdio.h>
#include <vector>
#include <stdio.h>




int main(int argc, char** argv) {
  int numprocs, rank, namelen;
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int iam = 0, np = 2;
    int i, online=0;
  ulong ncores = sysconf(_SC_NPROCESSORS_CONF);
  cpu_set_t *setp = CPU_ALLOC(ncores);
  ulong setsz = CPU_ALLOC_SIZE(ncores);

  CPU_ZERO_S(setsz, setp);

  if (sched_getaffinity(0, setsz, setp) == -1) {
    perror("sched_getaffinity(2) failed");
    exit(errno);
  }

  for (i=0; i < CPU_COUNT_S(setsz, setp); i++) {
    if (CPU_ISSET_S(i, setsz, setp))
      online++;
  }

  printf("%d cores configured, %d cpus allowed in affinity mask\n", ncores, online);
  CPU_FREE(setp);
   ncnn::set_cpu_powersave(0);
    ncnn::set_omp_num_threads(1);
  omp_set_num_threads(1);

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Get_processor_name(processor_name, &namelen);


    ncnn::Layer* test_mylayer = ncnn::create_layer("MyLayer");

#pragma omp parallel default(shared) private(iam, np)
  {
    np = omp_get_num_threads();
    iam = omp_get_thread_num();
    printf("Hello from thread %d out of %d from process %d out of %d on %s\n",
           iam, np, rank, numprocs, processor_name);
  }

  MPI_Finalize();


//    ncnn::set_cpu_powersave(0);
//
//
//    // Initialize the MPI environment
//    MPI_Init(NULL, NULL);
//
//    // Get the number of processes
//    int world_size;
//    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
//
//    // Get the rank of the process
//    int world_rank;
//    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
//
//    // Get the name of the processor
//    char processor_name[MPI_MAX_PROCESSOR_NAME];
//    int name_len;
//    MPI_Get_processor_name(processor_name, &name_len);
//
//    // Print off a hello world message
//    printf("Hello world from processor %s, rank %d out of %d processors\n",
//            processor_name, world_rank, world_size);
//
//    ncnn::ParamDict pd;
//    //pd.set(0, 1); //Pooling type = avg
//    //pd.set(4, 1);// Global avg pooling
//    //test_mylayer->load_param(pd);
//
//    //test_mylayer->create_pipeline(opt);
//    // Finalize the MPI environment.
//    MPI_Finalize();
}

