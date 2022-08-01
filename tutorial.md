# Check MPI installation & Cluster Env
## Hello world for mpirun
Before running into multi-node or single-node-multi-threads programs, we need to check if our MPI environments ready or not
    
      // helloworld.c
      #include <mpi.h>
      #include <stdio.h>

      int main(int argc, char** argv) {
      // Initialize the MPI environment
      MPI_Init(NULL, NULL);

      // Get the number of processes
      int world_size;
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);

      // Get the rank of the process
      int world_rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

      // Get the name of the processor
      char processor_name[MPI_MAX_PROCESSOR_NAME];
      int name_len;
      MPI_Get_processor_name(processor_name, &name_len);

      // Print off a hello world message
      printf("Hello world from processor %s, rank %d out of %d processors\n",
             processor_name, world_rank, world_size);

      // Finalize the MPI environment.
      MPI_Finalize();
     }
     
    (shell)$ mpicc helloworld.c -o helloworld && mpirun -np 2 ./helloworld
    
And if you have multiple devices, please check the official [MPI tutorials](https://mpitutorial.com/tutorials/running-an-mpi-cluster-within-a-lan/). It introduces how to build a local LAN cluster with MPI. For super cluster like Slurm, PBS, DAS-5, please look for their own official instructions for MPI configurations.

# Download ONNX models & Specify Mappings
In this tutorial, we choose and [download AlexNet](https://github.com/onnx/models/blob/main/vision/classification/alexnet/model/bvlcalexnet-9.onnx) for a simple demo to show how to define Mapping Specification.
We can find more [onnx models](https://github.com/onnx/models) and download them according to your peferences.
Please note: we only test for opset version-9 onnx models.
For details step by step, check the jupyter notebook .[alexnet partition](tools/distributed/vertical/vertical\ partition\ tutorial.ipynb)
