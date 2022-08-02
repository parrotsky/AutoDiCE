# Check MPI installation & Cluster Env
## Hello world for mpirun
Before running into multi-node or single-node-multi-threads programs, we need to check if our MPI environment is installed correctly. Copy the following C program into the file `helloworld.c`:

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
     
Then compile and run it with the command:

    (shell)$ mpicc helloworld.c -o helloworld && mpirun -np 2 ./helloworld

    
And if you have multiple devices, please check the official MPI tutorials. It introduces how to build a local LAN cluster with MPI. For a supercluster like Slurm, PBS, DAS-5, we refer to their official instructions for MPI configurations.

# Download ONNX models & Specify Mappings

In this tutorial, you need to download [AlexNet](https://github.com/onnx/models/blob/main/vision/classification/alexnet/model/bvlcalexnet-9.onnx) for a simple demo to show how to define the Mapping Specification.
You can find more [onnx models](https://github.com/onnx/models) and download them according to your preferences.
Please note: we only test for opset version-9 onnx models.

1. Install the python package `onnxruntime`. Either using conda:

    ```
        (shell) $ conda install onnxruntime
    ```
    Or using pip3:
    ```
        (shell) $ pip3 install onnxruntime
    ```
   You might also need to install `onnx` in the same way if you haven't already.
    Then copy the AlexNet onnx file:
    ```
        (shell) $ cd ./tools/distributed/vertical/ && cp ~/(Downloaded Dir)/bvlcalexnet-9.onnx .
    ```    

2. Now we can define the Mapping Specification. We want to use two cpu cores in this demo to simulate a multi-node scenario. For this example, the hostname of machine is "lenovo". We define the two keys in the [mapping.json](https://github.com/parrotsky/AutoDiCE/blob/main/tools/distributed/vertical/mapping.json) file as: "lenovo_cpu0"  and "lenovo_cpu1".
Important: Modify the mapping.json file according to the hostname of your machine. Make sure to replace both occurrences of "lenovo" with the output of the `hostname` command. Then we can generate two submodels according to our mapping specification file:

```    
     (shell) $ ./run.sh
```
    
3. In the run.sh script, we copy the cpp file to the directory (./examples/) and compile it into an executable binary. 
4. Deploy the executable file with its submodels into multiple edge devices. 
5. Now we can finally use the `mpirun` command to run the multi-node inference application.
```    
    (shell) $ cd models/ && mpirun -rf rankfile ./multinode dog.jpg
    215 = 0.593469
    207 = 0.125113
    213 = 0.102682
    Brittany spaniel 
    golden retriever 
    Irish setter, red setter 
```    
    
For more information we refer to the jupyter notebook [alexnet partition](https://github.com/parrotsky/AutoDiCE/blob/main/tools/distributed/vertical/vertical%20partition%20tutorial.ipynb). It shows how to generate multi-node inference c++ code file and corresponding sub-models in details.
