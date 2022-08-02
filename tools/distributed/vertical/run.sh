#!/bin/sh
cp ../../../build/tools/onnx/onnx2ncnn .
cp synset_words.txt ./models/
cp dog.jpg ./models/
# Generate Sub-models
echo "Generated Sub-models"
python3 interface.py  && cp models/multinode.cpp ../../../examples/ 
# Compile
echo "compile cpp file into executable binary (./multinode)"
cd ../../../build/ && make -j6 && cp examples/multinode* ../tools/distributed/vertical/models/
cd ../tools/distributed/vertical/

#python3 onnx_ncnn.py $1 $2 $3
for i in `ls ./models/*.onnx`; do
    ./onnx2ncnn $i
done
#cd models/ && mpirun -rf rankfile ./multinode dog.jpg

