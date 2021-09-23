#!/bin/sh
cp synset_words.txt ./models/
python3 onnx_ncnn.py $1 $2 $3
for i in `ls ./models/*.onnx`; do
    ./onnx2ncnn $i
done
