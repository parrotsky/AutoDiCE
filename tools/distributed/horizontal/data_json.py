# -*- coding: utf-8 -*-
import argparse
import os
import time
import random
import sys
import json
import csv
#### mongo db part

def save_json(vod ,output_file):
    movie_content = json.dumps(vod)
    jsonFile = open(output_file, "w")
    jsonFile.write(movie_content)
    jsonFile.close()
    #print (str(output_file)+".json saved.")

def print_json(input_file):
    fileObject = open(input_file, "r")
    jsonContent = fileObject.read()
    vod = json.loads(jsonContent)
    print (vod)

def load_json(input_file):
    fileObject = open(input_file, "r")
    jsonContent = fileObject.read()
    vod = json.loads(jsonContent)
    return vod

def generate_pareto_json(input_dir='./Result'):
    res = []
    header = ["Energy (mJ)", "Performance (ms)", "Memory (MB)"]
    res.append(header)

    with open(input_dir+"ObjV.csv") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            #print (row)
            res.append(row)
    save_json(res, 'result.json')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Pareto inputdir of Result ")
    args = parser.parse_args()
    generate_pareto_json(args.input_dir)

