# -*- coding: utf-8 -*-
import os
import time
import random
import sys
import json
#### mongo db part

def save_json(vod ,output_file):
    movie_content = json.dumps(vod)
    jsonFile = open(output_file, "w")
    jsonFile.write(movie_content)
    jsonFile.close()
    print (str(output_file)+".json saved.")
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

