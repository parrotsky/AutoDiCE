# -*- coding: utf-8 -*-
from data_json import *
import numpy as np
from onnx_split import *
from code_generator import CppFile
from cpp_generator import *
import itertools as it
import onnx
from onnx import helper, checker
from onnx import TensorProto
import multiprocessing as mp
from multiprocessing import Process, Pool
import os, psutil
from copy import deepcopy

class ComputingNode:
    def __init__(self, **node_details):
# name; inbuffs; receiver; outbuffs; sender ; hardware ;number;cores; dist_layers
        self.__dict__.update(node_details)

class Interface:
    def __init__(self, **input_specs):
        self.__dict__.update(input_specs)
        # {'nx01_arm0123': ['conv1_1', 'conv1_2', 'norm1_1', 'pool1_1'], ....}
        # dictionary {layer_name: attributes, weights}
        self.nodes = list(self.mappings.keys())
        # ['nx01_arm0123','nx01_gpu',...]
        nodes_list = []
        self.computingnodes = {}
        total_gpu  = 0
        total_cpu = 0
        for i in range(len(self.nodes)):
            device = self.nodes[i].split('_')[0]
            resource = self.nodes[i].split('_')[1]
            nodes_list.append(device)
            num = 0
            if "gpu" in resource:
                hardware = 'gpu'
                cores = '5'
                num = 1
                total_gpu = total_gpu + num
            if "arm" in resource:
                hardware = 'cpu'
                cores = ",".join(resource.replace("arm",""))
                num = len(cores.split(','))
                total_cpu = total_cpu + num

            if "cpu" in resource:
                hardware = 'cpu'
                cores = ",".join(resource.replace("cpu",""))
                num = len(cores.split(','))
                total_cpu = total_cpu + num

            self.computingnodes[self.nodes[i]] = ComputingNode(
                        name=self.nodes[i].split('_')[0],
                        inbuffs = [],
                        receiver = {},
                        outbuffs = [],
                        sender = {},
                        hardware = hardware,
                        number = num,
                        cores = cores,
                        dist_layers = deepcopy(self.mappings[self.nodes[i]]))
            #print (self.computingnodes[self.nodes[i]].__dict__)
        ### self.computingnodes.
        print ("Total GPU: %d, Total CPU: %d" %(total_gpu, total_cpu))
        self.devices = list(set(nodes_list))
        # ['nx01', 'nx02', ...]
        self.outputs = []
        self.InitialAttributes()
        self.inputs = getInputlayers(self.model)
        init_model = onnx.load(self.model)
        self.opset_version = init_model.opset_import[0].version
        #)self.GenerateClose()

    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = ObjectDict(value)
        return value
    def InitialAttributes(self):
        self.graph = load_onnx(self.model)
        for i in self.graph.output:
            self.outputs.append(i.name)
        self.value_map = generate_node_dict(self.graph.value_info)
        self.layers = generate_node_dict(self.graph.node)



    def HorizontalInplace(self, horizontal_file):
        split_way = {"lop":-1,"lip":0,"height":1, "width":2}
        split_name = {"lop":"_splitco_","lip":"_splitic_","height":"_splith_","width":"_splitw_"}
        #concatsum = {"lop":"_oconcat","lip":"_hsum","height":"_hconcat","width":"_wconcat"}
        horizontal_spec = load_json(horizontal_file)
        new_model = "horizontal.onnx" 
        for layer, attribute in horizontal_spec.items():
            split_ranks = []
            # attr[2] ---> platforms.
            for k in attribute[2]:
                split_ranks.append(self.nodes.index(k))
                self.mappings[k] = list(map(lambda x: x.replace(layer, layer + split_name[attribute[0]] + str(self.nodes.index(k))), self.mappings[k]))

            # TODO: FIX THIS BECAUSE THE VARIABLE MIGHT NOT BE SET
            # this is a problem when splitting the first layer
            # layer_input_node variable is used for splits other than lop
            try:
                layer_inputs = self.graph.node[find_node_index(self.graph, layer)].input
                layer_input_node = [n for n in layer_inputs if n in self.layers][0]
            except Exception as _:
                print("Exception in finding layer_input_node")
                pass

            next_layer = [n for n in self.layers if layer in self.graph.node[find_node_index(self.graph, n)].input]

            layer_output_node = None
            for n in self.layers:
                if layer in self.graph.node[find_node_index(self.graph, n)].input:
                    layer_output_node = n
                    break
            
            if layer_output_node is None:
                layer_idx = find_node_index(self.graph, layer)
                for n in self.layers:
                    if n in self.graph.node[layer_idx].input:
                        layer_output_node = n
                        break

            new_mapping = {}
            for k, v in self.mappings.items():
                if layer_output_node in v:
                    self.mappings[k].append(str(layer))
                    break

            if split_way[attribute[0]] >=0:
                for k, v in self.mappings.items():
                    if layer_input_node in v:
                        self.mappings[k].append(layer+"_hsplit_"+ str(split_ranks[0]))
                        break
            
            self.graph = horizontal_split(self.graph, layer, split_ranks, split_way[attribute[0]], attribute[1])
            self.layers = generate_node_dict(self.graph.node)
            
            model = helper.make_model(self.graph, producer_name='AutoDiCE', opset_imports=[helper.make_opsetid("", self.opset_version)])
            onnx.save(model, "modified.onnx")
            self.model = "modified.onnx"
            self.inputs = getInputlayers(self.model)
            self.layers = generate_node_dict(self.graph.node)
            self.value_map = generate_node_dict(self.graph.value_info)

            for i in self.graph.output:
                self.outputs.append(i.name)

         
        
            





    def ModelSplit(self):
        # Split model into sub-models.
        #process = psutil.Process(os.getpid())
        #currentmemory = process.memory_info().rss
        #availablememory = psutil.virtual_memory().available
        ##print (availablememory/ currentmemory, currentmemory)
        #poollimit = (int) (availablememory / (currentmemory*2))

        #splitjobs = []
        #poolcurrent = 0
        #for i in range(len(self.nodes)):
        #    splitjobs.append(Process(target=onnx_extract, args=(self.model, './models/'+self.nodes[i]+'.onnx', self.mappings[self.nodes[i]])))
        #    splitjobs[i].start()
        #    poolcurrent = poolcurrent + 1
        #    if poolcurrent>=poollimit or i == len(self.nodes)-1:
        #        for j in range(i-poolcurrent+1, i+1):
        #            #print (j)
        #            splitjobs[j].join()
        #        poolcurrent = 0
        for i in range(len(self.nodes)):
            onnx_extract(self.model, './models/'+self.nodes[i]+'.onnx', self.mappings[self.nodes[i]], self.opset_version)
        # print ("Generate ", (i+1), " Sub-Models.")


    def GenerateComm(self):
        #Mar.06 2022 ""DUMMY cause error in communication
        #
        ### add multi-output into dist_layer list.
        for i in range(len(self.nodes)):
            for each_node in self.mappings[self.nodes[i]]:
                for j in self.layers[each_node].output:
                    if j not in self.computingnodes[self.nodes[i]].dist_layers:
                        self.computingnodes[self.nodes[i]].dist_layers.append(j)

        for i in range(len(self.nodes)):
            input_buffers =  getInputlayers('./models/'+self.nodes[i]+'.onnx')
            self.computingnodes[self.nodes[i]].inbuffs = input_buffers

            for j in range(len(self.nodes)):
                if j==i:
                    continue
                else:
                    for input_buff in input_buffers:
                        if input_buff in self.computingnodes[self.nodes[j]].dist_layers:
                            self.computingnodes[self.nodes[i]].receiver.setdefault(input_buff,[]).append(j)

        receiver_dict_list = {}
        with open("./models/receiver.json", 'w') as rf:
            for i in range(len(self.nodes)):
                receiver_dict_list[self.nodes[i]] = self.computingnodes[self.nodes[i]].receiver
            receiver_dict_content = json.dumps(receiver_dict_list)
            rf.write(receiver_dict_content)

            ### input buffers for each sub-model
            #print ("In: ", self.computingnodes[self.nodes[i]].inbuffs)
        for i in range(len(self.nodes)):
            j = 0
            for j in range(len(self.nodes)):
                if j == i:
                    continue
                else:
                    input_buffers = self.computingnodes[self.nodes[j]].inbuffs
                    for input_buff in input_buffers:
                        if input_buff in self.computingnodes[self.nodes[i]].dist_layers:
                            #### send input_buff from i to j.
                            #### sender
                            self.computingnodes[self.nodes[i]].sender.setdefault(input_buff, []).append(j)
                            if input_buff not in self.computingnodes[self.nodes[i]].outbuffs:
                                self.computingnodes[self.nodes[i]].outbuffs.append(input_buff)
        for i in range(len(self.nodes)):
            for j in load_onnx('./models/'+self.nodes[i]+'.onnx').output:
                self.computingnodes[self.nodes[i]].outbuffs.append(j.name)

        for i in range(len(self.nodes)):
            orderoutbuffs = []
            ### re order output buffer according to original model
            ### 重新排序输出的buffer
            for layername in self.layers.keys():
                #temp_output_buffs = (self.computingnodes[self.nodes[i]].outbuffs).copy()
                #for each_node in self.layers[layername].outputs:
                #    temp_output_buffs.append(each_node)

                if layername in self.computingnodes[self.nodes[i]].outbuffs:
                #if layername in temp_output_buffs:
                    orderoutbuffs.append(layername)
                else:
                    for each_node in self.layers[layername].output:
                        if each_node in self.computingnodes[self.nodes[i]].outbuffs:
                            orderoutbuffs.append(each_node)
                            break


            self.computingnodes[self.nodes[i]].outbuffs = orderoutbuffs

        ### Output buffers for each sub-model
        sender_dict_list = {}
        with open("./models/sender.json", 'w') as rf:
            for i in range(len(self.nodes)):
                sender_dict_list[self.nodes[i]] = self.computingnodes[self.nodes[i]].sender
            sender_dict_content = json.dumps(sender_dict_list)
            rf.write(sender_dict_content)

    def GenerateRankFile(self):
        # generate essential rankfile.
        with open("./models/rankfile", 'w') as rf:
            devices_list = []
            for i in range(len(self.computingnodes)):
                device = self.computingnodes[self.nodes[i]].name
                devices_list.append(device)
                cores = self.computingnodes[self.nodes[i]].cores
                rf.write('rank '+ str(i) + '=' + device+'    slots=' + cores +'\n')
            devices_list = set(devices_list)
            # print ("Devices: %d" % len(devices_list))

        with open("./models/hostfile", 'w') as hs:
            for i in range(len(self.platforms)):
                hs.write(str(self.platforms[i])+'\n')



    def ConsistencyCheck(self):
        # Check model is consistent with mapping.
        mapping_layers_list =[]
        for key, value in self.mappings.items():
            for layer_name in value:
                mapping_layers_list.append(layer_name)
        mapping_layers = set(mapping_layers_list)
        node_map = generate_node_dict(self.graph.node)
        node_map_list = set (list (node_map))
        if (node_map_list ^ mapping_layers):
            print ("Consistency Check Fail.")
        else:
            print ("Consistency Check Pass.")

        if (mapping_layers - node_map_list):
            print ("Original model doesn't contain: ", mapping_layers - node_map_list)
        if (node_map_list - mapping_layers):
            print ("Given mapping file require nodes: ", node_map_list - mapping_layers)

class EngineCode():
    def __init__(self, **intermediate):
        self.__dict__.update(intermediate)
        if (self.Benchmark):
            self.cpp = CppFile(self.CppName + "_bench.cpp")
        else:
            self.cpp = CppFile(self.CppName + ".cpp")
        self.commtag = {}
        tag = 0
        for i in range(len(self.NodesList)):
            engine = self.NodesList[i]
            for output_buff in self.ComputingNodes[engine].outbuffs:
                if output_buff not in self.Outputs:
                    for j in self.ComputingNodes[engine].sender[output_buff]:
                        self.commtag[(i, output_buff, j)] = tag
                    #### i send output_buff to j
                        tag = tag + 1
            tag = tag + 10
        print (self.commtag)
        self.GenerateHeader(self.cpp)
        self.GenerateBody(self.cpp)
        self.GenerateMain(self.cpp)
        self.cpp.close()

    def GenerateHeader(self, cpp):
        cpp("#include \"net.h\"")
        cpp("#include <algorithm>")
        cpp("#include <opencv2/core/core.hpp>")
        cpp("#include <opencv2/highgui/highgui.hpp>")
        cpp("#include \"benchmark.h\"")
        cpp("#include \"cpu.h\"")
        cpp("#include \"gpu.h\"")
        cpp("#include <stdio.h>")
        cpp("#include <vector>")
        cpp("#include <mpi.h>")
        cpp("#include <stdlib.h>")
        cpp("#include <errno.h>")
        cpp("#include <unistd.h>")

        cpp("#include <sys/types.h>")
        cpp("#include <sys/stat.h>")
        cpp("#include <fcntl.h>")

        cpp("static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;")
        cpp("static ncnn::PoolAllocator g_workspace_pool_allocator;")

        cpp("static int load_labels(std::string path, std::vector<std::string>& labels)")
        cpp("{       ") 
        cpp("    FILE* fp = fopen(path.c_str(), \"r\");") 
        cpp("    while (!feof(fp))") 
        cpp("    {       ") 
        cpp("        char str[1024];") 
        cpp("        fgets(str, 1024, fp); ")  
        cpp("        std::string str_s(str);") 
        cpp("        if (str_s.length() > 0)") 
        cpp("        {   ") 
        cpp("            for (int i = 0; i < str_s.length(); i++)") 
        cpp("            {   ") 
        cpp("                if (str_s[i] == ' ')") 
        cpp("                {   ") 
        cpp("                    std::string strr = str_s.substr(i, str_s.length() - i - 1);") 
        cpp("                    labels.push_back(strr);") 
        cpp("                    i = str_s.length();") 
        cpp("                }") 
        cpp("            }") 
        cpp("        }   ") 
        cpp("    }       ") 
        #cpp("    fclose(fp);")
        cpp("    return 0;") 
        cpp("}  ") 
        cpp("\n") 
        cpp("//static int print_topk(const std::vector<float>& cls_scores, int topk)") 
        cpp("static int print_topk(const std::vector<float>& cls_scores, int topk, std::vector<int>& index_result,")
        cpp("             std::vector<float>& score_result)") 
        cpp("{")
        cpp("    // partial sort topk with index")
        cpp("    int size = cls_scores.size();")
        cpp("    std::vector<std::pair<float, int>> vec;")
        cpp("    vec.resize(size);")
        cpp("    for (int i = 0; i < size; i++)")
        cpp("    {")
        cpp("        vec[i] = std::make_pair(cls_scores[i], i);")
        cpp("    }")
        cpp("")
        cpp("    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),")
        cpp("                      std::greater<std::pair<float, int> >());")
        cpp("\n")
        cpp("    // print topk and score")
        cpp("    for (int i = 0; i < topk; i++)")
        cpp("    {")
        cpp("        float score = vec[i].first;")
        cpp("        int index = vec[i].second;")
        cpp("        fprintf(stderr, \"%d = %f\\n\", index, score);")
        cpp("        index_result.push_back(index);")
        cpp("        score_result.push_back(score);")
        cpp("    }")
        cpp("    return 0;")
        cpp("}")



        cpp.newline(1)
    def GenerateBody(self, cpp):
        if (self.Benchmark):
            self.MemoryTest(self.cpp)
            #self.PowerTest(self.cpp)
        self.ReadInput(self.cpp)
        for i in range(len(self.NodesList)):
            self.GenerateEngine(self.cpp, i)
        self.MultiClassify(self.cpp)

    def PowerTest(self, cpp):
        cpp("inline bool FileExists( const std::string& Filename ) ")
        cpp("{")
        cpp("    struct stat buffer;")
        cpp("return (stat (Filename.c_str(), &buffer) == 0);")
        cpp("}")
        cpp.newline(1)
        cpp("void powerstart(int world_rank)")
        cpp("{")
        cpp("std::string filename = std::to_string(world_rank) + \"_sfifo\";")
        cpp("const char* sfile = filename.c_str();")
        #cpp("if(FileExists(filename)) {")
        #cpp("printf(\"Power Fifo Start Existed.\\n\");")
        #cpp("}else{")
        #cpp("int namepipe = mkfifo(sfile, S_IFIFO|0666);")
        #cpp("//创建一个存取权限为0666的命名管道")
        #cpp("if(namepipe == -1){perror(\"mkfifo error\");exit(1);}")
        #cpp("printf(\"FiFo Created.\\n\");")
        #cpp("}")
        cpp.newline(1)
        cpp("int fd=open(sfile, O_RDWR);")
        #cpp("int fd=open(sfile, O_WRONLY);")
        cpp("//打开该命名管道")
        cpp("if(fd == -1){perror(\"open error\");exit(2);}")
        cpp("char buf[1]={'s'};")
        cpp("if(write(fd,buf,1) == -1){")
        cpp("//把该消息写入到命名管道中")
        cpp("perror(\"write error\");")
        cpp("exit(3);")
        cpp("}")
        cpp("close(fd);")
        cpp("}")
        cpp.newline(2)
        cpp("void powerend(int world_rank)")
        cpp("{")
        cpp("std::string filename = std::to_string(world_rank) + \"_sfifo\";")
        cpp("const char* sfile = filename.c_str();")
        #cpp("if(FileExists(filename)) {")
        #cpp("printf(\"Power Measure End Fifo existed.\\n\");")
        #cpp("}else{")
        #cpp("int namepipe=mkfifo(sfile, S_IFIFO|0666);")
        #cpp("//创建一个存取权限为0666的命名管道")
        #cpp("if(namepipe == -1){perror(\"mkfifo error\");exit(1);}")
        #cpp("printf(\"FiFo Created.\\n\");")
        #cpp("}")
        cpp.newline(1)
        #cpp("int fd=open(sfile, O_WRONLY);")
        cpp("int fd=open(sfile, O_RDWR);")
        #cpp("int fd = open(sfile, O_RDWR | O_NONBLOCK);")
        cpp("//打开该命名管道")
        cpp("if(fd == -1){perror(\"open error\");exit(2);}")
        cpp("char buf[1]={'q'};")
        cpp("if(write(fd,buf,1) == -1){")
        cpp("//把该消息写入到命名管道中")
        cpp("perror(\"write error\");")
        cpp("exit(3);")
        cpp("}")
        cpp("close(fd);")
        cpp("}")
        cpp.newline(2)


        #cpp("void powerend(int world_rank)")
        #cpp("{")
        #cpp("std::string filename = std::to_string(world_rank) + \"_sfifo\";")
        #cpp("const char* sfile = filename.c_str();")
        #cpp("    int fd;")
        #cpp("    while(1){")
        #cpp("        if( FileExists(filename)) {")
        #cpp("            fd=open(sfile, O_WRONLY|O_NONBLOCK);")
        #cpp("            char buf[1]={'q'};")
        #cpp("            if(write(fd,buf,1) != -1) break;")
        #cpp("        }")
        #cpp("    }")
        #cpp("    close(fd);")
        #cpp("}")
        #cpp.newline(1)


        #cpp("void powerend(int world_rank)")
        #cpp("{")
        #cpp("std::string filename = std::to_string(world_rank) + \"_sfifo\";")
        #cpp("const char* sfile = filename.c_str();")
        #cpp("if(FileExists(filename)) {")
        #cpp("printf(\"Power Measure End Fifo existed.\\n\");")
        #cpp("}else{")
        #cpp("int namepipe=mkfifo(sfile, S_IFIFO|0666);")
        #cpp("//创建一个存取权限为0666的命名管道")
        #cpp("if(namepipe == -1){perror(\"mkfifo error\");exit(1);}")
        #cpp("printf(\"FiFo Created.\\n\");")
        #cpp("}")
        #cpp.newline(1)
        #cpp("int fd = open(sfile, O_RDWR);")
        #cpp("//打开该命名管道")
        #cpp("if(fd == -1){perror(\"open error\");exit(2);}")
        #cpp("char buf[1]={'q'};")
        #cpp("if(write(fd,buf,1) == -1){")
        #cpp("//把该消息写入到命名管道中")
        #cpp("perror(\"write error\");")
        #cpp("exit(3);")
        #cpp("}")
        #cpp("close(fd);")
        #cpp("}")
        #cpp.newline(2)

    def MemoryTest(self, cpp):
        cpp("int parseLine(char* line){")
        cpp("    // This assumes that a digit will be found and the line ends in Kb.")
        cpp("    int i = strlen(line);")
        cpp("    const char* p = line;")
        cpp("    while (*p <'0' || *p > '9') p++;")
        cpp("    line[i-3] = '\0';")
        cpp("    i = atoi(p);")
        cpp("    return i;")
        cpp("}\n")

        cpp("int getVirtual(){ //Note: this value is in KB!")
        cpp("    FILE* file = fopen(\"/proc/self/status\", \"r\");")
        cpp("    int result = -1;")
        cpp("    char line[128];\n")

        cpp("    while (fgets(line, 128, file) != NULL){")
        cpp("        if (strncmp(line, \"VmSize:\", 7) == 0){")
        cpp("            result = parseLine(line);")
        cpp("            break;")
        cpp("        }")
        cpp("    }")
        cpp("    fclose(file);")
        cpp("    return result;")
        cpp("}\n")

        cpp("int getPhysical(){ //Note: this value is in KB!")
        cpp("    FILE* file = fopen(\"/proc/self/status\", \"r\");")
        cpp("    int result = -1;")
        cpp("    char line[128];\n")

        cpp("    while (fgets(line, 128, file) != NULL){")
        cpp("        if (strncmp(line, \"VmRSS:\", 6) == 0){")
        cpp("            result = parseLine(line);")
        cpp("            break;")
        cpp("        }")
        cpp("    }")
        cpp("    fclose(file);")
        cpp("    return result;")
        cpp("}\n")

    def LoadLabels(self, cpp):
        cpp("static int load_labels(std::string path, std::vector<std::string>& labels)")
        cpp("{    ")
        cpp("    FILE* fp = fopen(path.c_str(), \"r\");")
        cpp("     ")
        cpp("    while (!feof(fp))")
        cpp("    {")
        cpp("        char str[1024];")
        cpp("        fgets(str, 1024, fp);  ")
        cpp("        std::string str_s(str);")
        cpp("     ")
        cpp("        if (str_s.length() > 0)")
        cpp("        {")
        cpp("            for (int i = 0; i < str_s.length(); i++)")
        cpp("            {")
        cpp("                if (str_s[i] == ' ')")
        cpp("                {")
        cpp("                    std::string strr = str_s.substr(i, str_s.length() - i - 1);")
        cpp("                    labels.push_back(strr);")
        cpp("                    i = str_s.length();")
        cpp("                }")
        cpp("            }")
        cpp("        }")
        cpp("    }")
        cpp("    return 0;")
        cpp("}    ")

        cpp("//static int print_topk(const std::vector<float>& cls_scores, int topk)")

    def PrintTopK(self, cpp):
        cpp("static int print_topk(const std::vector<float>& cls_scores, int topk, std::vector<int>& index_result, std::vector<float>& score_result)")
        cpp("{   ")
        cpp("    // partial sort topk with index")
        cpp("    int size = cls_scores.size();")
        cpp("    std::vector<std::pair<float, int> > vec;")
        cpp("    vec.resize(size); ")
        cpp("    for (int i = 0; i < size; i++)")
        cpp("    {   ")
        cpp("        vec[i] = std::make_pair(cls_scores[i], i);")
        cpp("    }\n")
        
        cpp("    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),")
        cpp("                      std::greater<std::pair<float, int> >());\n")
        
        cpp("    // print topk and score")
        cpp("    for (int i = 0; i < topk; i++)")
        cpp("    {   ")
        cpp("        float score = vec[i].first;")
        cpp("        int index = vec[i].second;")
        cpp("        fprintf(stderr, \"%d = %f\\n\", index, score);")
        cpp("        index_result.push_back(index);")
        cpp("        score_result.push_back(score);")
        cpp("    }\n")
        
        cpp("    return 0;")
        cpp("}\n")

    def ReadInput(self, cpp):
        cpp("static int read_input(const char* imagepath, ncnn::Mat& in)")
        cpp("{")
        cpp("   cv::Mat m = cv::imread(imagepath, 1);")
        cpp("   if (m.empty())")
        cpp("   {")
        cpp("       fprintf(stderr, \"cv::imread %s failed\\n\", imagepath);")
        cpp("       return -1;")
        cpp("   }")
        cpp("   // m: BGR format.")
        cpp("   in = ncnn::Mat::from_pixels_resize(m.data, ncnn::Mat::PIXEL_BGR, m.cols, m.rows, 224, 224);")
        cpp("   const float mean_vals[3] = {104.f, 117.f, 123.f};")
        cpp("   in.substract_mean_normalize(mean_vals, 0);")
        cpp("   return 0;")
        cpp("}")
        cpp.newline(1)

    def MultiClassify(self, cpp):
        cpp("static int multi_classify(const char* imagepath, std::vector<float>& cls_scores)")
        cpp("{")
        cpp("   int irank; MPI_Comm_rank(MPI_COMM_WORLD, &irank);")
        for i in range(len(self.NodesList)):
            cpp ("  if (irank == " + str(i) + ") {")
            engine = self.NodesList[i]

            for input_buff in self.ComputingNodes[engine].inbuffs:
                if input_buff in self.Inputs:
                    cpp("       ncnn::Mat " + input_buff +";")
                else:
                    buff_shape = get_node_output_shape(self.ValueInfos[input_buff])
                    buff_shape_rs = buff_shape[0:1] + buff_shape[2:4] + buff_shape[1:2]
                    j_shape = str(buff_shape_rs[1:]).replace('[','(').replace(']',')')
                    cpp("       ncnn::Mat "+ str(input_buff)+ j_shape+";")

            for output_buff in self.ComputingNodes[engine].outbuffs:
                if output_buff in self.Outputs:
                    cpp("       ncnn::Mat " + output_buff +";")
                else:
                    buff_shape = get_node_output_shape(self.ValueInfos[output_buff])
                    buff_shape_rs = buff_shape[0:1] + buff_shape[2:4] + buff_shape[1:2]
                    j_shape = str(buff_shape_rs[1:]).replace('[','(').replace(']',')')
                    cpp("       ncnn::Mat "+ str(output_buff)+ j_shape+";")

            if self.Inputs[0] in self.ComputingNodes[engine].inbuffs:
                cpp(" ")
                cpp.append("      model"+str(i)+"_engine(imagepath")
            else:
                cpp(" ")
                cpp.append("      model"+str(i) +"_engine(0")

            for input_buff in self.ComputingNodes[engine].inbuffs:
                cpp.append(", "+input_buff+"")
            for output_buff in self.ComputingNodes[engine].outbuffs:
                cpp.append(", "+output_buff+"")
            cpp.append(');')
            cpp.newline(1)

            for output_buff in self.ComputingNodes[engine].outbuffs:
                if output_buff in self.Outputs:
                    cpp("    cls_scores.resize("+str(output_buff)+".w);\n")
                    cpp("    for (int j = 0; j < "+str(output_buff)+".w; j++)")
                    cpp("    {")
                    cpp("        cls_scores[j] = "+str(output_buff)+"[j];")
                    cpp("    }\n")

                    cpp("std::vector<std::string> labels; \n")
                    cpp("load_labels(\"synset_words.txt\", labels);\n")
                    cpp("std::vector<int> index;\n")
                    cpp("std::vector<float> score;\n")
                    cpp("print_topk(cls_scores, 3, index, score);\n")
                    cpp("   for (int i = 0; i < index.size(); i++)\n")
                    cpp("   {\n")
                    cpp("       fprintf(stderr, \"%s \\n\", labels[index[i]].c_str());\n")
                    cpp("   }\n")       

            cpp("   }")
        cpp("return 0;")
        cpp("}\n")

    def GenerateEngine(self, cpp, i=0):
        # The i_th sub-model generate engine.
        engine = self.NodesList[i]
        enginegraph = onnx.load('./models/'+engine+'.onnx').graph
        engineinput_map = generate_node_dict(enginegraph.input)
        engineoutput_map = generate_node_dict(enginegraph.output)
        engineinitializer_map = generate_node_dict(enginegraph.initializer)
        enginevalue_map = generate_node_dict(enginegraph.value_info)
        enginenode_map = generate_node_dict(enginegraph.node)

        if self.Inputs[0] in self.ComputingNodes[engine].inbuffs:
            cpp("")
            cpp.append("void model" + str(i) + "_engine(")
            cpp.append("const char* imagepath")
        else:
            cpp("")
            cpp.append("void model" + str(i) +"_engine(")
            cpp.append("int index")
        for input_buff in self.ComputingNodes[engine].inbuffs:
            cpp.append(", ncnn::Mat& "+input_buff+"")
        for output_buff in self.ComputingNodes[engine].outbuffs:
            cpp.append(", ncnn::Mat& "+output_buff+"")
        cpp.append('){')
        cpp.newline(1)
        commdict = {}
        index = 0
        for buffer, value in self.ComputingNodes[engine].sender.items():
            for j in value:
                ### i send buffer to j
                commdict[(i, buffer, j)] = index
                index = index + 1

        for buffer, value in self.ComputingNodes[engine].receiver.items():
            for j in value:
                commdict[(j, buffer, i)] = index
                index = index + 1

        cpp("   MPI_Request requests["+ str(len(list(commdict.keys()))) +"];")
        cpp("   MPI_Status status["+ str(len(list(commdict.keys()))) +"];\n")
        cpp("   int irank; MPI_Comm_rank(MPI_COMM_WORLD, &irank);")
        if (self.Benchmark):
            cpp("   std::string memory_pre = std::to_string(irank) + \"_MEMORY.txt\";")
            cpp("   std::string perf_pre = std::to_string(irank) + \"_PERFORMANCE.txt\";")
            cpp("   const char* memory = memory_pre.c_str();")
            cpp("   const char* perf = perf_pre.c_str();")
            cpp("   FILE* pm = fopen(memory, \"wb\");")
            cpp("   FILE* pp = fopen(perf, \"wb\");")

        engine = self.NodesList[i]
        net_name = engine + "_model_" + str(i)
        for input_buff in self.ComputingNodes[engine].inbuffs:
            if input_buff in self.Inputs:
                #cpp("   ncnn::Mat "";")
                cpp("    read_input(imagepath, " + input_buff +");")

        cpp("     ncnn::Net "+ net_name + ";")
        cpp("    "+net_name+".opt.blob_allocator = &g_blob_pool_allocator;")
        cpp("    "+net_name+".opt.workspace_allocator = &g_workspace_pool_allocator;")
        if self.ComputingNodes[engine].hardware == 'cpu':
            cpp("    "+net_name+".opt.use_vulkan_compute = false;")
        if self.ComputingNodes[engine].hardware == 'gpu':
            cpp("    "+net_name+".opt.use_vulkan_compute = true;")
        cpp("    "+net_name+".opt.use_winograd_convolution = true;")
        cpp("    "+net_name+".opt.use_sgemm_convolution = true;")
        cpp("    "+net_name+".opt.use_int8_inference = true;")
        cpp("    "+net_name+".opt.use_fp16_packed = true;")
        cpp("    "+net_name+".opt.use_fp16_storage = true;")
        cpp("    "+net_name+".opt.use_fp16_arithmetic = true;")
        cpp("    "+net_name+".opt.use_int8_storage = true;")
        cpp("    "+net_name+".opt.use_int8_arithmetic = true;")
        cpp("    "+net_name+".opt.use_packing_layout = true;")
        cpp("    "+net_name+".opt.use_shader_pack8 = false;")
        cpp("    "+net_name+".opt.use_image_storage = false;")
        cpp("    "+net_name+".opt.num_threads = " + str(self.ComputingNodes[engine].number) + ";")
        cpp("    ncnn::set_omp_dynamic(0);")
        cpp("    ncnn::set_omp_num_threads("+ str(self.ComputingNodes[engine].number) +");")

        cpp("    "+net_name+".load_param(\""+ engine +".param\");")
        cpp("    "+net_name+".load_model(\""+ engine +".bin\");\n")
        cpp("    ncnn::Extractor ex"+str(i)+" = "+str(net_name)+".create_extractor();\n")

        if (self.Benchmark):
            cpp("    int g_warmup_loop_count = 4;")
            cpp("    int g_loop_count = 20;")
            cpp("// warm up")
            cpp("    for (int i = 0; i < g_warmup_loop_count; i++)")
            cpp("    {")

            for input_buff, source in self.ComputingNodes[engine].receiver.items():
                input_size = str(input_buff)+".total()"
                #if (input_buff not in self.Inputs) and ("DUMMY" not in input_buff):
                if (input_buff not in self.Inputs):
                    recv_id = self.ComputingNodes[engine].receiver[input_buff]
                    request_id = commdict[(source[0], input_buff, i)]
                    cpp("        MPI_Irecv((float* )"+ str(input_buff)+ ", " +input_size+
                        ", MPI_FLOAT, "+str(source[0])+", ")
                    cpp("                    "+str(self.commtag[(source[0],input_buff,i)])+", MPI_COMM_WORLD, &requests[" + str(request_id) +"]);\n")
########    ##### i send output_buff to j

            cpp("        ex"+str(i)+" = "+str(net_name)+".create_extractor();\n")

            already_receive_buffs = []
            for output_buff in self.ComputingNodes[engine].outbuffs:
                node_input_names = []
                node_input_names =  traceUpNodes(enginegraph, output_buff,
                                                 node_input_names, enginenode_map, 0, engineinitializer_map)
                related_input_buffers = []
                for node in node_input_names:
                    if node in list(enginenode_map.keys()):
                        for input_name in enginenode_map[node].input:
                            if input_name in self.ComputingNodes[engine].inbuffs:
                                related_input_buffers.append(input_name)
                #print ("related_input_buffers ",related_input_buffers)
                for input_buff in related_input_buffers:
                    #if (input_buff not in self.Inputs ) and (input_buff not in already_receive_buffs) and ("DUMMY" not in input_buff):
                    if (input_buff not in self.Inputs ) and (input_buff not in already_receive_buffs):
                        source = self.ComputingNodes[engine].receiver[input_buff][0]
                        request_id = commdict[(source, input_buff, i)]
                        cpp("        MPI_Wait(&requests[" +str(request_id)+"], &status["+str(request_id)+"]);")
                        already_receive_buffs.append(input_buff)
                for input_buff in related_input_buffers:
                    cpp("        ex"+str(i)+".input(\""+str(input_buff)+"\", "+str(input_buff)+");\n")
        #            if "DUMMY" not in input_buff:
        #                cpp("        ex"+str(i)+".input(\""+str(input_buff)+"\", "+str(input_buff)+");\n")

                #if "DUMMY" not in output_buff:
                if True:
                    cpp("       ex"+str(i)+".extract(\""+str(output_buff)+"\", "+str(output_buff)+");")

                if output_buff not in self.Outputs:
                    output_size = str(output_buff)+".total()"
                    dest_list = self.ComputingNodes[engine].sender[output_buff]
                    for dest in dest_list:
                        request_id = commdict[(i, output_buff, dest)]
                        #if "DUMMY" not in output_buff:
                        if True:
                            cpp("        MPI_Isend((float* )"+ str(output_buff)+ ", " +output_size+
                                ", MPI_FLOAT, "+str(dest)+", ")
                            cpp("                "+str(self.commtag[(i,output_buff,dest)])+", MPI_COMM_WORLD, &requests[" + str(request_id) +"]);\n")

            for output_buff in self.ComputingNodes[engine].outbuffs:
                if output_buff not in self.Outputs:
                    dest_list = self.ComputingNodes[engine].sender[output_buff]
                    for dest in dest_list:
                        #if "DUMMY" not in output_buff:
                        if True:
                            request_id = commdict[(i, output_buff, dest)]
                            cpp("            MPI_Wait(&requests[" +str(request_id)+"], &status["+str(request_id)+"]);")
                #printf("Iteration: %d, IRank: %d  time = %7.2f\n",i, irank, time);  
            cpp("    }\n")

            cpp("    double time_min = DBL_MAX;")
            cpp("    double time_max = -DBL_MAX;")
            cpp("    double time_avg = 0;\n")
            cpp("    for (int i = 0; i < g_loop_count; i++)")
            cpp("    {")
            cpp("        double start = ncnn::get_current_time();\n")

        for input_buff, source in self.ComputingNodes[engine].receiver.items():
            input_size = str(input_buff)+".total()"
            #if (input_buff not in self.Inputs) and ("DUMMY" not in input_buff):
            if (input_buff not in self.Inputs) :
                recv_id = self.ComputingNodes[engine].receiver[input_buff]
                request_id = commdict[(source[0], input_buff, i)]
                cpp("        MPI_Irecv((float* )"+ str(input_buff)+ ", " +input_size+
                    ", MPI_FLOAT, "+str(source[0])+", ")
                cpp("                    "+str(self.commtag[(source[0],input_buff,i)])+", MPI_COMM_WORLD, &requests[" + str(request_id) +"]);\n")
############# i send output_buff to j

        cpp("        ex"+str(i)+" = "+str(net_name)+".create_extractor();\n")

        already_receive_buffs = []
        for output_buff in self.ComputingNodes[engine].outbuffs:
            node_input_names = []
            node_input_names =  traceUpNodes(enginegraph, output_buff,
                                             node_input_names, enginenode_map, 0, engineinitializer_map)
            related_input_buffers = []
            for node in node_input_names:
                if node in list(enginenode_map.keys()):
                    for input_name in enginenode_map[node].input:
                        if input_name in self.ComputingNodes[engine].inbuffs:
                            related_input_buffers.append(input_name)
            #print ("related_input_buffers ",related_input_buffers)
            for input_buff in related_input_buffers:
                #if (input_buff not in self.Inputs ) and (input_buff not in already_receive_buffs) and ("DUMMY" not in input_buff):
                if (input_buff not in self.Inputs ) and (input_buff not in already_receive_buffs) :
                    source = self.ComputingNodes[engine].receiver[input_buff][0]
                    request_id = commdict[(source, input_buff, i)]
                    cpp("        MPI_Wait(&requests[" +str(request_id)+"], &status["+str(request_id)+"]);")
                    already_receive_buffs.append(input_buff)
            for input_buff in related_input_buffers:
            #    if "DUMMY" not in input_buff:
                cpp("        ex"+str(i)+".input(\""+str(input_buff)+"\", "+str(input_buff)+");\n")
            #if "DUMMY" not in output_buff:
            if True:
                cpp("       ex"+str(i)+".extract(\""+str(output_buff)+"\", "+str(output_buff)+");")

            if output_buff not in self.Outputs:
                output_size = str(output_buff)+".total()"
                dest_list = self.ComputingNodes[engine].sender[output_buff]
                for dest in dest_list:
                    request_id = commdict[(i, output_buff, dest)]
                    #if "DUMMY" not in output_buff:
                    if True:
                        cpp("        MPI_Isend((float* )"+ str(output_buff)+ ", " +output_size+
                            ", MPI_FLOAT, "+str(dest)+", ")
                        cpp("                "+str(self.commtag[(i,output_buff,dest)])+", MPI_COMM_WORLD, &requests[" + str(request_id) +"]);\n")

        for output_buff in self.ComputingNodes[engine].outbuffs:
            if output_buff not in self.Outputs:
                dest_list = self.ComputingNodes[engine].sender[output_buff]
                for dest in dest_list:
                    #if "DUMMY" not in output_buff:
                    if True:
                        request_id = commdict[(i, output_buff, dest)]
                        cpp("            MPI_Wait(&requests[" +str(request_id)+"], &status["+str(request_id)+"]);")
        if (self.Benchmark):
            cpp("        double end = ncnn::get_current_time();")
            cpp("        double time = end - start;")
            cpp("        time_min = std::min(time_min, time);")
            cpp("        time_max = std::max(time_max, time);")
            cpp("        time_avg += time;")
            #printf("Iteration: %d, IRank: %d  time = %7.2f\n",i, irank, time);  
            cpp("    }")
            cpp("    time_avg /= g_loop_count;")
            cpp("    fprintf(stderr, \"IRank: %d  min = %7.2f  max = %7.2f  avg = %7.2f\\n\", irank, time_min, time_max, time_avg);")
            cpp("    fprintf(pp, \"%.2f\", time_avg); // Unit: ms")
            cpp("    fprintf(pm, \"%.3f\", getPhysical()*1.0/1024); // Unit:MBytes")
            #cpp("    std::cout <<\"IRank: \"<< irank << \", Virtual Memory Usage (KB): \"<<getVirtual()<< std::endl;")
            cpp("    std::cout <<\"IRank: \"<< irank << \", Physical Memory Usage (KB): \"<<getPhysical()<< std::endl;")

            cpp("   fclose(pm);")
            cpp("   fclose(pp);")
        # cpp("MPI_Waitall("+str(request_len)+", requests, status);\n")
        cpp("}\n")

    def GenerateMain(self, cpp):
        cpp("int main(int argc, char** argv)")
        cpp("{")
        cpp("   int provided;")
        cpp("   MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);")

        cpp("   if (provided < MPI_THREAD_MULTIPLE) {")
        cpp("       fprintf(stderr, \"xxx MPI does not provide needed thread support!\\n\");")
        cpp("       return -1;")

        cpp("   // Error - MPI does not provide needed threading level")
        cpp("   }")

        cpp("    // MPI::Init(argc, argv);\n")
        cpp("    // Get the number of processes")
        cpp("    int world_size;")
        cpp("    MPI_Comm_size(MPI_COMM_WORLD, &world_size);\n")
        cpp("    // Get the rank of the process")
        cpp("    int world_rank;")
        cpp("    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);\n")

        cpp("    if (argc != 2)")
        cpp("    {")
        cpp("        fprintf(stderr, \"Usage: %s [imagepath]\\n\", argv[0]);")
        cpp("        return -1;")
        cpp("    }\n")
        cpp("   const char* imagepath = argv[1];")

        #if (self.Benchmark):
        #    cpp("if (world_rank < " + str(len(self.Platforms)) + "){")
        #    cpp("    powerstart(world_rank);")
        #    cpp("}\n")

        for i in range(len(self.NodesList)):
            if self.ComputingNodes[self.NodesList[i]].hardware == 'gpu':
                cpp("    if(world_rank=="+ str(i) + ") {")
                cpp("    ncnn::create_gpu_instance();")
                cpp("   }")

        cpp("   g_blob_pool_allocator.set_size_compare_ratio(0.0f);")
        cpp("   g_workspace_pool_allocator.set_size_compare_ratio(0.5f);")
        cpp("   std::vector<float> cls_scores;")
        cpp("   multi_classify(imagepath, cls_scores);")
        for i in range(len(self.NodesList)):
            if self.ComputingNodes[self.NodesList[i]].hardware == 'gpu':
                cpp("    if(world_rank=="+ str(i) + ") {")
                cpp("    ncnn::destroy_gpu_instance();")
                cpp("   }")
        cpp("   // Finalize the MPI environment.")

        #if (self.Benchmark):
        #    cpp("if (world_rank < " + str(len(self.Platforms)) + "){")
        #    cpp("    powerend(world_rank);")
        #    cpp("}\n")
        cpp("   MPI_Finalize();")
        cpp("   return 0;")
        cpp("}")

class Resource():
    def __init__(self, **map_details):
# name; inbuffs; receiver; outbuffs; sender ; hardware ;number;cores; dist_layers
        self.__dict__.update(map_details)
        self.platforms = ['edge01_arm_012345_gpu_0',
                          'edge02_arm_012345_gpu_0']
        self.resourceid = {}
        self.ResoureGenerator(self.platforms)

    def ResoureGenerator(self, platforms):
        elem_id = 0
        for platform in platforms:
            PlatformList = platform.split('_')
            platform_name = PlatformList[0]
            platform_arch = PlatformList[1]
            cpu_num = len(PlatformList[2])
            platform_gpu = PlatformList[3]
            gpu_num = len(PlatformList[4])

            for i in range(1, cpu_num+1):
                for elem in it.combinations(PlatformList[2], i):
                    key = platform_name +'_'+platform_arch+''.join(elem)
                    self.resourceid[elem_id] = key
                    elem_id = elem_id + 1

            for i in range(1, gpu_num+1):
                for elem in it.combinations(PlatformList[4], i):
                    key = platform_name +'_gpu'+''.join(elem)
                    self.resourceid[elem_id] = key
                    elem_id = elem_id + 1
        # for key, value in self.resourceid.items():
        #     print (key, '    ', value)

def ChromosomeGen(resourceid, gene=[3], modellength=24):
    resourcenum = len(resourceid.keys())
    chromosome = np.zeros((modellength, ))
    genelist = list(np.append(gene, [modellength]))
    ordergene = np.sort(np.append(gene, [0, modellength])).astype(int)
     
    for i in range(resourcenum):
        chromosome[ordergene[i]:ordergene[i+1]] = genelist.index(ordergene[i+1])
    return chromosome     

def ChromosomeSplitGen(gene, modellength):
    # gene represnet the splitting points.
    # gene: np.array([0,4,2,8])
    chromosome = np.array([]) 
    gene_map = np.argsort(gene)
    gene_expand = np.diff(np.sort(np.append(gene, modellength)))
    for i in range(len(gene_map)):
        chromosome  = np.append(chromosome, [gene_map[i]]*gene_expand[i])
    return chromosome

def ChromosomePartGen(gene, modellength, resourceid):
    # gene represnet the splitting points.
    # only use part of resources.
    # 0322 want to experiment with full cpu cores, gpu.
    # gene: np.array([0,4,2,8])
    chromosome = np.array([])
    gene_map = np.argsort(gene)

    negative_lines = []       
    for i in range(len(gene)):
        if gene[i]>modellength:   #### this is no-resource used.                          
            negative_lines.append(i)
    gene=np.delete(gene, np.array(negative_lines).astype(int), axis=0) 
    gene_map=gene_map[:len(gene)]

    gene_expand = np.diff(np.sort(np.append(gene, modellength)))
    resourcelist = list(resourceid.keys())
    for i in range(len(gene_map)):
        #chromosome  = np.append(chromosome, [gene_map[i]]*gene_expand[i])
        chromosome  = np.append(chromosome, [resourcelist[gene_map[i]]]*gene_expand[i])
    return chromosome


def GroupChromosomeGen(resourceid,gene,modellength, platforms):
    # [379 656   0]
    # group means same configurations.
    # inside group, order is important.
    idresource = dict((v,k) for k,v in resourceid.items())
    chromosome = np.zeros((modellength, ))
    gene = np.append(gene, [modellength])
    ordergene = np.sort(gene)
    #ordergene = np.sort(np.append(gene, [self.genetype_len]))
    genelist = list(gene)        
    resourcelist = list(resourceid.values())
    start = 0
    resourceid_clip = {}
    # generate chromosome to ensure that consective sub-models are deployed at the same device to reduce communication.
     
    for device in platforms:
        res = [key for key, val in idresource.items() if device in key]
        resnum = len(res)
        gene_clip = gene[sorted(np.argsort(gene)[start:start+len(res)+1])]
        # e.g. [799, 729, 882]
        gene_clip_list = list(np.delete(gene_clip,np.argmax(gene_clip)))
         
        for i in range(len(res)):
            cstart = gene[np.argsort(gene)[start]]
            cstop = gene[np.argsort(gene)[start+1]]
            index= gene_clip_list.index(cstart) % len(res)
            chromosome[cstart:cstop] = idresource[res[index]]
            start = start + 1
    return chromosome



def MappingGenerator(resourceid, chromosome, modelfile):
    model = onnx.load(modelfile)
    #model = onnx.shape_inference.infer_shapes(model)
    graph = model.graph
    for n in graph.node:
        if n.name == '':
            n.name = str(n.output[0])
    mapping = {}
    for i in range(len(chromosome)):
        gene = resourceid[chromosome[i]]   ### name of resource.
        #print (gene)
        mapping.setdefault(gene, []).append(graph.node[i].name)

    return mapping


if __name__ == '__main__':

    output_dirs= './models'
    if not os.path.exists(output_dirs):
    # Create a new directory because it does not exist
        os.makedirs(output_dirs)
        print("The output directory %s is created!" % (output_dirs))

    origin_model = "bvlcalexnet-9.onnx"
    #origin_model = "resnet101-v2-7.onnx"
    #origin_model = "densenet-9.onnx"
    input_model = format_onnx(origin_model)
    model =  onnx.load(input_model)
    model_len = len(model.graph.node)
    resourceid = { 1:'lenovo_cpu0', 2:'lenovo_cpu1'}
    platforms = ['lenovo']
    horizontal_file = './horizontal.json'
    horizontal_spec = load_json('./horizontal.json')

    random_map = load_json('./hmapping.json')
    #print (horizontal_spec["fc6_1"])
    #print (horizontal_spec["fc6_1"][0])
    #print (horizontal_spec["fc6_1"][1])
    #print (horizontal_spec["fc6_1"][2])
    #print (random_map)



    #M=3
    #gene = np.array([0,18]).astype(int)
    #chromosome = ChromosomePartGen(gene, model_len, resourceid)
    #random_map = MappingGenerator(resourceid, chromosome, input_model)
    #print ("Mapping: ", random_map)
    #save_json(random_map, './mapping.json')

    start_time = time.time()
    InputSpecs = Interface(model=input_model, mappings=random_map, platforms=platforms)
    InputSpecs.HorizontalInplace(horizontal_file)
    InputSpecs.ConsistencyCheck()
    InputSpecs.GenerateRankFile()
    InputSpecs.ModelSplit()
    InputSpecs.GenerateComm()
#
#    print ("Front End time: %f (s)"%(time.time() - start_time))
#    #cppname, NodesList, ComputingNodes
    GenerateCode = EngineCode(
        CppName = "./models/multinode",
        Platforms = InputSpecs.platforms,
        NodesList = InputSpecs.nodes,
        ComputingNodes = InputSpecs.computingnodes,
        ValueInfos = InputSpecs.value_map,
        Inputs = InputSpecs.inputs,
        Outputs = InputSpecs.outputs,
        Benchmark = False)



