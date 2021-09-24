# -*- coding: UTF-8 -*-
import onnx
from onnx import helper, checker
import re
import argparse
import json,copy
import numpy as np
from data_json import *
from onnx_split import *
from CodeGen import *
from onnx_bench import *


def onnx_ncnn(origin_model, mapping_file, platform_file):
###### 
#### 生成ncnn的多机模型代码 #
####
    with open(platform_file, 'r') as f:
            platform = f.readlines()
            #print (platform)
            for i in range(len(platform)):
                temp = platform[i].split(':')[0]
                platform[i] = temp.replace('\n','')
    #print (platform)
    platform_num = len(platform)
    
    platform_dict={}
    ######platform 对应id
    for i in range(platform_num):
        platform_dict[platform[i]] = i
    
    print ("platform dict: ", platform_dict)
    
    platform_mapping = load_json(mapping_file)
    engine_num = len(list(platform_mapping))
    input_tensor_dict = {}
    mapping_node_name_list =[]
    for key, value in platform_mapping.items():    
        onnx_extract(origin_model, './models/'+key+'.onnx', value)
        input_tensor_dict[key] =  getInputlayers('./models/'+key+'.onnx')
        for node_name in value:
            mapping_node_name_list.append(node_name)
    input_tensors_jsonFile = open('./models/input_tensors_list.json', "w")
    input_tensors_content = json.dumps(input_tensor_dict)
    input_tensors_jsonFile.write(input_tensors_content)
    input_tensors_jsonFile.close()
    input_tensor_dict = load_json('./models/input_tensors_list.json')
    output_tensor_dict = {}
    
    platform_list = list(platform_dict)
    
    ###########################3
    ####检查mapping是否完整
    ### check consistency of model mapping
    graph = load_onnx(origin_model)
    node_map = generate_node_dict(graph.node)
    mapping_node_name_list = set(mapping_node_name_list)
    node_map_list = set (list (node_map))
    if (node_map_list ^ mapping_node_name_list):
        print ("Consistency Check Fail.")
        if (mapping_node_name_list - node_map_list):
            print ("Original model doesn't contain: ", mapping_node_name_list - node_map_list)
        if (node_map_list - mapping_node_name_list):
            print ("Given mapping file require nodes: ", node_map_list - mapping_node_name_list)
    
    #对于每个engine的node
    for i in range(engine_num):
        platform_name = platform_list[i]
        output_list = []
        computing_nodes = list(platform_mapping[platform_name])
        for node in computing_nodes:
            for key, value in input_tensor_dict.items():
                if key==platform_name:
                    continue
                if node in value:
                    if node not in output_list:
                        output_list.append(node)
    
        output_tensor_dict[platform_name] = output_list
    
    print ("per_engine output_dict_list: ",output_tensor_dict)
    output_tensors_jsonFile = open('./models/output_tensors_list.json', "w")
    output_tensors_content = json.dumps(output_tensor_dict)
    output_tensors_jsonFile.write(output_tensors_content)
    output_tensors_jsonFile.close()
    
    send_jsonFile = open('./models/sender.json', "w")
    recv_jsonFile = open('./models/receiver.json', "w")
    
    
    receiver_dict_list = {}
    sender_dict_list = {}
    
    tag_dict_list = {}
    tag = 0
    send_len = 0

    comm_dict_list = {}
    comm_tag = 0
    for key, value in platform_mapping.items():
        receiver_dict = {}
        sender_dict ={}
        tag_dict= {}
        input_tensor = input_tensor_dict[key]
        for k, v in input_tensor_dict.items():
            if k == key:
                continue
            for per_input_tensor in v:
                if per_input_tensor in value:
                    comm_dict_list[(key,k,per_input_tensor)] = comm_tag
                    comm_tag +=1
                    if per_input_tensor in sender_dict:
                        sender_dict[per_input_tensor].append(k)
                        tag_dict[k] = tag
                        # source: key dest: k; comm: per_input_tensor
                        tag+=1 
                        send_len +=1
                    else:
                        sender_dict[per_input_tensor] = [k]
                        tag_dict[k] = tag
                        tag+=1
                        send_len +=1
    
        sender_dict_list[key] = sender_dict
        tag_dict_list[key] = tag_dict
    
        for per_tensor in input_tensor:
    ###--------------------------------
            find = None
            for search_key, search_value in platform_mapping.items():
                if per_tensor in search_value:
                    find = search_key
                    break
            if find:
                if per_tensor in sender_dict:
                    receiver_dict[per_tensor].append(find)
                else:
                    receiver_dict[per_tensor] = [find]
        receiver_dict_list[key] = receiver_dict
        #print (key, "receiver:  ",receiver_dict)
    
    print ("communication: ",comm_dict_list)
    receiver_dict_content = json.dumps(receiver_dict_list)
    recv_jsonFile.write(receiver_dict_content)
    
    sender_dict_content = json.dumps(sender_dict_list)
    send_jsonFile.write(sender_dict_content)
    
    send_jsonFile.close()
    recv_jsonFile.close()
    
    origin_input_tensor = getInputlayers(origin_model)
    origin_output_tensor = []
    for i in load_onnx(origin_model).output:
        origin_output_tensor.append(i.name)
    print ("orgin input tensors: *** :",origin_input_tensor)
    print ("origin output tensors: ^^^ :", origin_output_tensor)
    
    for i in range(engine_num):
        platform_name = platform_list[i]
        computing_nodes = list(platform_mapping[platform_name])
        sender_list = list(sender_dict_list[platform_name])
    
        model = onnx.load('./models/'+platform_name+'.onnx')
        model = onnx.shape_inference.infer_shapes(model)
        graph = model.graph
    
        #print("Check input model Errors: ", onnx.checker.check_model(model))
        #Generate a name for all node if they have none.
        nodeIdx = 0;
        for n in graph.node:
            if n.name == '':
                n.name = str(n.output[0])
    
        input_map = generate_node_dict(graph.input)
        output_map = generate_node_dict(graph.output)
        initializer_map = generate_node_dict(graph.initializer)
        value_map = generate_node_dict(graph.value_info)
        node_map = generate_node_dict(graph.node)
    
    
        for j in graph.output:
            sender_list.append(j.name)
        order_sender_list = [n for n in list(node_map.keys()) if n in sender_list]
        #print (platform_mapping[platform[i]])
    
        for j in order_sender_list:
            engine_name = str(platform[i]) + str(j) +'.onnx'
            node_input_names = []
            node_input_names =  traceUpNodes(graph, j,node_input_names, node_map, 0, initializer_map)
            onnx_extract('./models/'+str(platform[i])+'.onnx', './models/'+engine_name, node_input_names)
    
    
    # output_names_list = []
    # for i in graph.node:
    #     output_names_list.append(i.output[0])
    # print (output_names_list)
    ##############
    #-------------
    ##############
    cpp = CppFile("./models/multinode.cpp")
    
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

    cpp("static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;")
    cpp("static ncnn::PoolAllocator g_workspace_pool_allocator;\n")

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
    
    
    
    cpp("static int multi_classify(const cv::Mat& bgr, std::vector<float>& cls_scores)")
    cpp("{")
    
    #getInputlayers(input_model)
    recv_node_list = []
    for k, v in receiver_dict_list.items():
        for key, value in v.items():
            recv_node_list.append(key)
    print (recv_node_list)
    request_len = len(recv_node_list) * 2
    ### recv + send = len
    cpp("int irank = MPI::COMM_WORLD.Get_rank();")
    cpp("int num_threads = ncnn::get_cpu_count();")
    
    cpp("MPI_Request requests["+str(send_len*2)+"];")
    cpp("MPI_Status status["+str(send_len*2)+"];\n")
    
    recv_request_index = 0
    send_request_index = len(recv_node_list)
    
    recv_request_dict={}
    send_request_dict={}
    
    #recore request index of mpi communication process
    print ("communication tag dict: ", tag_dict_list) 
    
    
    for i in range(engine_num):
        platform_name = platform_list[i]
        computing_nodes = list(platform_mapping[platform_name])
        sender_list = list(sender_dict_list[platform_name])
        receiver_list = list(receiver_dict_list[platform_name])
        tag_dict = tag_dict_list[platform_name]
        model = onnx.load('./models/'+platform_name+'.onnx')
        model = onnx.shape_inference.infer_shapes(model)
        graph = model.graph
    
        #print("Check input model Errors: ", onnx.checker.check_model(model))
        #Generate a name for all node if they have none.
        nodeIdx = 0;
        for n in graph.node:
            if n.name == '':
                n.name = str(n.output[0])
    
        input_map = generate_node_dict(graph.input)
        output_map = generate_node_dict(graph.output)
        initializer_map = generate_node_dict(graph.initializer)
        value_map = generate_node_dict(graph.value_info)
        node_map = generate_node_dict(graph.node)
    
        for j in graph.output:
            sender_list.append(j.name)
        order_sender_list = [n for n in list(node_map.keys()) if n in sender_list]
    
        net_name = "resnet" + str(i)
        cpp("if(irank=="+str(i)+"){")
        
        for j in receiver_list:
              #      int tag, MPI_Comm comm, MPI_Request * request)
            jj_shape = get_node_output_shape(value_map[j])
            new_shape = jj_shape[0:1] + jj_shape[2:4] + jj_shape[1:2]
    #         new_shape = jj_shape
            j_shape = str(new_shape[1:]).replace('[','(').replace(']',')')
            j_size = str(j)+".total()"
            cpp("    ncnn::Mat "+ str(j)+ j_shape+";")
            
        for j in order_sender_list:
            if j in origin_output_tensor:
                cpp("    ncnn::Mat "+str(j)+";")
            else:
                jj_shape = get_node_output_shape(value_map[j])
                new_shape = jj_shape[0:1] + jj_shape[2:4] + jj_shape[1:2]
        #         new_shape = jj_shape
                j_shape = str(new_shape[1:]).replace('[','(').replace(']',')')
                cpp("    ncnn::Mat "+str(j)+ j_shape+";")
            engine_name = platform[i] + j +'.onnx'
            net_name = platform[i] + j
            cpp("    ncnn::Net "+ net_name + ";")
            cpp("    "+net_name+".opt.use_vulkan_compute = false;")
            #cpp("    "+net_name+".opt.lightmode = false;")

            cpp("    "+net_name+".opt.blob_allocator = &g_blob_pool_allocator;")
            cpp("    "+net_name+".opt.workspace_allocator = &g_workspace_pool_allocator;")

            cpp("    "+net_name+".opt.num_threads = num_threads;")
            cpp("    "+net_name+".load_param(\""+net_name+".param\");")
            cpp("    "+net_name+".load_model(\""+net_name+".bin\");\n")
    
                
        for j in receiver_list:
            # MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source,
          
            #sender_dict_list[platform_name][j]:
            #receiver_dict_list[platform_name][j][0]
    #         print ("tag: ",tag_index)
            recv_source = receiver_dict_list[platform_name][j][0]
            tag_index = tag_dict_list[recv_source][platform_name]
            j_size = str(j)+".total()"

            comm_id = comm_dict_list[(recv_source,platform_name,j)]
    
            cpp("    MPI_Irecv((float* )"+ str(j)+ ", " +j_size+
                ", MPI_FLOAT, "+str(platform_dict[recv_source])+", ")
            cpp("        "+str(comm_id)+", MPI_COMM_WORLD, &requests[" + str(comm_id+ send_len) +"]);\n")
            recv_request_dict[j] = recv_request_index
            recv_request_index+=1
    
        gen_in = True
        for j in order_sender_list:
            engine_name = platform[i] + j +'.onnx'
            net_name = platform[i] + j
            input_list = getInputlayers('./models/'+platform[i]+j+'.onnx')
            cpp("    ncnn::Extractor ex"+str(j)+" = "+str(net_name)+".create_extractor();\n")
            for per_input in input_list:
                #cpp("    ex"+str(j)+" = "+str(net_name)+".create_extractor();\n")
                if gen_in and (per_input in origin_input_tensor):
                    cpp("    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 224, 224);")
                    cpp("    const float mean_vals[3] = {104.f, 117.f, 123.f};")
                    cpp("    in.substract_mean_normalize(mean_vals, 0);")
                    gen_in = False
            
            for per_input in input_list:
                #cpp("    ex"+str(j)+" = "+str(net_name)+".create_extractor();\n")
                if (per_input in origin_input_tensor):
                    cpp("    ex"+str(j)+".input(\""+str(per_input)+"\", in);\n")
                else:
                    recv_source = receiver_dict_list[platform_name][per_input][0]
                    tag_index = tag_dict_list[recv_source][platform_name] + send_len
                    comm_id = comm_dict_list[(recv_source, platform_name, per_input)]
                    cpp("    MPI_Wait(&requests[" +str(comm_id+send_len)+"], &status["+str(comm_id+send_len)+"]);")
                    cpp("    ex"+str(j)+".input(\""+str(per_input)+"\", "+str(per_input)+");")
#                    cpp("    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 224, 224);")
#                    cpp("    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};")
#                    cpp("    const float norm_vals[3] = {1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5};")
#                    cpp("    in.substract_mean_normalize(mean_vals, norm_vals);\n")

            cpp("    ex"+str(j)+".extract(\""+str(j)+"\", "+str(j)+");")
    #             int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
    #               MPI_Comm comm, MPI_Request *request)
            ### comm tag problem
            j_size = str(j)+".total()"
    
            if j in origin_output_tensor:
                cpp("    cls_scores.resize("+str(j)+".w);\n")
                cpp("    for (int j = 0; j < "+str(j)+".w; j++)")
                cpp("    {")
                cpp("        cls_scores[j] = "+str(j)+"[j];")
                cpp("    }\n")
                #cpp("    print_topk(cls_scores, 3);\n")
                cpp("    std::vector<std::string> labels;")
                cpp("    load_labels(\"synset_words.txt\", labels);")
                cpp("    std::vector<int> index;")
                cpp("    std::vector<float> score;")
                cpp("    print_topk(cls_scores, 3, index, score);")
                cpp("    for (int i = 0; i < index.size(); i++)")
                cpp("    {")
                cpp("        fprintf(stderr, \"%s \\n\", labels[index[i]].c_str());")
                cpp("    }")
            else:
                for dest in sender_dict_list[platform_name][j]:
                    tag_index = tag_dict[dest]
                    comm_id = comm_dict_list[(platform_name, dest,j)]
    #                 print ("tag: ",tag_index)
                    cpp("    MPI_Isend((float* )"+ str(j)+ ", " +j_size+
                    ", MPI_FLOAT, "+str(platform_dict[dest])+", ")
                    cpp("        "+str(comm_id)+", MPI_COMM_WORLD, &requests[" + str(comm_id) +"]);\n")
                    cpp("    MPI_Wait(&requests[" +str(comm_id)+"], &status["+str(comm_id)+"]);")
                    tag_index += 1
                    send_request_dict[j] = send_request_index
                    send_request_index+=1
   #     if j not in origin_output_tensor:
   #        for dest in sender_dict_list[platform_name][j]:
   #            tag_index = tag_dict[dest]
   # #            print ("tag: ",tag_index)
   #            cpp("    MPI_Wait(&requests[" +str(tag_index)+"], &status["+str(tag_index)+"]);")


    
        cpp(" }\n")
    cpp("return 0;\n")
    # cpp("MPI_Waitall("+str(request_len)+", requests, status);\n")
    cpp("}\n")
    cpp("int main(int argc, char** argv)")
    cpp("{")
    
    cpp("    MPI::Init(argc, argv);\n")
    
    cpp("    // Get the number of processes")
    cpp("    int world_size;")
    cpp("    world_size = MPI::COMM_WORLD.Get_size();\n")
    
    cpp("    // Get the rank of the process")
    cpp("    int world_rank;")
    cpp("    world_rank = MPI::COMM_WORLD.Get_rank();\n")
    
    
    cpp("    if (argc != 2)")
    cpp("    {")
    cpp("        fprintf(stderr, \"Usage: %s [imagepath]\\n\", argv[0]);")
    cpp("        return -1;")
    cpp("    }\n")

    cpp("    const char* imagepath = argv[1];\n")
    cpp("    g_blob_pool_allocator.set_size_compare_ratio(0.0f);")
    cpp("    g_workspace_pool_allocator.set_size_compare_ratio(0.5f);\n")
    
    cpp("    cv::Mat m = cv::imread(imagepath, 1);")
    cpp("    if (m.empty())")
    cpp("    {")
    cpp("        fprintf(stderr, \"cv::imread %s failed\\n\", imagepath);")
    cpp("        return -1;")
    cpp("    }\n")
    
    cpp("    std::vector<float> cls_scores;")
    cpp("    multi_classify(m, cls_scores);\n")

    cpp("    // Finalize the MPI environment.")
    cpp("    MPI::Finalize();\n")
    cpp("    return 0;")
    cpp("}\n            ")
    cpp.close()




 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("origin_model", help="original onnx model input. ")
    parser.add_argument("mapping", help="Details of Distribution of layers on each node")
    parser.add_argument("platform", help="Platforms......")
    args = parser.parse_args()
    onnx_ncnn(args.origin_model, args.mapping, args.platform)
    onnx_bench(args.origin_model, args.mapping, args.platform)

