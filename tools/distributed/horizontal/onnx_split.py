# -*- coding: utf-8 -*-
import onnx
from onnx import helper, checker
import re
import argparse
import json,copy
import numpy as np
from data_json import *
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto


def find_node_index(graph, node_name):
    idx = 0
    for n in graph.node:
        if n.name == node_name:
            return idx
        idx = idx +1
    return -1

def gen_essential(platform, nodes_number):
    with open("platform.txt", 'w') as p_file:
        for i in range((int)(nodes_number)):
            p_file.write(platform[i]+'\n')
    #rank 0=nx01 slot=0-5
    with open("hostfile", 'w') as h_file:
        for i in range((int)(nodes_number)):
            h_file.write(platform[i]+'    slots=1 \n')

    with open("rankfile", 'w') as h_file:
        for i in range((int)(nodes_number)):
            h_file.write('rank '+ str(i) + '=' + platform[i]+'    slots=0-5 \n')

##############################
def format_onnx(input_model):
    model = onnx.load(input_model)
    model = onnx.shape_inference.infer_shapes(model)
    graph = model.graph
    for init_i in range(len(graph.initializer)):
        graph.initializer[init_i].name = graph.initializer[init_i].name.replace('/','_')
    for i in range(len(graph.input)):
        graph.input[i].name = graph.input[i].name.replace('/','_')
    for i in range(len(graph.output)):
        graph.output[i].name = graph.output[i].name.replace('/','_')


    # print (" MODEL has total %d layers." % len(graph.node))
    for i in range(len(graph.node)):
        graph.node[i].name = str(graph.node[i].output[0]).replace('/','_')
        for input_i in range(len(graph.node[i].input)):
            graph.node[i].input[input_i] = (graph.node[i].input[input_i]).replace('/','_')
        for output_i in range(len(graph.node[i].output)):
            graph.node[i].output[output_i] = graph.node[i].output[output_i].replace('/','_')
    
    try:
        model_name = 'format_'+input_model.split('/')[-1]
        onnx.save(model, model_name)
        print("Check input model:::", model_name, " Format Errors: ", onnx.checker.check_model(model))
    except:
        print ("Check Model invalid or Path not Exists.")
    return model_name
############## 标准读取 onnx model ################
##############################


def load_onnx(input_model):
    model = onnx.load(input_model)
    checker.check_model(model)
    model = onnx.shape_inference.infer_shapes(model)

    graph = model.graph

    #print("Check input model Errors: ", onnx.checker.check_model(model))

    #Generate a name for all node if they have none.
    for n in graph.node:
        if n.name == '':
            n.name = str(n.output[0])
    return graph

################非常标准的求模型的input名字

def getInputlayers(input_model):

    graph = load_onnx(input_model)
    input_map = generate_node_dict(graph.input)
    output_map = generate_node_dict(graph.output)
    initializer_map = generate_node_dict(graph.initializer)
    value_map = generate_node_dict(graph.value_info)
    node_map = generate_node_dict(graph.node)

    net_input = list(set(list(input_map))  - set(list(initializer_map)))
    return net_input

def getOutputlayers(input_model):
 
    graph = load_onnx(input_model)
    input_map = generate_node_dict(graph.input)
    output_map = generate_node_dict(graph.output)
    initializer_map = generate_node_dict(graph.initializer)                                                                           
    value_map = generate_node_dict(graph.value_info)
    node_map = generate_node_dict(graph.node)
    output_names = []
    for node in list(node_map):
        if node_map[node].op_type == "Dropout":
            continue
        for output_name in node_map[node].output:
            output_names.append(output_name)

    
    net_output = set(output_names)  - set(list(node_map))
    net_output = list(net_output - set(list(initializer_map)))
    return net_output


def create_initializer_tensor(
        name: str,
        tensor_array: np.ndarray,
        data_type: onnx.TensorProto = onnx.TensorProto.FLOAT
) -> onnx.TensorProto:

    # (TensorProto)
    initializer_tensor = onnx.helper.make_tensor(
        name=name,
        data_type=data_type,
        dims=tensor_array.shape,
        vals=tensor_array.flatten().tolist())

    return initializer_tensor

def split_name_list(name_list,new_names_all):
    #splits input/output list to identify removed, retained and totally new nodes    
    removed_names=[]
    retained_names=[]
    for n in name_list:
        if n.name not in new_names_all:                
            removed_names.append(n.name)              
        if n.name in new_names_all:                
            retained_names.append(n.name)                      
    return [removed_names,retained_names]

def get_node_output_size(node_value_info):
    # e.g. from the graph.node[0].value_info to  get the shape 
    # return e.g. [1, 384, 26, 26]
    # output_type: List[]
    size = 1 
    for i in node_value_info.type.tensor_type.shape.dim:
        size *= int(i.dim_value)
    return size 

def get_node_output_shape(node_value_info):
    # e.g. from the graph.node[0].value_info to  get the shape 
    # return e.g. [1, 384, 26, 26]
    # output_type: List[]
    shape = []
    for i in node_value_info.type.tensor_type.shape.dim:
        # fix for ResNet, which has batch size = N, it gets converted to 0
        # if this causes problems, we can change "N" -> 1
        if i.dim_value == 0:
            shape.append("N")
        else:
            shape.append(i.dim_value)
    #return n,c,h,w
    return shape 
    #new_shape = shape[0:1]+shape[2:4]+shape[1:2]
    #return new_shape

def get_node_output_shape_safe(graph, splitnode_name, value_map):
    # prevents errors when we want the output of the last layer
    # last layer may not be in value_map
    if splitnode_name in value_map.keys():
        split_node_shape = get_node_output_shape(value_map[splitnode_name])
    else:
        if graph.node[-1].name == splitnode_name:
            output_shape = graph.output[0].type.tensor_type.shape.dim
            split_node_shape = [d.dim_value for d in output_shape]
    return split_node_shape

def get_node_input_shape(graph, layer, value_map):
    # get the input shape of a layer
    # if we are on the first layer, we cannot use the value_map way
    input_node_shape = None
    try:
        input_node_shape = get_node_output_shape(value_map[layer.input[0]])
    except Exception as _:
        pass
    if input_node_shape is None:
        if layer.input[0] == graph.input[0].name:
            input_node_shape = [dim.dim_value for dim in graph.input[0].type.tensor_type.shape.dim]
        else:
            for graph_in in graph.input:
                if graph_in.name == layer.input[0]:
                    input_node_shape = [dim.dim_value for dim in graph_in.type.tensor_type.shape.dim]
                    break

    return input_node_shape


def generate_node_dict(graph_member_list):
    # To generate a dictionary for the graph node, value_info, initializer, ....
    # keys(): name
    # value: depend on input.
    member_map=dict();
    for n in graph_member_list:
        member_map[n.name]=n;
    return member_map


def traceUpNodes(graph, name, node_input_names, node_map, start_index, initializer_map):
    # recurisvely traces all dependent nodes for a given output nodes in a graph    
    #  valid_node_names = traceUpNodes(graph, output_node_name,
    #                valid_node_names,node_map, input_node_names, initializer_map)
    # trace all node from the partition start index to the next partition start_index.

    for n in graph.node[start_index:]:
        for noutput in n.output:       
            if (noutput == name) and (n.name not in node_input_names):
                # give node "name" is node n's output, so add node "n" to node_input_names list 
                node_input_names.append(n.name)
                if n.name in node_map.keys():
                    for ninput in node_map[n.name].input:
                        # trace input node's inputs 
                        node_input_names = traceUpNodes(graph,ninput,node_input_names,node_map, start_index, initializer_map)                                        
    # don't forget the initializers they can be terminal inputs on a path.                    
    if name in initializer_map.keys():
        node_input_names.append(name)                    
    return node_input_names    

#         """ edits and modifies an onnx model to extract a subgraph based on input/output node names and shapes.
#     Arguments:
#         input_model: path of input onnx model
#         output_model: path base of output onnx model
#         input_names_list: list of input_name.

#     """
def extract_subgraph(graph, subgraph, sub_nodes):
    input_tensors = getInputlayers(graph)
    input_map = generate_node_dict(graph.input)
    output_map = generate_node_dict(graph.output)
    initializer_map = generate_node_dict(graph.initializer)
    value_map = generate_node_dict(graph.value_info)
    node_map = generate_node_dict(graph.node)

    sub_node_names = [n for n in list(node_map) if n in sub_node_names_list]
    #  找出节点名称

    input_names = []
    output_names =[]
    # GET ALL INPUT NAMES (including weights)
    for node in sub_node_names:
        for input_name in node_map[node].input:
            input_names.append(input_name)

    # GET ALL OUTPUT NAMES (including weights)
    for node in sub_node_names:
        for output_name in node_map[node].output:
            output_names.append(output_name)

    #print (output_names)
    # needs to be topology sorted.
    graph_node_names = [n for n in list(node_map) if n in sub_node_names]
    input_node_names = [n for n in list(input_map) if n in input_names]
    output_node_names = [n for n in list(output_map) if n in output_names]
    print (input_node_names)

    [removed_names,retained_names]=split_name_list(graph.input, input_node_names)


    for name in removed_names:
        if name in input_map.keys():
            graph.input.remove(input_map[name])

    [removed_names,retained_names]=split_name_list(graph.output, output_names)
    for name in removed_names:
        if name in output_map.keys():
            graph.output.remove(output_map[name])

    valid_node_names = set(graph_node_names) | set(input_node_names)


    #print (valid_node_names)
    invalid_node_names = list( (set(node_map.keys()) | set(initializer_map.keys())) - set(valid_node_names))
    print ("invalid-----")
    print (invalid_node_names)
    #print ("invalid_node names: ", invalid_node_names)

    #removed_node_names = [n for n in node_map.keys() if n in invalid_node_names]
    #print ("removed_node: ", removed_node_names)

    # Remove all the invalid nodes from the graph
    for name in invalid_node_names:
        if name in initializer_map.keys():
            graph.initializer.remove(initializer_map[name])
        #if name in input_map.keys():
        #    graph.input.remove(input_map[name])
        if name in node_map.keys():
            graph.node.remove(node_map[name])

    #print ("invalid node names: ", invalid_node_names)
    new_node_map = generate_node_dict(graph.node)
    new_input_map = generate_node_dict(graph.input)
    new_output_map = generate_node_dict(graph.output)
    new_initializer_map = generate_node_dict(graph.initializer)
    new_value_map = generate_node_dict(graph.value_info)
    new_input_all= []



    for n in graph.node:
        for input_name in n.input:
            new_input_all.append(input_name)
    new_input_all = set(new_input_all)
    extra_tensor = list(set(new_input_all) - (set(new_node_map.keys()) | set(new_initializer_map.keys())))
    #print ("need input tensor: ", extra_tensor)
    extra_tensor = list(set(extra_tensor) - set(input_tensors))
    for per_tensor in extra_tensor:
        try:
            new_tensor_input = onnx.helper.make_tensor_value_info(per_tensor,
                               onnx.TensorProto.FLOAT,get_node_output_shape(value_map[per_tensor]))
        except:
            new_tensor_input = onnx.helper.make_tensor_value_info(per_tensor,
                               onnx.TensorProto.FLOAT,[1])
        graph.input.extend([new_tensor_input])

    subgraph = graph


def onnx_extract(input_model, output_model, sub_node_names_list, opset_version):
    model = onnx.load(input_model)
    #model = onnx.shape_inference.infer_shapes(model)
    graph = model.graph

   ## print("Check input model Errors: ", onnx.checker.check_model(model))

    ##Generate a name for all node if they have none.
    #nodeIdx = 0;
    #for n in graph.node:
    #    if n.name == '':
    #        n.name = str(n.output[0])
    input_tensors = getInputlayers(input_model)
    input_map = generate_node_dict(graph.input)
    output_map = generate_node_dict(graph.output)
    initializer_map = generate_node_dict(graph.initializer)
    value_map = generate_node_dict(graph.value_info)
    node_map = generate_node_dict(graph.node)


    sub_node_names = [n for n in list(node_map) if n in sub_node_names_list]
    #  找出节点名称

    input_names = []
    output_names =[]
    # GET ALL INPUT NAMES (including weights)
    for node in sub_node_names:
        for input_name in node_map[node].input:
            input_names.append(input_name)

    # GET ALL OUTPUT NAMES (including weights)
    for node in sub_node_names:
        for output_name in node_map[node].output:
            output_names.append(output_name)

    # needs to be topology sorted.
    graph_node_names = [n for n in list(node_map) if n in sub_node_names]
    input_node_names = [n for n in list(input_map) if n in input_names]
    output_node_names = [n for n in list(output_map) if n in output_names]

    [removed_names,retained_names]=split_name_list(graph.input, input_node_names)


    for name in removed_names:
        if name in input_map.keys():
            graph.input.remove(input_map[name])

    [removed_names,retained_names]=split_name_list(graph.output, output_node_names)
    for name in removed_names:
        if name in output_map.keys():
            graph.output.remove(output_map[name])

    valid_node_names = set(graph_node_names) | set(input_node_names)


    #print (valid_node_names)
    invalid_node_names = list( (set(node_map.keys()) | set(initializer_map.keys())) - set(valid_node_names))
    #print ("invalid_node names: ", invalid_node_names)

    #removed_node_names = [n for n in node_map.keys() if n in invalid_node_names]
    #print ("removed_node: ", removed_node_names)

    # Remove all the invalid nodes from the graph
    for name in invalid_node_names:
        if name in initializer_map.keys():
            graph.initializer.remove(initializer_map[name])
        #if name in input_map.keys():
        #    graph.input.remove(input_map[name])
        if name in node_map.keys():
            graph.node.remove(node_map[name])

    #print ("invalid node names: ", invalid_node_names)
    new_node_map = generate_node_dict(graph.node)
    new_input_map = generate_node_dict(graph.input)
    new_output_map = generate_node_dict(graph.output)
    new_initializer_map = generate_node_dict(graph.initializer)
    new_value_map = generate_node_dict(graph.value_info)
    new_input_all = []
    new_output_all = []


    for n in graph.node:
        for input_name in n.input:
            new_input_all.append(input_name)
    new_input_all = set(new_input_all)
    extra_tensor = list(set(new_input_all) - (set(new_node_map.keys()) | set(new_initializer_map.keys())))
    #print ("need input tensor: ", extra_tensor)
    extra_tensor = list(set(extra_tensor) - set(input_tensors))
    for per_tensor in extra_tensor:
        try:
            new_tensor_input = onnx.helper.make_tensor_value_info(per_tensor,
                               onnx.TensorProto.FLOAT,get_node_output_shape(value_map[per_tensor]))
        except:
            new_tensor_input = onnx.helper.make_tensor_value_info(per_tensor,
                               onnx.TensorProto.FLOAT,[1])
        graph.input.extend([new_tensor_input])

    for n in graph.node:
        for output_name in n.output:
            new_output_all.append(output_name)
    new_output_all = set(new_output_all)

    #print("New Output Model", str(output_model), " Generated. Errors: ", onnx.checker.check_model(model))
    model = helper.make_model(graph, producer_name='AutoDiCE', opset_imports=[helper.make_opsetid("", opset_version)]) 
    onnx.save(model, output_model)

def onnx_split(input_model, output_model, split):
    """ edits and modifies an onnx model to extract a subgraph based on input/output node names and shapes.
    Arguments: 
        input_model: path of input onnx model
        output_model: path base of output onnx model    
        split: "split.txt", definition of partition.
        ###根据split.txt的分割点分割模型并产生mapping文件.
        
    """
        
    # LOAD MODEL AND PREP MAPS
    model = onnx.load(input_model)
    #model = onnx.shape_inference.infer_shapes(model)

    graph = model.graph
    
    #print(" Check input model Errors: ", onnx.checker.check_model(model))
    
    #Generate a name for all node if they have none.
    nodeIdx = 0;
    for n in graph.node:
        if n.name == '':
            n.name = str(n.output[0])
    
    node_map = generate_node_dict(graph.node)
    input_map = generate_node_dict(graph.input)
    output_map = generate_node_dict(graph.output)
    initializer_map = generate_node_dict(graph.initializer)
    value_map = generate_node_dict(graph.value_info)
    origin_node_names = list(node_map)
    origin_input_node_names = list(input_map)
    origin_output_node_names = list(output_map)
    
    origin_input_initializer =  list(initializer_map)
    net_input = list(set(origin_input_node_names)  - set(origin_input_initializer))
    
    #print (net_input, origin_output_node_names)
    split_definition = []
    position_definition=[]

    sender_dict = {}
    
    with open(split, 'r') as f:
        temp = f.readlines()
        for i in temp:
            split_definition.append(i.rstrip('\n'))
        position_definition = split_definition[1].split(',')
        for i in range(len(position_definition)):
            position_definition[i] = position_definition[i].replace(' ','')
    position_definition.append(origin_output_node_names[0])


    if split_definition[0] == 'vertical':
        print ("Vertical Partition!")
    
    #print (position_definition)
    
    parts = 0
    split_parts = []
    start_index = 0
    part_index = 0


    mapping_file = 'mapping.json'
    mapping_dict = {}
    mapping_dict_list = {}
    jsonFile = open(mapping_file, "w")

    
    for n in graph.node:
        if n.name in position_definition:       
            output_path = output_model+"/export" + str(parts) +".onnx"
            model = onnx.load(input_model)
            model = onnx.shape_inference.infer_shapes(model)

            graph = model.graph
            for n in graph.node:
                if n.name == '':
                    n.name = str(n.output[0])
            node_map = generate_node_dict(graph.node)
            input_map = generate_node_dict(graph.input)
            output_map = generate_node_dict(graph.output)
            initializer_map = generate_node_dict(graph.initializer)
            value_map = generate_node_dict(graph.value_info)
            
            input_names = []
            output_names = []
            graph_node_names = []
  
            for i in range(start_index, part_index+1):
                graph_node_names.append(graph.node[i].name)
                for input_name in graph.node[i].input:
                    input_names.append(input_name)
            for output_name in graph.node[part_index].output:
                output_names.append(output_name)
    
            input_node_names = set(input_names)
            output_node_names = set(output_names)
            graph_node_names = set(graph_node_names)
            # needs to be topology sorted.
            input_node_names = [n for n in origin_input_node_names if n in input_node_names]
            graph_node_names = [n for n in origin_node_names if n in graph_node_names]
            [removed_names,retained_names]=split_name_list(graph.input,input_node_names)

            for name in removed_names:
                if name in input_map.keys():
                    graph.input.remove(input_map[name])  
           
            #input_map = generate_node_dict(graph.input)   
            # MODIFY OUTPUTS
            
            [removed_names,retained_names]=split_name_list(graph.output,output_node_names)
            for name in removed_names:
                if name in output_map.keys():
                    graph.output.remove(output_map[name])  
            output_map = generate_node_dict(graph.output)      

            valid_node_names = set(graph_node_names) |set(input_node_names) 
            
                    #for output_node_name in output_node_names:
            #    valid_node_names=traceUpNodes(graph,output_node_name,
            #                                 valid_node_names,node_map, start_index, initializer_map)
            #valid_node_names = list(input_node_names)
            #valid_node_names = set(valid_node_names)
            print ("-----------")

            #print ("valide_node names: ", valid_node_names)
            
            invalid_node_names = list( (set(node_map.keys()) | set(initializer_map.keys())) - set(valid_node_names))
            #print ("invalide_node names: ", invalid_node_names)
            
            
            removed_node_names = [n for n in node_map.keys() if n in invalid_node_names]
            #print ("removed_node: ", removed_node_names)
            
            
            # Remove all the invalid nodes from the graph               
            for name in invalid_node_names:       
                if name in initializer_map.keys():
                    graph.initializer.remove(initializer_map[name])
                #if name in input_map.keys():
                #    graph.input.remove(input_map[name])  
                if name in node_map.keys():
                    graph.node.remove(node_map[name]) 
                #if name in value_map.keys():
                #    graph.value_info.remove(value_map[name])  
            
            new_node_map = generate_node_dict(graph.node)
            new_input_map = generate_node_dict(graph.input)
            new_output_map = generate_node_dict(graph.output)
            new_initializer_map = generate_node_dict(graph.initializer)
            new_value_map = generate_node_dict(graph.value_info)
            new_input_all= []
            for i in graph.node:
                for input_name in i.input:
                    new_input_all.append(input_name)
            new_input_all = set(new_input_all)
            extra_tensor = list(set(new_input_all) - (set(new_node_map.keys()) | set(new_initializer_map.keys())))
            #print ("need input tensor: ", extra_tensor)
            if parts > 0:
                for per_tensor in extra_tensor:
                    new_tensor_input = onnx.helper.make_tensor_value_info(per_tensor,
                                           onnx.TensorProto.FLOAT,get_node_output_shape(value_map[per_tensor]))
                    graph.input.extend([new_tensor_input])
            #print("Output Model ",parts, " Generated. Errors: ", onnx.checker.check_model(model))
            mapping_dict = {}
            mapping_dict["export"+str(parts) +".onnx"] = list(new_node_map)            
            new_mapping_dict = copy.copy(mapping_dict)
            #mapping_dict_list.append(new_mapping_dict)
            mapping_dict_list["export"+str(parts) +".onnx"] = list(new_node_map)

            onnx.save(model, output_path)
            parts += 1
            start_index = part_index + 1
        part_index += 1  
        
    mapping_content = json.dumps(mapping_dict_list)
    jsonFile.write(mapping_content)
    jsonFile.close()
    print ("Generate ", parts, " models.")
        # SAVE MODEL


def horizontal_split(graph, splitnode_name, split_ranks, split_axis, split_points):
    # split_ranks : record all ranks(node) for distributing horizontally.
    # split_axis: (-1), LOP split. (0) split channel (1) Split Height (2) Split Width.
    # split_points: (-1) output channels split (0) input channel split. (1&2) output height width split.
    # e.g.:
    # split_axis = 1   means split the input data from the height dimension.
    # split_ranks = [0,1] means distribute the computation over rank 0, 1 (two computing nodes.)
    #split_points = [3,3000] means the output height is 3, and the other is 3000.
    split_starts = np.zeros(np.shape(split_ranks), dtype=int)
    split_ends = np.zeros(np.shape(split_ranks),  dtype=int)

   # model = onnx.load(input_model)
   # graph = model.graph

   # for n in graph.node:
   #     if n.name == '':
   #         n.name = str(n.output[0])
    
    nodeIdx = find_node_index(graph, splitnode_name)

    
    input_map = generate_node_dict(graph.input)
    output_map = generate_node_dict(graph.output)
    initializer_map = generate_node_dict(graph.initializer)
    value_map = generate_node_dict(graph.value_info)
    node_map = generate_node_dict(graph.node)
    node_inputs = [n for n in list(node_map[splitnode_name].input) if n not in initializer_map]
    # the output shape of the input layer.
    input_node_shape = get_node_input_shape(graph, graph.node[nodeIdx], value_map)

    split_node_shape = get_node_output_shape_safe(graph, splitnode_name, value_map)

   ###############
   ### Split Input
   ###############
    split_output_names = []
    for i in range(len(split_ranks)):
        split_output_names.append(splitnode_name+'_hsplit_'+str(split_ranks[i]))
    
    gemm_attributes = {'alpha':1.0, 'beta':1.0, 'transA':0, 'transB':1}
    conv_attributes = {'auto_pad':[],'dilations':[],'group':1,'kernel_shape':[],'pads':[],'strides':[]}
    fc_attributes = {}
    node_attribute_names = []
    for i in range(len(node_map[splitnode_name].attribute)):
        node_attribute_names.append(node_map[splitnode_name].attribute[i].name)

    # Get attributes of Conv layer, pads, kernel, stride, etc.

    if node_map[splitnode_name].op_type == 'Conv':
        for i in range(len(node_map[splitnode_name].attribute)):
            attribute_name = node_attribute_names[i]
            if node_map[splitnode_name].attribute[i].ints:
                conv_attributes[attribute_name] = node_map[splitnode_name].attribute[i].ints
            else:
                conv_attributes[attribute_name] = node_map[splitnode_name].attribute[i].i

    if node_map[splitnode_name].op_type == 'Gemm':   
        for i in range(len(node_map[splitnode_name].attribute)):
            attribute_name = node_attribute_names[i] 
            if node_map[splitnode_name].attribute[i].ints:                      
                gemm_attributes[attribute_name] = node_map[splitnode_name].attribute[i].ints
            else:                                    
                gemm_attributes[attribute_name] = node_map[splitnode_name].attribute[i].i

    weight_names = [name for name in list(node_map[splitnode_name].input) if name in initializer_map]
    for name in weight_names:
        if len(initializer_map[name].dims) == 4: # Cout * Cin * Kh * Kw.
            w = onnx.numpy_helper.to_array(initializer_map[name]) 
        if len(initializer_map[name].dims) == 2: # Cout * Cin
            w = onnx.numpy_helper.to_array(initializer_map[name]) 
        if len(initializer_map[name].dims) == 1: # Cout.
            b = onnx.numpy_helper.to_array(initializer_map[name])               

    if split_axis == -1: # do not split for the input.
        _offset = 0
        for i in range(len(split_ranks)):
            split_starts[i] = 0
            split_ends[i] = 0
            _offset = split_ends[i]
            
    if split_axis == 0: # channel
        _offset = 0

        for i in range(len(split_ranks)):
            split_starts[i] = _offset
            split_ends[i] = _offset + split_points[i]
            _offset = split_ends[i]

        if (conv_attributes['group'] == 2):            
            # split_points = np.array(split_points, dtype=int)
            # split_points = np.repeat(split_points, 2)
            split_starts = np.repeat(split_starts, 2)
            split_ends = np.repeat(split_ends, 2)
            for i in range(len(split_ranks)):
                split_starts[2*i+1] = split_starts[2*i+1] + np.shape(w)[1]
                split_ends[2*i+1] = split_ends[2*i+1] + np.shape(w)[1]


    if split_axis == 1: # height
        _offset = -conv_attributes['pads'][0] 
        for i in range(len(split_ranks)):
            split_starts[i] = max(_offset, 0)
            split_ends[i] = _offset + (split_points[i]-1) * conv_attributes['strides'][0] 
            _offset = split_ends[i] + conv_attributes['strides'][0]
            split_ends[i] = max(split_ends[i]+ conv_attributes['kernel_shape'][0], 0)
            
    if split_axis == 2: # width.
        _offset = -conv_attributes['pads'][0] 
        for i in range(len(split_ranks)):
            split_starts[i] = max(_offset, 0)
            split_ends[i] = _offset + (split_points[i]-1) * conv_attributes['strides'][0] 
            _offset = split_ends[i] + conv_attributes['strides'][0] 
            split_ends[i] = max(split_ends[i]+ conv_attributes['kernel_shape'][0], 0)
    
        
    if split_axis >= 0:
        ##conv_hsplit = onnx.helper.make_node(
        ##    name=splitnode_name + "_hsplit",  # Name is optional.
        ##    op_type = "NSplit",
        ##    inputs = node_inputs, # inputs
        ##    outputs=split_output_names, # outputs    
        ##)    
    # Create a node (NodeProto)
        conv_hsplit = onnx.helper.make_node(
            #name=splitnode_name + "_hsplit",  # Name is optional.
            name=split_output_names[0],  # Name is optional.
            op_type = "HSplit",
            inputs = node_inputs, # inputs
            outputs=split_output_names, # outputs    
            axis=split_axis,
            starts=split_starts,
            ends=split_ends,
            sranks=split_ranks,
            spoints=split_points,
        )
        graph.node.insert(nodeIdx, conv_hsplit)
        nodeIdx = nodeIdx + 1

        _offset = 0
        hsplit_shape = input_node_shape.copy()
        for i in range(len(split_ranks)):
            if split_axis == 0: # channel
                hsplit_shape[1] = min(input_node_shape[1] - _offset, split_points[i])
                _offset = _offset + hsplit_shape[1]
            if split_axis == 1: # height
                hsplit_shape[2] = min(input_node_shape[2] - _offset, split_starts[i] - split_ends[i])
                _offset = _offset + hsplit_shape[2]
            if split_axis == 2:
                hsplit_shape[3] = min(input_node_shape[3] - _offset, split_starts[i] - split_ends[i])
                _offset = _offset + hsplit_shape[3]

            graph.value_info.append(onnx.helper.make_tensor_value_info(split_output_names[i],
                                                                onnx.TensorProto.FLOAT,
                                                                hsplit_shape))





   ###############
   ### Generate Multiple Input: split_output_names
   ###############
    splitnode_weights = {}
    splitnode_bias = {}

   ###############
   ### split Computations...
   ###############

    if split_axis == -1:
        # split LOP
        # print ("Horizontally Split Layer (LOP) ", splitnode_name)
        conv_output_names = []
        weight_names = [name for name in list(node_map[splitnode_name].input) if name in initializer_map]
        for name in weight_names:
            if node_map[splitnode_name].op_type == 'Conv':
                if len(initializer_map[name].dims) == 4:
                    # should be weights not bias.
                    w = onnx.numpy_helper.to_array(initializer_map[name]) 
                    _offset = 0
                    for i in range(len(split_ranks)): 
                        # split into n ranks. n = len(conv1_1_split_ranks)
                        new_splitnode_weight_name = splitnode_name + '_w_' + str(i)
                        splitnode_weights[new_splitnode_weight_name] = w[_offset: split_points[i],:,:,:]
                        _offset = split_points[i]

            if node_map[splitnode_name].op_type == 'Gemm':
                if len(initializer_map[name].dims) == 2: # Cout * Cin
                    # should be weights not bias.
                    # print ("Split Fully-Connect weights Channel")

                    w = onnx.numpy_helper.to_array(initializer_map[name])
                    _offset = 0
                    for i in range(len(split_ranks)):
                        # split into n ranks. n = len(conv1_1_split_ranks)
                        new_splitnode_weight_name = splitnode_name + '_w_' + str(i)
                        splitnode_weights[new_splitnode_weight_name] = w[_offset: split_points[i],:]
                        _offset = split_points[i]
                                
            if len(initializer_map[name].dims) == 1:
                b = onnx.numpy_helper.to_array(initializer_map[name]) 
                _offset = 0
                for i in range(len(split_ranks)):                     
                    new_splitnode_bias_name = splitnode_name + '_b_'  + str(i)
                    splitnode_bias[new_splitnode_bias_name] = b[_offset: split_points[i]]
                    _offset = split_points[i]
    
    
        _valueoffset = 0
        for i in range(len(split_ranks)):            
            conv1_output_node_name = splitnode_name + '_splitco_' +str(split_ranks[i])
            conv_output_names.append(conv1_output_node_name)
            W_initializer_tensor_name = splitnode_name + '_w_' + str(i)
            W_initializer_tensor = create_initializer_tensor(
                name=W_initializer_tensor_name,
                tensor_array=splitnode_weights[W_initializer_tensor_name],
                data_type=onnx.TensorProto.FLOAT)            
            graph.initializer.append(W_initializer_tensor)
            W_value_info =  onnx.helper.make_tensor_value_info( W_initializer_tensor_name,
                                                                onnx.TensorProto.FLOAT,
                                                                np.shape(splitnode_weights[W_initializer_tensor_name]))
            graph.input.append(W_value_info)

            input_lst = [
                        node_inputs[0],
                        W_initializer_tensor_name,
            ]
            B_initializer_tensor_name = splitnode_name + '_b_' + str(i)
            if B_initializer_tensor_name in splitnode_bias:
                B_initializer_tensor = create_initializer_tensor(
                        name=B_initializer_tensor_name,
                        tensor_array=splitnode_bias[B_initializer_tensor_name],
                        data_type=onnx.TensorProto.FLOAT)
                graph.initializer.append(B_initializer_tensor)
                B_value_info =  onnx.helper.make_tensor_value_info( B_initializer_tensor_name,
                                                                onnx.TensorProto.FLOAT,
                                                                np.shape(splitnode_bias[B_initializer_tensor_name]))
                graph.input.append(B_value_info)
                input_lst.append(B_initializer_tensor_name)
            

            if node_map[splitnode_name].op_type == 'Conv':
                conv1_node = onnx.helper.make_node(
                            name=conv1_output_node_name,  # Name is optional.
                            op_type="Conv",
            # Must follow the order of input and output definitions.
            # https://github.com/onnx/onnx/blob/rel-1.9.0/docs/Operators.md#inputs-2---3
                            inputs=input_lst,
                            outputs=[conv1_output_node_name],
                            # The following arguments are attributes.
                    )
                for attr, val in conv_attributes.items():
                    if val:
                        conv1_node.attribute.extend([onnx.helper.make_attribute(attr, val)])


            if node_map[splitnode_name].op_type == 'Gemm':
                conv1_node = onnx.helper.make_node(
                                name=conv1_output_node_name,  # Name is optional.
                                op_type="Gemm",
                # Must follow the order of input and output definitions.
                # https://github.com/onnx/onnx/blob/rel-1.9.0/docs/Operators.md#inputs-2---3
                    inputs=input_lst,
                    outputs=[conv1_output_node_name],
                # The following arguments are attributes.
                )
                for attr, val in gemm_attributes.items():
                    if val:
                        conv1_node.attribute.extend([onnx.helper.make_attribute(attr, val)])
            
            graph.node.insert(nodeIdx, conv1_node)
            nodeIdx = nodeIdx + 1
            # add shape info of outputs
            hsplit_shape = split_node_shape.copy()
            # hsplit_shape[1] = min(split_node_shape[1] - _valueoffset, split_points[i])
            hsplit_shape[1] = split_points[i] - _valueoffset
            _valueoffset = _valueoffset + hsplit_shape[1]
            graph.value_info.append(onnx.helper.make_tensor_value_info(conv1_output_node_name,
                                                            onnx.TensorProto.FLOAT,
                                                            hsplit_shape))

        # Create a node (NodeProto)
        conv_hsum = onnx.helper.make_node(
            name=splitnode_name,  # Name is optional.
            op_type = "Concat",
            inputs = conv_output_names, # inputs
            outputs= node_map[splitnode_name].output, # outputs    
            axis=1
        )
        for name in weight_names:
            graph.input.remove(input_map[name])
            graph.initializer.remove(initializer_map[name])
        graph.node.insert(nodeIdx,conv_hsum)
        nodeIdx = nodeIdx + 1

    if split_axis == 0:  
        # print ("Split Layer (LIP--Channel) ", splitnode_name)
        conv_output_names = []
        weight_names = [name for name in list(node_map[splitnode_name].input) if name in initializer_map]
        for name in weight_names:
            if node_map[splitnode_name].op_type == 'Conv':
                if len(initializer_map[name].dims) == 4:
                    # should be weights not bias. 
                    w = onnx.numpy_helper.to_array(initializer_map[name]) 
                    _offset = 0
                    for i in range(len(split_ranks)): 
                        # split into n ranks. n = len(conv1_1_split_ranks)
                        new_splitnode_weight_name = splitnode_name + '_w_' + str(i)
                        splitnode_weights[new_splitnode_weight_name] = w[:,_offset: split_points[i]+_offset,:,:]
                        _offset = _offset + split_points[i]


            if node_map[splitnode_name].op_type == 'Gemm':
                if len(initializer_map[name].dims) == 2:
                    # should be weights not bias. 
                    w = onnx.numpy_helper.to_array(initializer_map[name]) 
                    _offset = 0
                    for i in range(len(split_ranks)): 
                        # split into n ranks. n = len(conv1_1_split_ranks)
                        new_splitnode_weight_name = splitnode_name + '_w_' + str(i)
                        splitnode_weights[new_splitnode_weight_name] = w[:,_offset: split_points[i]+_offset]
                        _offset = _offset + split_points[i]
                  

            if len(initializer_map[name].dims) == 1:
                b = onnx.numpy_helper.to_array(initializer_map[name]) 
                new_splitnode_bias_name = splitnode_name + '_b_0' 
                splitnode_bias[new_splitnode_bias_name] = b
    
    
        #conv_attributes   
        for i in range(len(split_ranks)):            
            conv1_output_node_name = splitnode_name + '_splitic_' +str(split_ranks[i]) 
            conv_output_names.append(conv1_output_node_name)
            # default the first rank(node) with bias.
            W_initializer_tensor_name = splitnode_name + '_w_' + str(i)
            W_initializer_tensor = create_initializer_tensor(
                name=W_initializer_tensor_name,
                tensor_array=splitnode_weights[W_initializer_tensor_name],
                data_type=onnx.TensorProto.FLOAT)            
            
            B_initializer_tensor_name = splitnode_name + '_b_' + str(i)
            if i==0:    
                B_initializer_tensor = create_initializer_tensor(
                    name=B_initializer_tensor_name,
                    tensor_array=splitnode_bias[B_initializer_tensor_name],
                    data_type=onnx.TensorProto.FLOAT)
                B_value_info =  onnx.helper.make_tensor_value_info( B_initializer_tensor_name,
                                                                onnx.TensorProto.FLOAT,
                                                                np.shape(splitnode_bias[B_initializer_tensor_name]))
                graph.input.append(B_value_info)
            else:
                B_initializer_tensor = create_initializer_tensor(
                    name=B_initializer_tensor_name,
                    tensor_array=np.zeros(np.shape(b),dtype=float),
                    data_type=onnx.TensorProto.FLOAT)
                B_value_info =  onnx.helper.make_tensor_value_info( B_initializer_tensor_name,
                                                                onnx.TensorProto.FLOAT,
                                                                np.shape(b))
                graph.input.append(B_value_info)
                        
            graph.initializer.append(W_initializer_tensor)
            graph.initializer.append(B_initializer_tensor)
            W_value_info =  onnx.helper.make_tensor_value_info( W_initializer_tensor_name,
                                                                onnx.TensorProto.FLOAT,
                                                                np.shape(splitnode_weights[W_initializer_tensor_name]))
            graph.input.append(W_value_info)
            if node_map[splitnode_name].op_type == 'Conv':
                conv1_node = onnx.helper.make_node(
                                name=conv1_output_node_name,  # Name is optional.
                                op_type="Conv",
                                # Must follow the order of input and output definitions.
                                # https://github.com/onnx/onnx/blob/rel-1.9.0/docs/Operators.md#inputs-2---3
                                inputs=[
                                        splitnode_name+'_hsplit_'+str(split_ranks[i]), W_initializer_tensor_name,
                                        B_initializer_tensor_name],
                                outputs=[conv1_output_node_name],
                                 # The following arguments are attributes.
                                auto_pad =  conv_attributes['auto_pad'],
                                dilations = conv_attributes['dilations'],
                                group = conv_attributes['group'],
                                kernel_shape = conv_attributes['kernel_shape'],
                                pads = conv_attributes['pads'],
                                strides = conv_attributes['strides']
                                )
            if node_map[splitnode_name].op_type == 'Gemm':
                conv1_node = onnx.helper.make_node(
                                name=conv1_output_node_name,  # Name is optional.
                                op_type="Gemm",
                                # Must follow the order of input and output definitions.
                                # https://github.com/onnx/onnx/blob/rel-1.9.0/docs/Operators.md#inputs-2---3
                                inputs=[
                                        splitnode_name+'_hsplit_'+str(split_ranks[i]) , W_initializer_tensor_name,
                                        B_initializer_tensor_name],
                                outputs=[conv1_output_node_name],
                    alpha = gemm_attributes['alpha'],
                    beta = gemm_attributes['beta'],
                    transA = gemm_attributes['transA'],
                    transB = gemm_attributes['transB']
                                )
            graph.node.insert(nodeIdx, conv1_node)
            nodeIdx = nodeIdx + 1
            # add shape info of outputs
            hsplit_shape = split_node_shape.copy()
            graph.value_info.append(onnx.helper.make_tensor_value_info(conv1_output_node_name,
                                                                onnx.TensorProto.FLOAT,
                                                                hsplit_shape))
        # Create a node (NodeProto)
        conv_hsum = onnx.helper.make_node(
            name=splitnode_name,  # Name is optional.
            op_type = "Sum",
            inputs = conv_output_names, # inputs
            outputs= node_map[splitnode_name].output, # outputs    
            #axis=split_axis,
            #sranks=split_ranks,
        )
        for name in weight_names:
            graph.input.remove(input_map[name])
            graph.initializer.remove(initializer_map[name])
        graph.node.insert(nodeIdx,conv_hsum)
        nodeIdx = nodeIdx + 1
    
    if split_axis == 1:  
        # print ("Split Convolution Layer (LIP--Height) ", splitnode_name)
        conv_output_names = []
        weight_names = [name for name in list(node_map[splitnode_name].input) if name in initializer_map]
        for name in weight_names:
            if node_map[splitnode_name].op_type == 'Conv':
                if len(initializer_map[name].dims) == 4:
                    # should be weights not bias.
                    w = onnx.numpy_helper.to_array(initializer_map[name]) 
                    for i in range(len(split_ranks)): 
                        # split into n ranks. n = len(conv1_1_split_ranks)
                        new_splitnode_weight_name = splitnode_name + '_w_' + str(i)
                        splitnode_weights[new_splitnode_weight_name] = w
                    
                                
            if len(initializer_map[name].dims) == 1:
                b = onnx.numpy_helper.to_array(initializer_map[name]) 
                _offset = 0                
                for i in range(len(split_ranks)): 
                    new_splitnode_bias_name = splitnode_name + '_b_' + str(i)
                    splitnode_bias[new_splitnode_bias_name] = b
    
    
        #conv_attributes   
        for i in range(len(split_ranks)):            
            conv1_output_node_name = splitnode_name + '_splith_' +str(split_ranks[i]) 
            conv_output_names.append(conv1_output_node_name)
            # default the first rank(node) with bias.
            W_initializer_tensor_name = splitnode_name + '_w_' + str(i)
            W_initializer_tensor = create_initializer_tensor(
                name=W_initializer_tensor_name,
                tensor_array=splitnode_weights[W_initializer_tensor_name],
                data_type=onnx.TensorProto.FLOAT)            
            
            B_initializer_tensor_name = splitnode_name + '_b_' + str(i)
            B_initializer_tensor = create_initializer_tensor(
                    name=B_initializer_tensor_name,
                    tensor_array=splitnode_bias[B_initializer_tensor_name],
                    data_type=onnx.TensorProto.FLOAT)
                        
            graph.initializer.append(W_initializer_tensor)
            graph.initializer.append(B_initializer_tensor)
            W_value_info =  onnx.helper.make_tensor_value_info( W_initializer_tensor_name,
                                                                onnx.TensorProto.FLOAT,
                                                                np.shape(splitnode_weights[W_initializer_tensor_name]))
            graph.input.append(W_value_info)
            B_value_info =  onnx.helper.make_tensor_value_info( B_initializer_tensor_name,
                                                                onnx.TensorProto.FLOAT,
                                                                np.shape(splitnode_bias[B_initializer_tensor_name]))
            graph.input.append(B_value_info)
    
            conv1_node = onnx.helper.make_node(
                            name=conv1_output_node_name,  # Name is optional.
                            op_type="Conv",
            # Must follow the order of input and output definitions.
            # https://github.com/onnx/onnx/blob/rel-1.9.0/docs/Operators.md#inputs-2---3
                inputs=[
                        splitnode_name+'_hsplit_'+str(split_ranks[i]), W_initializer_tensor_name,
                        B_initializer_tensor_name],
                outputs=[conv1_output_node_name],
            # The following arguments are attributes.
                auto_pad =  conv_attributes['auto_pad'],
                dilations = conv_attributes['dilations'],
                group = conv_attributes['group'],
                kernel_shape = conv_attributes['kernel_shape'],
                pads = conv_attributes['pads'],
                strides = conv_attributes['strides']
            )
            
            graph.node.insert(nodeIdx, conv1_node)
            nodeIdx = nodeIdx + 1
            # add shape info of outputs
            _offset = 0
            hsplit_shape = split_node_shape.copy()
            for i in range(len(split_ranks)):
                hsplit_shape[2] = min(input_node_shape[2] - _offset, split_points[i])
                _offset = _offset + hsplit_shape[2]
                graph.value_info.append(onnx.helper.make_tensor_value_info(conv1_output_node_name,
                                                                onnx.TensorProto.FLOAT,
                                                                hsplit_shape))




        # Create a node (NodeProto)
        conv_hsum = onnx.helper.make_node(
            #name=splitnode_name + "_hconcat",  # Name is optional.
            name=splitnode_name,  # Name is optional.
            op_type = "Concat",
            inputs = conv_output_names, # inputs
            outputs= node_map[splitnode_name].output, # outputs    
            axis=2,
         )
        for name in weight_names:
            graph.input.remove(input_map[name])
            graph.initializer.remove(initializer_map[name])
        graph.node.insert(nodeIdx,conv_hsum)
        nodeIdx = nodeIdx + 1
        
        
    if split_axis == 2:  
        # print ("Split Convolution Layer (LIP--Width) ", splitnode_name)
        conv_output_names = []
        weight_names = [name for name in list(node_map[splitnode_name].input) if name in initializer_map]
        for name in weight_names:
            if len(initializer_map[name].dims) == 4:
                # should be weights not bias.
                w = onnx.numpy_helper.to_array(initializer_map[name]) 
                for i in range(len(split_ranks)): 
                    # split into n ranks. n = len(conv1_1_split_ranks)
                    new_splitnode_weight_name = splitnode_name + '_w_' + str(i)
                    splitnode_weights[new_splitnode_weight_name] = w
                    
                                
            if len(initializer_map[name].dims) == 1:                                
                b = onnx.numpy_helper.to_array(initializer_map[name]) 
                for i in range(len(split_ranks)): 
                    b = onnx.numpy_helper.to_array(initializer_map[name]) 
                    new_splitnode_bias_name = splitnode_name + '_b_' + str(i)
                    splitnode_bias[new_splitnode_bias_name] = b
    
    
        #conv_attributes   
        for i in range(len(split_ranks)):            
            conv1_output_node_name = splitnode_name + '_splitw_' +str(i)
            conv_output_names.append(conv1_output_node_name)
            # default the first rank(node) with bias.
            W_initializer_tensor_name = splitnode_name + '_w_' + str(i)
            W_initializer_tensor = create_initializer_tensor(
                name=W_initializer_tensor_name,
                tensor_array=splitnode_weights[W_initializer_tensor_name],
                data_type=onnx.TensorProto.FLOAT)            
            
            B_initializer_tensor_name = splitnode_name + '_b_' + str(i)
            B_initializer_tensor = create_initializer_tensor(
                    name=B_initializer_tensor_name,
                    tensor_array=splitnode_bias[B_initializer_tensor_name],
                    data_type=onnx.TensorProto.FLOAT)
                        
            graph.initializer.append(W_initializer_tensor)
            graph.initializer.append(B_initializer_tensor)
            W_value_info =  onnx.helper.make_tensor_value_info( W_initializer_tensor_name,
                                                                onnx.TensorProto.FLOAT,
                                                                np.shape(splitnode_weights[W_initializer_tensor_name]))
            graph.input.append(W_value_info)
            B_value_info =  onnx.helper.make_tensor_value_info( B_initializer_tensor_name,
                                                                onnx.TensorProto.FLOAT,
                                                                np.shape(splitnode_bias[B_initializer_tensor_name]))
            graph.input.append(B_value_info)
    
            conv1_node = onnx.helper.make_node(
                            name=conv1_output_node_name,  # Name is optional.
                            op_type="Conv",
            # Must follow the order of input and output definitions.
            # https://github.com/onnx/onnx/blob/rel-1.9.0/docs/Operators.md#inputs-2---3
                inputs=[
                        splitnode_name+'_hsplit_'+str(split_ranks[i]) , W_initializer_tensor_name,
                        B_initializer_tensor_name],
                outputs=[conv1_output_node_name],
            # The following arguments are attributes.
                auto_pad =  conv_attributes['auto_pad'],
                dilations = conv_attributes['dilations'],
                group = conv_attributes['group'],
                kernel_shape = conv_attributes['kernel_shape'],
                pads = conv_attributes['pads'],
                strides = conv_attributes['strides']
            )
            
            graph.node.insert(nodeIdx, conv1_node)
            nodeIdx = nodeIdx + 1
            # add shape info of outputs
            _offset = 0
            hsplit_shape = split_node_shape.copy()
            for i in range(len(split_ranks)):
                hsplit_shape[3] = min(input_node_shape[3] - _offset, split_points[i])
                _offset = _offset + hsplit_shape[3]
                graph.value_info.append(onnx.helper.make_tensor_value_info(conv1_output_node_name,
                                                                onnx.TensorProto.FLOAT,
                                                                hsplit_shape))


        # Create a node (NodeProto)
        conv_hsum = onnx.helper.make_node(
            #name=splitnode_name + "_wconcat",  # Name is optional.
            name=splitnode_name ,  # Name is optional.
            op_type = "Concat",
            inputs = conv_output_names, # inputs
            outputs= node_map[splitnode_name].output, # outputs    
            axis=3,
         )
        for name in weight_names:
            graph.input.remove(input_map[name])
            graph.initializer.remove(initializer_map[name])
        graph.node.insert(nodeIdx,conv_hsum)
        nodeIdx = nodeIdx + 1
        
    nodeIdx = find_node_index(graph, splitnode_name)
    if nodeIdx>=0:
        graph.node.remove(node_map[splitnode_name])    


    #onnx.save(model, output_model)
    return graph
    
    
    
