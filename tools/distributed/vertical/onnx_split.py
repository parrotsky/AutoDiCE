import onnx
from onnx import helper, checker
from onnx import TensorProto
import re
import argparse
import json,copy
import numpy as np
from data_json import *

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
        print ("Check Model or Path Exists.")
    return model_name
############## 标准读取 onnx model ################
##############################


def load_onnx(input_model):
    model = onnx.load(input_model)
    model = onnx.shape_inference.infer_shapes(model)

    graph = model.graph

    print("Check input model Errors: ", onnx.checker.check_model(model))

    #Generate a name for all node if they have none.
    nodeIdx = 0;
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
        shape.append(i.dim_value)
    #return n,c,h,w
    return shape 
    #new_shape = shape[0:1]+shape[2:4]+shape[1:2]
    #return new_shape



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
def onnx_extract(input_model, output_model, sub_node_names_list):

#     origin_graph = load_onnx(input_model)
#     origin_input_map = generate_node_dict(origin_graph.input)
#     origin_output_map = generate_node_dict(origin_graph.output)
#     origin_initializer_map = generate_node_dict(origin_graph.initializer)
#     origin_value_map = generate_node_dict(origin_graph.value_info)
#     origin_node_map = generate_node_dict(origin_graph.node)

    model = onnx.load(input_model)
    model = onnx.shape_inference.infer_shapes(model)

    graph = model.graph

   # print("Check input model Errors: ", onnx.checker.check_model(model))

    #Generate a name for all node if they have none.
    nodeIdx = 0;
    for n in graph.node:
        if n.name == '':
            n.name = str(n.output[0])
    input_tensors = getInputlayers(input_model)
    input_map = generate_node_dict(graph.input)
    output_map = generate_node_dict(graph.output)
    initializer_map = generate_node_dict(graph.initializer)
    value_map = generate_node_dict(graph.value_info)
    node_map = generate_node_dict(graph.node)

        #print (newmodel_filename)

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
        new_tensor_input = onnx.helper.make_tensor_value_info(per_tensor,
                               onnx.TensorProto.FLOAT,get_node_output_shape(value_map[per_tensor]))
        graph.input.extend([new_tensor_input])

    print("New Output Model", str(output_model), " Generated. Errors: ", onnx.checker.check_model(model))

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
    model = onnx.shape_inference.infer_shapes(model)

    graph = model.graph
    
    print(" Check input model Errors: ", onnx.checker.check_model(model))
    
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
            print ("need input tensor: ", extra_tensor)
            if parts > 0:
                for per_tensor in extra_tensor:
                    new_tensor_input = onnx.helper.make_tensor_value_info(per_tensor,
                                           onnx.TensorProto.FLOAT,get_node_output_shape(value_map[per_tensor]))
                    graph.input.extend([new_tensor_input])
            print("Output Model ",parts, " Generated. Errors: ", onnx.checker.check_model(model))
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

