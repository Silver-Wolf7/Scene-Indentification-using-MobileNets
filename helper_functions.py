import numpy as np
import os
import cv2
import re
from keras.models import Model
from datetime import datetime

def list_labels(file):
    labels_file = open(file, "r")
    labels = []
    
    for line in labels_file:
        label = line.strip()
        labels.append(label)
    
    labels_file.close()
    
    return labels

def load_data(folder, class_labels_name):
    Category = ["training", "test", "validation"]
    output = []
    
    for category in Category:
        print("Loading {}".format(category))
        path = os.path.join(folder, category)
        print(path)
        images = []
        labels = []
        
        for sub_folder in os.listdir(path):
            label = class_labels_name[sub_folder]
            
            #Iterating through all images
            for file in os.listdir(os.path.join(path, sub_folder)):
                
                #getting the image path
                img_path = os.path.join(os.path.join(path, sub_folder), file)
                
                #appending image and corresponding label
                images.append(cv2.resize(cv2.imread(img_path), (224, 224)))
                labels.append(label)
            
        images = (np.array(images, dtype='float32')/127.5)-1
        labels = np.array(labels, dtype='int8')
        
        output.append((images, labels))
        
    return output


def inference_time(model, test_images, test_labels):
    # the first time is just to warm up the hardware
    model.evaluate(test_images, test_labels)
    
    # measuring inference time 3 times then taking average
    start = datetime.now()
    model.evaluate(test_images, test_labels)
    end = datetime.now()
    time = end-start
    
    start2 = datetime.now()
    model.evaluate(test_images, test_labels)
    end2 = datetime.now()
    time2 = end2-start2

    start3 = datetime.now()
    model.evaluate(test_images, test_labels)
    end3 = datetime.now()
    time3 = end3-start3
    
    return (time + time2 + time3) / 3


#cited from https://stackoverflow.com/questions/49492255/how-to-replace-or-insert-intermediate-layer-in-keras-model
def insert_layer_nonseq(model, layer_regex, insert_layer_factory,
                        insert_layer_name=None, position='after'):

    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: model.input})

    # Iterate over all layers after the input
    model_outputs = []
    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux] 
                for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if name matches the regular expression
        if re.match(layer_regex, layer.name):
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before, after or replace')

            new_layer = insert_layer_factory()
            if insert_layer_name:
                new_layer._name = insert_layer_name
            else:
                x = new_layer(x)
 
            if position == 'before':
                x = layer(x)
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

        # Save tensor in output list if it is output in initial model
        if layer_name in model.output_names:
            model_outputs.append(x)

    return Model(inputs=model.inputs, outputs=model_outputs)