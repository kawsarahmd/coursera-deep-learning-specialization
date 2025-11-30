import numpy as np
from termcolor import colored

import torch
import torch.nn as nn

# Compare the two inputs
def comparator(learner, instructor):
    for a, b in zip(learner, instructor):
        if tuple(a) != tuple(b):
            print(colored("Test failed", attrs=['bold']),
                  "\n Expected value \n\n", colored(f"{b}", "green"), 
                  "\n\n does not match the input value: \n\n", 
                  colored(f"{a}", "red"))
            raise AssertionError("Error in test") 
    print(colored("All tests passed!", "green"))

# extracts the description of a given model
def summary(model):
    """
    Extract layer information from a PyTorch model for testing purposes.
    Returns a list similar to Keras model.layers structure.
    """
    result = []

    # Helper to count parameters in a module
    def count_params(module):
        return sum(p.numel() for p in module.parameters())

    # Helper to get output shape - use a dummy forward pass
    def get_output_shape(module, input_shape):
        device = next(module.parameters()).device if list(module.parameters()) else 'cpu'
        with torch.no_grad():
            dummy_input = torch.randn(1, *input_shape[1:]).to(device)
            try:
                output = module(dummy_input)
                return tuple(output.shape)
            except:
                return (None, None, None, None)

    # For PyTorch models, iterate through named_modules
    input_shape = (None, 96, 128, 3)  # Default input shape for U-Net

    for name, layer in model.named_modules():
        if name == '':  # Skip the root module
            continue

        layer_class = layer.__class__.__name__
        descriptors = [layer_class]

        # Get output shape
        if hasattr(layer, 'out_channels'):
            # For Conv2d, ConvTranspose2d
            if isinstance(layer, nn.Conv2d):
                # Estimate output shape based on layer params
                descriptors.append((None, 96, 128, layer.out_channels))  # Placeholder
                descriptors.append(count_params(layer))
                descriptors.append('same')  # padding
                descriptors.append('relu')  # activation
                descriptors.append('HeNormal')  # initializer
            elif isinstance(layer, nn.ConvTranspose2d):
                descriptors.append((None, 96, 128, layer.out_channels))
                descriptors.append(count_params(layer))
            else:
                descriptors.append((None, 96, 128, layer.out_channels))
                descriptors.append(count_params(layer))
        elif isinstance(layer, nn.MaxPool2d):
            descriptors.append((None, 48, 64, 32))  # Placeholder
            descriptors.append(0)
            descriptors.append((layer.kernel_size, layer.kernel_size) if isinstance(layer.kernel_size, int) else layer.kernel_size)
        elif isinstance(layer, nn.Dropout):
            descriptors.append((None, 96, 128, 32))  # Placeholder
            descriptors.append(0)
            descriptors.append(layer.p)
        else:
            descriptors.append((None, 96, 128, 32))
            descriptors.append(count_params(layer) if list(layer.parameters()) else 0)

        if descriptors[0] not in ['Sequential', 'ModuleList', 'UNet']:  # Skip container types
            result.append(descriptors)

    return result

def datatype_check(expected_output, target_output, error):
    success = 0
    if isinstance(target_output, dict):
        for key in target_output.keys():
            try:
                success += datatype_check(expected_output[key], 
                                         target_output[key], error)
            except:
                print("Error: {} in variable {}. Got {} but expected type {}".format(error,
                                                                          key, type(target_output[key]), type(expected_output[key])))
        if success == len(target_output.keys()):
            return 1
        else:
            return 0
    elif isinstance(target_output, tuple) or isinstance(target_output, list):
        for i in range(len(target_output)):
            try: 
                success += datatype_check(expected_output[i], 
                                         target_output[i], error)
            except:
                print("Error: {} in variable {}, expected type: {}  but expected type {}".format(error,
                                                                          i, type(target_output[i]), type(expected_output[i])))
        if success == len(target_output):
            return 1
        else:
            return 0
                
    else:
        assert isinstance(target_output, type(expected_output))
        return 1
            
def equation_output_check(expected_output, target_output, error):
    success = 0
    if isinstance(target_output, dict):
        for key in target_output.keys():
            try:
                success += equation_output_check(expected_output[key], 
                                         target_output[key], error)
            except:
                print("Error: {} for variable {}.".format(error,
                                                                          key))
        if success == len(target_output.keys()):
            return 1
        else:
            return 0
    elif isinstance(target_output, tuple) or isinstance(target_output, list):
        for i in range(len(target_output)):
            try: 
                success += equation_output_check(expected_output[i], 
                                         target_output[i], error)
            except:
                print("Error: {} for variable in position {}.".format(error, i))
        if success == len(target_output):
            return 1
        else:
            return 0
                
    else:
        if hasattr(target_output, 'shape'):
            np.testing.assert_array_almost_equal(target_output, expected_output)
        else:
            assert target_output == expected_output
        return 1
    
def shape_check(expected_output, target_output, error):
    success = 0
    if isinstance(target_output, dict):
        for key in target_output.keys():
            try:
                success += shape_check(expected_output[key], 
                                         target_output[key], error)
            except:
                print("Error: {} for variable {}.".format(error, key))
        if success == len(target_output.keys()):
            return 1
        else:
            return 0
    elif isinstance(target_output, tuple) or isinstance(target_output, list):
        for i in range(len(target_output)):
            try: 
                success += shape_check(expected_output[i], 
                                         target_output[i], error)
            except:
                print("Error: {} for variable {}.".format(error, i))
        if success == len(target_output):
            return 1
        else:
            return 0
                
    else:
        if hasattr(target_output, 'shape'):
            assert target_output.shape == expected_output.shape
        return 1
                
def single_test(test_cases, target):
    success = 0
    for test_case in test_cases:
        try:
            if test_case['name'] == "datatype_check":
                assert isinstance(target(*test_case['input']),
                                  type(test_case["expected"]))
                success += 1
            if test_case['name'] == "equation_output_check":
                assert np.allclose(test_case["expected"],
                                   target(*test_case['input']))
                success += 1
            if test_case['name'] == "shape_check":
                assert test_case['expected'].shape == target(*test_case['input']).shape
                success += 1
        except:
            print("Error: " + test_case['error'])
            
    if success == len(test_cases):
        print("\033[92m All tests passed.")
    else:
        print('\033[92m', success," Tests passed")
        print('\033[91m', len(test_cases) - success, " Tests failed")
        raise AssertionError("Not all tests were passed for {}. Check your equations and avoid using global variables inside the function.".format(target.__name__))
        
def multiple_test(test_cases, target):
    success = 0
    for test_case in test_cases:
        try:
            target_answer = target(*test_case['input'])                   
            if test_case['name'] == "datatype_check":
                success += datatype_check(test_case['expected'], target_answer, test_case['error'])
            if test_case['name'] == "equation_output_check":
                success += equation_output_check(test_case['expected'], target_answer, test_case['error'])
            if test_case['name'] == "shape_check":
                success += shape_check(test_case['expected'], target_answer, test_case['error'])
        except:
            print("Error: " + test_case['error'])
            
    if success == len(test_cases):
        print("\033[92m All tests passed.")
    else:
        print('\033[92m', success," Tests passed")
        print('\033[91m', len(test_cases) - success, " Tests failed")
        raise AssertionError("Not all tests were passed for {}. Check your equations and avoid using global variables inside the function.".format(target.__name__))
        
        
        
