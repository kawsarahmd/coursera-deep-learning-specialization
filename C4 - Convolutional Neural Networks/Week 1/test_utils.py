import numpy as np
import torch
import torch.nn as nn


def summary(model):
    """
    Generate a summary of a PyTorch model similar to Keras model.summary()

    Arguments:
    model -- PyTorch model (nn.Module)

    Returns:
    model_summary -- list of layer information
    """
    model_summary = []

    for name, layer in model.named_modules():
        if len(list(layer.children())) == 0 and name != '':  # Skip container modules
            layer_info = [layer.__class__.__name__]

            # Get output shape (None for batch dimension)
            layer_info.append((None,))  # Placeholder for shape

            # Count parameters
            params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
            layer_info.append(params)

            # Add additional layer-specific info
            if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                layer_info.append(layer.padding if isinstance(layer.padding, str) else 'valid' if layer.padding == (0, 0) else 'same')
                layer_info.append('linear')  # activation
                layer_info.append('GlorotUniform')  # weight initializer
            elif isinstance(layer, nn.MaxPool2d):
                layer_info.append(layer.kernel_size if isinstance(layer.kernel_size, tuple) else (layer.kernel_size, layer.kernel_size))
                layer_info.append(layer.stride if isinstance(layer.stride, tuple) else (layer.stride, layer.stride))
                layer_info.append('valid' if layer.padding == 0 else 'same')
            elif isinstance(layer, nn.Dense if hasattr(nn, 'Dense') else nn.Linear):
                activation = 'sigmoid' if hasattr(layer, 'activation') else 'linear'
                if hasattr(layer, 'activation'):
                    layer_info.append(layer.activation)

            model_summary.append(layer_info)

    return model_summary


def comparator(learner, instructor):
    """
    Compare learner's output with instructor's output

    Arguments:
    learner -- learner's model summary
    instructor -- expected model summary

    Returns:
    None (prints comparison results)
    """
    if len(learner) != len(instructor):
        print(f"\033[91mMismatch in number of layers: {len(learner)} vs {len(instructor)}\033[0m")
        return

    for i, (l, ins) in enumerate(zip(learner, instructor)):
        if l[0] != ins[0]:  # Compare layer type
            print(f"\033[91mLayer {i}: Type mismatch - {l[0]} vs {ins[0]}\033[0m")
            return

    print("\033[92mAll tests passed!\033[0m")


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
            np.testing.assert_array_almost_equal(
                target_output, expected_output)
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
                assert test_case['expected'].shape == target(
                    *test_case['input']).shape
                success += 1
        except:
            print("Error: " + test_case['error'])

    if success == len(test_cases):
        print("\033[92m All tests passed.")
    else:
        print('\033[92m', success, " Tests passed")
        print('\033[91m', len(test_cases) - success, " Tests failed")
        raise AssertionError(
            "Not all tests were passed for {}. Check your equations and avoid using global variables inside the function.".format(target.__name__))


def multiple_test(test_cases, target):
    success = 0
    for test_case in test_cases:
        try:
            target_answer = target(*test_case['input'])
            if test_case['name'] == "datatype_check":
                success += datatype_check(test_case['expected'],
                                          target_answer, test_case['error'])
            if test_case['name'] == "equation_output_check":
                success += equation_output_check(
                    test_case['expected'], target_answer, test_case['error'])
            if test_case['name'] == "shape_check":
                success += shape_check(test_case['expected'],
                                       target_answer, test_case['error'])
        except:
            print("Error: " + test_case['error'])

    if success == len(test_cases):
        print("\033[92m All tests passed.")
    else:
        print('\033[92m', success, " Tests passed")
        print('\033[91m', len(test_cases) - success, " Tests failed")
        raise AssertionError(
            "Not all tests were passed for {}. Check your equations and avoid using global variables inside the function.".format(target.__name__))
