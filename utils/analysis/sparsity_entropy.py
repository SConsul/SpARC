import numpy as np

def get_sparsity_entropy(model, activations, threshold):
    elt_sparsity = 0
    input_sparsity = 0
    output_sparsity = 0

    return elt_sparsity, input_sparsity, output_sparsity