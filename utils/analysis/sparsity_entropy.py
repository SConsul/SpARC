import numpy as np

def get_sparsity_entropy(model, activations, threshold):
    nonzero = total = 0
    enc_nonzero = enc_total = 0
    dec_nonzero = dec_total = 0
    for name, output in activations.items():
        tensor = output.data.cpu().numpy()
        new_mask = np.where(abs(tensor) <= threshold, 0, tensor)
        tensor = np.abs(new_mask)
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)

        if 'encoder' in name:
            enc_nonzero += nz_count
            enc_total += total_params
        elif 'decoder' in name:
            dec_nonzero += nz_count
            dec_total += total_params

        nonzero += nz_count
        total += total_params
            
    total_sparsity = (total - nonzero) / total
    enc_sparsity = (enc_total - enc_nonzero) / enc_total
    dec_sparsity = (dec_total - dec_nonzero) / dec_total

    print(f"Sparsity total: {total_sparsity:.3f}, encoder: {enc_sparsity:.3f}, decoder: {dec_sparsity:.3f}")

    return total_sparsity, enc_sparsity, dec_sparsity