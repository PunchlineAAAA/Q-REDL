import torch

def encode_complex(X, method="rect"):
    if method == "polar":
        return torch.complex(torch.cos(X), torch.sin(X))
    elif method == "rect":
        return torch.complex(X, torch.zeros_like(X))
    else:
        return X
