import torch
import torch.nn as nn

def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01
    
    # Hidden layer parameters
    W_xh = normal(((num_inputs, num_hiddens)))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)

    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for p in params:
        p.requires_grad_(True)
    return params

def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),)

def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []

    for X in inputs:
        H = torch.tanh(X @ W_xh + H @ W_hh + b_h)
        Y = H @ W_hq + b_q
        outputs.append(Y)

    # Stack all time steps: (num_steps * batch_size, vocab_size)
    return torch.cat(outputs, dim=0), (H,)

class RNNModel:
    def __init__(self, vocab_size, num_hiddens, device):
        self.vocab_size  = vocab_size
        self.num_hiddens = num_hiddens
        self.params      = get_params(vocab_size, num_hiddens, device)

    def __call__(self, X, state):
        # X comes in as token indices: (batch_size, num_steps)
        # Transpose to (num_steps, batch_size), then one-hot encode
        X = nn.functional.one_hot(X.T, self.vocab_size).float()
        return rnn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return init_rnn_state(batch_size, self.num_hiddens, device)
