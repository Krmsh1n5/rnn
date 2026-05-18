import torch
import pytest
import math
import torch.nn as nn
from rnn.model import RNNModel
from rnn.train import train_epoch
from rnn.data import load_data_time_machine

DEVICE = torch.device("cpu")

def test_output_shape():
    vocab_size = 28
    num_hiddens = 512
    batch_size = 2
    num_steps = 5

    model = RNNModel(vocab_size, num_hiddens, DEVICE)
    state = model.begin_state(batch_size, DEVICE)
    X = torch.randint(0, vocab_size, (batch_size, num_steps))

    Y, new_state = model(X, state)

    assert Y.shape == (num_steps * batch_size, vocab_size), \
        f"Expected ({num_steps * batch_size}, {vocab_size}), got {Y.shape}"
    assert new_state[0].shape == (batch_size, num_hiddens), \
        f"Expected ({batch_size}, {num_hiddens}), got {new_state[0].shape}"    
    
def test_hidden_state_updates():
    vocab_size  = 10
    num_hiddens = 32
    batch_size  = 4
    num_steps   = 3

    model = RNNModel(vocab_size, num_hiddens, DEVICE)
    state = model.begin_state(batch_size, DEVICE)
    X     = torch.randint(0, vocab_size, (batch_size, num_steps))

    _, new_state = model(X, state)

    assert not torch.equal(state[0], new_state[0]), \
        "Hidden state should change after forward pass"
    
def test_perplexity_decreases():
    data_iter, vocab = load_data_time_machine(batch_size=32, num_steps=35)
    model    = RNNModel(len(vocab), 128, DEVICE)
    loss_fn  = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.params, lr=1.0)

    # Collect one epoch of batches
    data_iter, _ = load_data_time_machine(batch_size=32, num_steps=35)
    ppl_before   = train_epoch(model, data_iter, loss_fn, optimizer, DEVICE, True)

    data_iter, _ = load_data_time_machine(batch_size=32, num_steps=35)
    ppl_after    = train_epoch(model, data_iter, loss_fn, optimizer, DEVICE, True)

    assert ppl_after < ppl_before, \
        f"Perplexity should decrease: before={ppl_before:.1f}, after={ppl_after:.1f}"