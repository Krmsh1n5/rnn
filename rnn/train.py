import torch
import torch.nn as nn
import math
from rnn.model import RNNModel
from rnn.data import load_data_time_machine, load_data_shakespeare_word


def grad_clipping(params, theta):
    norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params if p.grad is not None))
    if norm > theta:
        for p in params:
            if p.grad is not None:
                p.grad.mul_(theta / norm)

def train_epoch(model, data_iter, loss_fn, optimizer, device, use_random_iter):
    state = None
    total_loss, total_tokens = 0.0, 0

    for X, Y in data_iter:
        X, Y = X.to(device), Y.to(device)

        # Reset state at start or for random batches
        if state is None or use_random_iter:
            state = model.begin_state(X.shape[0], device)
        else:
            # Detach state to prevent backprop through entire history
            state = tuple(s.detach() for s in state)
        
        y_hat, state = model(X, state)
        y = Y.T.reshape(-1)

        loss = loss_fn(y_hat, y).mean()
        optimizer.zero_grad()
        loss.backward()
        grad_clipping(model.params, 1)
        optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    
    return math.exp(total_loss / total_tokens)  # perplexity

def predict(model, prefix, num_predict, vocab, device):
    state = model.begin_state(batch_size=1, device=device)

    # Convert prefix words to indices
    output = [vocab[w] for w in prefix.split()]

    # Warm up the hidden state by feeding the prefix
    for idx in output[:-1]:
        X = torch.tensor([[idx]], device=device)
        _, state = model(X, state)

    for _ in range(num_predict):
        X = torch.tensor([[output[-1]]], device=device)
        y_hat, state = model(X, state)
        next_idx = int(y_hat.argmax(dim=1).item())
        output.append(next_idx)

    return " ".join([vocab.idx_to_token[i] for i in output])

def train(model, batch_size, num_steps, lr, num_epochs, device, dataset="time_machine"):
    loaders = {
        "time_machine": load_data_time_machine,
        "shakespeare": load_data_shakespeare_word
    }    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.params, lr=lr)
    data_iter, vocab = loaders[dataset](batch_size, num_steps)
    for epoch in range(num_epochs):
        data_iter, _ = load_data_time_machine(batch_size, num_steps)
        ppl = train_epoch(model, data_iter, loss_fn, optimizer, device, use_random_iter=True)
        print(f"Epoch {epoch+1:3d} | Perplexity {ppl:.1f}")

    # After training, show some predictions
    test_prefixes = ['to be or not', 'the king is', 'love is']
    for prefix in test_prefixes:
        result = predict(model, prefix, num_predict=10, vocab=vocab, device=device)
        print(f"\nSeed: '{prefix}'")
        print(f"Generated: {result}")

if __name__ == "__main__":
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, vocab = load_data_shakespeare_word(batch_size=32, num_steps=35)

    model = RNNModel(
        vocab_size  = len(vocab),
        num_hiddens = 512,
        device      = device
    )

    train(model, batch_size=32, num_steps=35, lr=1.0,
          num_epochs=50, device=device, dataset='shakespeare')