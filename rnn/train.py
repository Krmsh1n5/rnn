import torch
import torch.nn as nn
import math
from rnn.model import RNNModel
from rnn.data import load_data_time_machine


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

def train(model, batch_size, num_steps, lr, num_epochs, device):    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.params, lr=lr)
    data_iter, _ = load_data_time_machine(batch_size, num_steps)

    for epoch in range(num_epochs):
        data_iter, _ = load_data_time_machine(batch_size, num_steps)
        ppl = train_epoch(model, data_iter, loss_fn, optimizer, device, use_random_iter=True)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Perplexity {ppl:.1f}")

if __name__ == "__main__":
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, vocab   = load_data_time_machine(batch_size=32, num_steps=35)

    model = RNNModel(
        vocab_size  = len(vocab),
        num_hiddens = 512,
        device      = device
    )

    train(model, batch_size=32, num_steps=35, lr=1.0, num_epochs=50, device=device)