import re
import collections
import random
import torch

def load_time_machine():
    try:
        with open("data/timemachine.txt", "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        import urllib.request
        import os
        os.makedirs("data", exist_ok=True)
        url = "https://www.gutenberg.org/files/35/35-0.txt"
        urllib.request.urlretrieve(url, "data/timemachine.txt")
        with open("data/timemachine.txt", "r") as f:
            lines = f.readlines()

    return [re.sub("[^a-z]+", " ", line.lower()) for line in lines]

def tokenize(lines, token='char'):
    if token == 'char':
        return [list(line) for line in lines]
    elif token == 'word':
        return [line.split() for line in lines]
    
class Vocab:
    def __init__(self, tokens, min_freq=0):
        # Flatten list of lists into one list
        flat = [t for line in tokens for t in line]

        # Count frequency of each token
        counter = collections.Counter(flat)

        # Sort by frequency, most common first
        self.token_freqs = sorted(counter.items(), key=lambda x: -x[1])

        # Unknown token gets index 0
        self.idx_to_token = ['<unk>'] + [
            t for t, freq in self.token_freqs if freq >= min_freq
        ]
        self.token_to_idx = {t: i for i, t in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if isinstance(tokens, list):
            return [self.token_to_idx.get(t, 0) for t in tokens]
        return self.token_to_idx.get(tokens, 0)
    
def build_corpus(lines, vocab, token='char'):
    tokens = tokenize(lines, token)
    corpus = [idx for line in tokens for idx in vocab[line]]
    return corpus, vocab

def seq_data_iter_random(corpus, batch_size, num_steps):
    start = random.randint(0, num_steps - 1)
    corpus = corpus[start:]

    # How many complete sequences fit?
    num_seqs = (len(corpus) - 1) // num_steps

    # All possible starting positions
    initial_indices = list(range(0, num_seqs * num_steps, num_steps))
    random.shuffle(initial_indices)

    def get_seq(pos):
        return corpus[pos: pos + num_steps]
    
    # Yield batches
    num_batches = num_seqs // batch_size
    for i in range(0, num_batches * batch_size, batch_size):
        batch_indices = initial_indices[i: i + batch_size]
        X = [get_seq(j) for j in batch_indices]
        Y = [get_seq(j + 1) for j in batch_indices]
        yield torch.tensor(X), torch.tensor(Y)

def load_data_time_machine(batch_size, num_steps, token="char"):
    lines = load_time_machine()
    corpus, vocab = build_corpus(lines, Vocab(tokenize(lines, token)))
    data_iter = seq_data_iter_random(corpus, batch_size, num_steps)
    return data_iter, vocab