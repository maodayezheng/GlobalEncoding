import numpy as np


def indices_to_sentence(indices, vocab_dir):
    vocab = []
    with open(vocab_dir, "r") as v:
        for line in v:
            vocab.append(line.rstrip("\n"))
