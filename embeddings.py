import numpy as np
import tensorflow as tf


def load_glove(filename, max_vocab, embedding_size):
    embeddings = np.zeros((max_vocab + 2, embedding_size), dtype=np.float32)
    token2id = {}
    id2token = {}

    token_id = len(token2id)
    token2id["#pad#"] = token_id
    id2token[token_id] = "#pad#"
    embeddings[token_id] = np.zeros(embedding_size, dtype=np.float32)

    token_id = len(token2id)
    token2id["#unk#"] = token_id
    id2token[token_id] = "#unk#"
    embeddings[token_id] = np.zeros(embedding_size, dtype=np.float32)

    with open(filename) as in_file:
        for line_index, line in enumerate(in_file):
            values = line.rstrip().split(" ")
            word = values[0]
            embedding = np.array(values[1:], dtype=np.float32)
            token_id = len(token2id)
            token2id[word] = token_id
            id2token[token_id] = word
            embeddings[token_id] = embedding

            if token_id == max_vocab + 1:
                break

    return embeddings, token2id, id2token


def glove_embeddings_initializer(embeddings):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        return embeddings

    return _initializer
