import pickle
import numpy as np

from FlagEmbedding import FlagModel


def main():
    with open("saved_emb.pkl", "rb") as f:
        embs = pickle.load(f)
    print(embs.keys())
    for k in embs:
        average = embs[k]
        average = average / np.linalg.norm(average)
        embs[k] = average
    print(embs[34096].shape)
    embs_stack = np.stack(list(embs.values())).T
    print(embs_stack.shape)
    similarity = embs_stack.T @ embs_stack
    print(similarity)


main()
