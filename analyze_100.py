import pandas as pd
import pickle
from FlagEmbedding import FlagModel
import numpy as np

# uid,profile,anime_uid,text,score,scores,link
# def split_text(model, text):
#     pass


def get_embeddings(model, text):
    sentences = text
    embeddings = model.encode(sentences, convert_to_numpy=True)
    return embeddings


def main():
    df = pd.read_csv("./data/reviews_100_2.csv")
    model = FlagModel("./model", use_fp16=True)
    all_embeddings = {}

    # TODO add groupby by animes
    # TODO analyze average length of 512 tokens in symbols and split that
    df_grouped = df.groupby("anime_uid")
    for anime_id, data in df_grouped:
        for index, row in data.iterrows():
            ani_id = row["anime_uid"]
            text = row["text"]
            embeddings = get_embeddings(model, text)
            # tokens = model.tokenizer.tokenize(text)
            # print(f"Index: {index}, id: {ani_id}, tokens: {len(tokens)}")
            # print(tokens)
            emb_list = all_embeddings.get(ani_id, [])
            emb_list.append(embeddings)
            all_embeddings[ani_id] = emb_list

    print(all_embeddings)
    for k, v in all_embeddings.items():
        average = np.mean(v, axis=0)
        all_embeddings[k] = average
        # average = np.zeros_like(v[0])
        # for emb in v:
    print(all_embeddings)
    with open("saved_emb.pkl", "wb") as f:
        pickle.dump(all_embeddings, f)


main()
