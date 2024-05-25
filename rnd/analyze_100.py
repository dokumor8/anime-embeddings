import pandas as pd
import pickle
from FlagEmbedding import FlagModel
import numpy as np
import json
import os

# uid,profile,anime_uid,text,score,scores,link
# def split_text(model, text):
#     pass


def get_embeddings(model, text):
    sentences = text
    embeddings = model.encode(sentences, convert_to_numpy=True, batch_size=64)
    return embeddings


def get_embeddings_of_file(model, path):
    df = pd.read_csv(path)

    texts = df["text"].to_list()
    texts = texts[:32]
    embeddings = model.encode(texts, convert_to_numpy=True, batch_size=32)
    average = np.mean(embeddings, axis=0)
    average = average / np.linalg.norm(average)
    return average


def main():
    df = pd.read_csv("./data/review_clean.csv")
    model = FlagModel("./model", use_fp16=True)

    ids = []
    all_embeddings = []
    df_grouped = df.groupby("anime_uid")
    with open("data/ids.log", "w") as f:
        for anime_id, data in df_grouped:
            # if anime_id < 5680:
            #     continue
            ids.append(anime_id)
            print(anime_id, file=f)
            print(anime_id)

            texts = data["text"].to_list()
            embeddings = model.encode(texts, convert_to_numpy=True, batch_size=64)
            average = np.mean(embeddings, axis=0)
            average = average / np.linalg.norm(average)
            all_embeddings.append(average)

            # np.save(f"data/embs/emb_{anime_id}.npy", average)
        # for index, row in data.iterrows():
        #     # ani_id = row["anime_uid"]
        #     text = row["text"]
        #     embeddings = get_embeddings(model, text)
        #     # tokens = model.tokenizer.tokenize(text)
        #     # print(f"Index: {index}, id: {ani_id}, tokens: {len(tokens)}")
        #     # print(tokens)
        #     emb_list = all_embeddings.get(ani_id, [])
        #     emb_list.append(embeddings)
        #     all_embeddings[ani_id] = emb_list

    # print(type(all_embeddings))
    # print(len(all_embeddings))
    # for x in all_embeddings:
    #     print(x.shape)
    # print(len(all_embeddings[0]))
    # print(all_embeddings[0].shape)

    all_embeddings = np.array(all_embeddings)
    print(all_embeddings.shape)

    np.save("data/all_embs.npy", all_embeddings)
    with open("data/ids.json", "w") as f:
        json.dump(ids, f)
    # for k, v in all_embeddings.items():
    #     average = np.mean(v, axis=0)
    #     all_embeddings[k] = average
    #     # average = np.zeros_like(v[0])
    #     # for emb in v:
    # print(all_embeddings)
    # with open("saved_emb.pkl", "wb") as f:
    #     pickle.dump(all_embeddings, f)


def main2():
    model = FlagModel("./model", use_fp16=True)
    ids = []
    embs = []
    for root, subdirs, files in os.walk("data/split"):
        for name in files:
            ani_id = name[8:]
            ani_id = ani_id[:-4]
            dir_num = os.path.basename(root)
            emb_path = f"data/separa_embs/{dir_num}/emb_{ani_id}.npy"
            if os.path.isfile(emb_path):
                print(f"Skipped {ani_id}")
                continue
            full_path = os.path.join(root, name)
            emb = get_embeddings_of_file(model, full_path)
            embs.append(emb)
            ids.append(ani_id)
            print(ani_id)

            np.save(emb_path, emb)

    embs = np.array(embs)
    print(embs.shape)
    np.save("data/all_embs_17.npy", embs)
    with open("data/ids_17.json", "w") as f:
        json.dump(ids, f)


main2()
