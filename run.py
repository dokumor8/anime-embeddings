import json
import numpy as np


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def get_scores(ani_id):
    embs = np.load("data/all_embs.npy")
    # idx to aniid
    ids = load_json("data/ids.json")
    aniid_to_index = {x: i for i, x in enumerate(ids)}
    emb_index = aniid_to_index[ani_id]
    query_emb = embs[emb_index, :]
    similarities = query_emb @ embs.T

    sim_list = list(zip(similarities, ids))
    sim_list.sort(reverse=True)
    # similarity_dict = {}
    # for aniid, sim in zip(ids, similarities):
        # similarity_dict[aniid] = sim
    return sim_list


# def main():
#     with open("saved_emb.pkl", "rb") as f:
#         embs = pickle.load(f)
#     print(embs.keys())
#     for k in embs:
#         average = embs[k]
#         average = average / np.linalg.norm(average)
#         embs[k] = average
#     print(embs[34096].shape)
#     embs_stack = np.stack(list(embs.values())).T
#     print(embs_stack.shape)
#     similarity = embs_stack.T @ embs_stack
#     print(similarity)


def run():
    # Input - anime rough name
    # Output - names of similar anime

    # 0. Anime rough name -> anime precise name
    # 1. Anime precise name -> Anime ID
    # 2. Anime ID -> embedding (lookup)
    # 3. Embedding -> similarities (compute, later worry about caching)
    # 4. Similarities -> top X IDs
    # 5. IDs -> names
    name_to_id = load_json("name_to_id.json")
    id_to_name = load_json("id_to_name.json")

    query_name = "Saint Seiya: Meiou Hades Meikai-hen"
    query_id = name_to_id[query_name]
    print(query_id)

    sim_dict = get_scores(query_id)
    print(sim_dict)
    for sim, aniid in sim_dict:
        name = id_to_name.get(aniid, "No name")

        print({sim}, name)


run()
