import pandas as pd
import json


# def main():
#     df = pd.read_csv("./data/reviews_100.csv")
#     df_subset = df.head(50)
#     df_subset.to_csv("./data/reviews_100_2.csv", index=False)


# main()

def prepare_id_name_list():
    df = pd.read_csv("./data/animes_100.csv")
    id_name = df[["uid", "title"]]
    id_name_list = id_name.to_dict("list")
    uid_list = id_name_list["uid"]
    name_list = id_name_list["title"]
    id_to_name_dict = {}
    name_to_id_dict = {}
    for i, n in zip(uid_list, name_list):
        id_to_name_dict[i] = n
        name_to_id_dict[n] = i

    with open("id_to_name.json", "w") as f:
        json.dump(id_to_name_dict, f)
    with open("name_to_id.json", "w") as f:
        json.dump(name_to_id_dict, f)


prepare_id_name_list()
