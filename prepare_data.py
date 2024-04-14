import pandas as pd
import json


# def cut_reviews():
#     df = pd.read_csv("./archive/reviews.csv")
#     print(len(df.index))
#     # df = pd.read_csv("./data/reviews_100.csv")
#     df_subset = df.head(10000)
#     df_subset.to_csv("./data/reviews_10k.csv", index=False)


def filter_text(original_text):
    enj_idx = original_text.find("Enjoyment")
    enj_idx = original_text.find("\n", enj_idx + 1)
    enj_idx = original_text.find("\n", enj_idx + 1)
    text = original_text[enj_idx + 1:]
    text = text.strip()
    # remove "helpful"
    text = text[:-8]
    text = text.strip()
    return text


def filter_reviews():
    df = pd.read_csv("./data/reviews_10k.csv")
    df = df[["anime_uid", "text"]]
    
    df["text"] = df["text"].apply(filter_text)
    df.to_csv("./data/review_clean.csv", index=False)


# main()

def prepare_id_name_list():
    df = pd.read_csv("./archive/animes.csv")
    id_name = df[["uid", "title"]]
    id_name_list = id_name.to_dict("list")
    uid_list = id_name_list["uid"]
    print(type(uid_list[0]))
    name_list = id_name_list["title"]
    id_to_name_dict = {}
    name_to_id_dict = {}
    for i, n in zip(uid_list, name_list):
        id_to_name_dict[i] = n
        name_to_id_dict[n] = i

    # with open("id_to_name.json", "w") as f:
    #     json.dump(id_to_name_dict, f)
    # with open("name_to_id.json", "w") as f:
    #     json.dump(name_to_id_dict, f)


# prepare_id_name_list()
# cut_reviews()
filter_reviews()
