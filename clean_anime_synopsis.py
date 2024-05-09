import pandas as pd


def filter_text(original_text):
    text = str(original_text)
    text = text.replace("[Written by MAL Rewrite]", "")
    source_marker = "(Source:"
    source_id = text.find(source_marker)
    if source_id != -1:
        text = text[:source_id]
    text = text.strip()
    return text


def filter_animes():
    df = pd.read_csv("./archive/animes.csv")
    df = df[["uid", "synopsis"]]
    
    df["synopsis"] = df["synopsis"].apply(filter_text)
    df.to_csv("./data/animes_clean.csv", index=False)


filter_animes()
