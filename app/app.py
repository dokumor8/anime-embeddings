import click
from flask import Flask, render_template, request, session
import sqlite3
import numpy as np
import json


app = Flask(__name__)
app.secret_key = "your_secret_key"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_ctx():
    ctx = {}
    embs = np.load("ignored/embs.npy")
    embs_synopsis = np.load("ignored/syn_embs.npy")
    rid_to_aid = load_json("ignored/idx_to_ani2.json")
    aid_to_rid = load_json("ignored/ani_to_idx2.json")
    ctx["review_embeddings"] = embs
    ctx["synopsis_embeddings"] = embs_synopsis
    ctx["rid_to_aid"] = rid_to_aid
    ctx["aid_to_rid"] = aid_to_rid
    return ctx


ctx = load_ctx()


def get_top_scores(ctx, ani_ids, synopsis_weight):

    emb_matrix = ctx["review_embeddings"]
    if synopsis_weight > 0:
        if synopsis_weight > 1:
            synopsis_weight = 1
        emb_matrix = (1 - synopsis_weight) * ctx["review_embeddings"] + synopsis_weight * ctx["synopsis_embeddings"]

    query_embs = np.zeros((len(ani_ids), ctx["review_embeddings"].shape[1]))
    for i, ani_id in enumerate(ani_ids):
        rid = ctx["aid_to_rid"][str(ani_id)]
        query_emb = emb_matrix[rid, :]
        query_embs[i, :] = query_emb

    average_emb = np.mean(query_embs, axis=0)
    average_emb = average_emb / np.linalg.norm(average_emb)
    similarities = average_emb @ emb_matrix.T

    sim_list = list(zip(similarities, ctx["rid_to_aid"].values()))
    sim_list.sort(reverse=True)
    sim_list = sim_list[:20]

    return sim_list


@app.route("/")
def index():
    return render_template("search.html.j2", results=[])


@app.route("/get_similar/<int:anime_id>", methods=["POST", "GET"])
def get_similar(anime_id):
    similar_ids = get_top_scores(ctx, [anime_id], synopsis_weight=0.1)

    conn = sqlite3.connect("data/anime_info.db")  # Connect to the SQLite database
    cursor = conn.cursor()  # Create a cursor
    # similar_ids = get_similar_ids(anime_id)
    # Fetch the titles of the anime corresponding to the similar_ids
    similar_results = []
    for sim, aniid in similar_ids:
        cursor.execute(
            "SELECT anime_id, title, url, main_picture FROM anime WHERE anime_id = ?",
            (aniid,),
        )
        result = cursor.fetchone()
        if result:
            # result.append(sim)
            sim_result = (*result, sim)
            similar_results.append(sim_result)

    return render_template("search.html.j2", results=similar_results, context="similar")


@app.route("/r")
def update_search():

    print(session)
    # update input ids with new query
    if "ids" in session:
        ids = session["ids"]
    else:
        ids = []

    search_string = ""
    if request.args.get("search"):
        search_string = request.args.get("search")
        session["text_search"] = search_string
    else:
        if "text_search" in session:
            search_string = session["text_search"]
        else:
            search_string = None

    is_sem_search = request.args.get("start")

    if request.args.get("i"):
        new_id = request.args.get("i")
        if new_id not in ids:
            ids.append(new_id)
            session["ids"] = ids

    if request.args.get("remove"):
        removing_id = request.args.get("remove")
        if removing_id in ids:
            ids.remove(removing_id)
            session["ids"] = ids

    selected_ids = ids.copy()

    if request.args.get("s"):
        is_sem_search = True
        selected_ids = [request.args.get("s")]

    conn = sqlite3.connect("data/anime_info.db")  # Connect to the SQLite database
    cursor = conn.cursor()  # Create a cursor

    text_search_results = []
    if search_string and not is_sem_search:
        partial_query = """
        SELECT anime_id, title, url, main_picture
        FROM anime
        WHERE anime_id IN (SELECT anime_id FROM popular_anime)
        AND (title LIKE ? OR title_english LIKE ? OR title_japanese LIKE ? OR title_synonyms LIKE ?)
        LIMIT 10
        """

        cursor.execute(
            partial_query,
            (
                f"%{search_string}%",
                f"%{search_string}%",
                f"%{search_string}%",
                f"%{search_string}%",
            ),
        )
        text_search_results = cursor.fetchall()  # Fetch partial matches
    results = text_search_results

    if is_sem_search:
        similar_ids = get_top_scores(ctx, selected_ids, synopsis_weight=0.1)

        conn = sqlite3.connect("data/anime_info.db")  # Connect to the SQLite database
        cursor = conn.cursor()  # Create a cursor
        similar_results = []
        for sim, aniid in similar_ids:
            cursor.execute(
                "SELECT anime_id, title, url, main_picture FROM anime WHERE anime_id = ?",
                (aniid,),
            )
            result = cursor.fetchone()
            if result:
                # result.append(sim)
                sim_result = (*result, sim)
                similar_results.append(sim_result)
        results = similar_results

    input_ids = []
    for input_id in ids:
        cursor.execute("SELECT anime_id, title, url FROM anime WHERE anime_id = ?", (input_id,))
        result = cursor.fetchone()
        if result:
            input_ids.append(result)
    return render_template("search.html.j2", results=results, input_ids=input_ids, search_string=search_string)


@app.route("/")
def home():
    return render_template("search.html.j2")


@click.command()
@click.option("--debug", is_flag=True, default=False, help="enable auto reload and debugging")
def main(debug: bool):
    app.run(debug=debug)


if __name__ == "__main__":
    main()
