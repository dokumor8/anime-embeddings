import click
from flask import Flask, render_template, request
import sqlite3
import numpy as np
import json


app = Flask(__name__)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_ctx():
    ctx = {}
    embs = np.load("ignored/embs.npy")
    embs_synopsis = np.load("ignored/syn_embs.npy")
    rid_to_aid = load_json("ignored/idx_to_ani2.json")
    aid_to_rid = load_json("ignored/ani_to_idx2.json")
    name_to_aid = load_json("ignored/name_to_id.json")
    aid_to_name = load_json("ignored/id_to_name.json")
    ctx["review_embeddings"] = embs
    ctx["synopsis_embeddings"] = embs_synopsis
    ctx["rid_to_aid"] = rid_to_aid
    ctx["aid_to_rid"] = aid_to_rid
    ctx["name_to_aid"] = name_to_aid
    ctx["aid_to_name"] = aid_to_name
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


@app.route('/')
def index():
    return render_template('search.html.j2', results=[])


@app.route('/get_similar/<int:anime_id>')
def get_similar(anime_id):
    similar_ids = get_top_scores(ctx, [anime_id], synopsis_weight=0.1)

    conn = sqlite3.connect('data/anime_info.db')  # Connect to the SQLite database
    cursor = conn.cursor()  # Create a cursor
    # similar_ids = get_similar_ids(anime_id)
    # Fetch the titles of the anime corresponding to the similar_ids
    similar_results = []
    for sim, aniid in similar_ids:
        cursor.execute("SELECT anime_id, title FROM anime WHERE anime_id = ?", (aniid,))
        result = cursor.fetchone()
        if result:
            # result.append(sim)
            sim_result = (*result, sim)
            similar_results.append(sim_result)

    return render_template('search.html.j2', results=similar_results, context='similar')


@app.route('/', methods=['POST', 'GET'])
def update_search():
    conn = sqlite3.connect('data/anime_info.db')  # Connect to the SQLite database
    cursor = conn.cursor()  # Create a cursor

    search_string = request.form["search"]

    # Construct the SQL query for partial matches

    partial_query = """
    SELECT anime_id, title
    FROM anime
    WHERE anime_id IN (SELECT anime_id FROM popular_anime)
    AND (title LIKE ? OR title_english LIKE ? OR title_japanese LIKE ? OR title_synonyms LIKE ?)
    LIMIT 50
    """

    cursor.execute(partial_query, (f"%{search_string}%", f"%{search_string}%", f"%{search_string}%", f"%{search_string}%"))
    partial_results = cursor.fetchall()  # Fetch partial matches

    results = partial_results

    return render_template('search.html.j2', results=results, context='search')


@click.command()
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="enable auto reload and debugging"
)
def main(debug: bool):
    app.run(debug=debug)



if __name__ == '__main__':
    main()
