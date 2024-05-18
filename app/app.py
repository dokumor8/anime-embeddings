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


@app.route('/')
def index():
    return render_template('search.html.j2', results=[])


@app.route('/', methods=['POST'])
def update_search():
    conn = sqlite3.connect('data/anime_info.db')  # Connect to the SQLite database
    cursor = conn.cursor()  # Create a cursor

    search_string = request.form["search"]

    # Construct the SQL query for partial matches
    partial_query = """
    SELECT a.anime_id, a.title
    FROM anime a
    JOIN popular_anime p ON a.anime_id = p.anime_id
    WHERE a.title LIKE ? OR a.title_english LIKE ? OR a.title_japanese LIKE ? OR a.title_synonyms LIKE ?
    LIMIT 20
    """

    # partial_query = """
    # SELECT anime_id,title FROM anime
    # WHERE title LIKE ? OR title_english LIKE ? OR title_japanese LIKE ? OR title_synonyms LIKE ?
    # LIMIT 20
    # """
    cursor.execute(partial_query, (f"%{search_string}%", f"%{search_string}%", f"%{search_string}%", f"%{search_string}%"))
    partial_results = cursor.fetchall()  # Fetch partial matches

    results = partial_results[:20]

    return render_template('search.html.j2', results=results)


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
