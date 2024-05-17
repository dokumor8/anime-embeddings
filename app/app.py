import click
from flask import Flask, render_template, request
# import sqlite3


app = Flask(__name__)


data = [
    {"title": "K-On!", "en_title": "K-On!", "alt_title": "kon"}
]


@app.route('/')
def index():
    return render_template('search.html.j2', results=[])


@app.route('/', methods=['POST'])
def update_search():
    def hits(search):
        return filter(
            lambda x: any(
                (
                    search in x["title"].lower(),
                    search in x["en_title"].lower(),
                    search in x["alt_title"].lower(),
                )
            ),
            data,
        )

    query = request.form["search"].lower()
    results = hits(query)

    # search_string = request.args.get("search")
    # print(search_string)
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
