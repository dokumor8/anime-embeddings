import sqlite3

def main():
    con = sqlite3.connect("tutorial.db")
    cur = con.cursor()
    cur.execute("CREATE TABLE anime(title, year, en_title, alt_title)")


main()
