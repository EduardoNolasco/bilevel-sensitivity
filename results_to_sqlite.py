"""
Build a small SQLite database from the reference results, and optionally run a .sql file against it.

    python results_to_sqlite.py                 # builds results.db
    python results_to_sqlite.py queries.sql     # builds results.db, then runs every query in queries.sql

Tables
    runs      one row per run (from the reference CSV)
    problems  one row per benchmark problem (dimensions, known objective value, linear lower level or not)
    table1    the 13 rows reported in Table 1 of the paper (typed in by hand)
Only the Python standard library is used.
"""
import csv
import os
import re
import sqlite3
import sys

CSV_PATH = os.path.join('reference_results', 'rerun_2026-10-09.csv')
DB_PATH = 'results.db'

# Problems flagged is_ll_linear in bolib_class.py (epsilon-regularised lower level)
LL_LINEAR = {'Bard_1991_ex1', 'Bard_Book_1998', 'DempeEtal2012', 'DempeFranke2011Ex41',
             'Dempe_Franke_2011_ex42', 'Dempe_Lohse_2011_ex31a', 'Dempe_Lohse_2011_ex31b'}

# Table 1 of the paper: (problem, x0, reported F, iterations, time in seconds)
TABLE1 = [
    ('AiyoshiShimizu1984Ex2',    '20,20',    5.0,   12, 2.20),
    ('AllendeStill2013',         '2,2',     -1.0,   11, 1.11),
    ('Bard_1988_ex1',            '2',       17.0,   11, 0.91),
    ('Bard_1991_ex1',            '4',        2.0,   11, 0.48),
    ('Bard_Book_1998',           '15,15',    0.0,    6, 0.28),
    ('ClarkWesterberg1990',      '1.7',      5.0,   11, 0.33),
    ('DempeEtal2012',            '0.9',     -1.0,   11, 0.22),
    ('Dempe_Franke_2011_ex42',   '-0.9,0.9', 3.0,   11, 2.75),
    ('Dempe_Lohse_2011_ex31a',   '-0.4,-0.4', -5.5, 11, 5.35),
    ('Dempe_Lohse_2011_ex31b',   '4,4,4',  -12.0,   12, 1.13),
    ('FloudasEtal2013',          '10,10',    0.0,   11, 0.48),
    ('Outrata_Cervinka_2009',    '-10,-1',   0.0,   12, 0.67),
    ('Shimizu_Aiyoshi_1981_ex2', '10,1',   225.0,   39, 4.20),
]

SCHEMA = """
CREATE TABLE problems (
    name      TEXT PRIMARY KEY,
    n         INTEGER,
    m         INTEGER,
    f_known   REAL,
    ll_linear INTEGER
);
CREATE TABLE runs (
    problem    TEXT REFERENCES problems(name),
    x0         TEXT,
    f_found    REAL,
    status     TEXT,
    iterations INTEGER,
    time_s     REAL,
    PRIMARY KEY (problem, x0)
);
CREATE TABLE table1 (
    problem             TEXT PRIMARY KEY REFERENCES problems(name),
    x0                  TEXT,
    f_reported          REAL,
    iterations_reported INTEGER,
    time_reported       REAL
);
"""


def clean_x0(text):
    """'[ 0.1 -0.9]' -> '0.1,-0.9' ; '[20. 20.]' -> '20,20'."""
    numbers = re.findall(r'-?\d+\.?\d*(?:e-?\d+)?', text)
    return ','.join('%g' % float(v) for v in numbers)


def build(db_path):
    if os.path.exists(db_path):
        os.remove(db_path)
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    problems = {}
    with open(CSV_PATH, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            name = row['Problem_Name']
            n, m = (int(v) for v in re.findall(r'\d+', row['Dimensions (n,m)']))
            problems[name] = (name, n, m, float(row['F_known']), int(name in LL_LINEAR))
            con.execute('INSERT INTO runs VALUES (?,?,?,?,?,?)',
                        (name, clean_x0(row['x0']), float(row['F_found']), row['Status'],
                         int(row['Iterations']), float(row['Time (s)'])))
    con.executemany('INSERT INTO problems VALUES (?,?,?,?,?)', problems.values())
    con.executemany('INSERT INTO table1 VALUES (?,?,?,?,?)', TABLE1)
    con.commit()
    return con


def run_file(con, path):
    """Run each statement of a .sql file and print the result as a plain-text table."""
    buffer = ''
    with open(path, encoding='utf-8') as f:
        for line in f:
            buffer += line
            if sqlite3.complete_statement(buffer):
                cur = con.execute(buffer)
                print('\n' + buffer.strip())
                if cur.description:
                    header = [d[0] for d in cur.description]
                    rows = [tuple('' if v is None else v for v in r) for r in cur.fetchall()]
                    widths = [max(len(str(x)) for x in [h] + [r[i] for r in rows]) for i, h in enumerate(header)]
                    print('  '.join(h.ljust(w) for h, w in zip(header, widths)))
                    for r in rows:
                        print('  '.join(str(v).ljust(w) for v, w in zip(r, widths)))
                    print('(%d rows)' % len(rows))
                buffer = ''


if __name__ == '__main__':
    connection = build(DB_PATH)
    for table in ('problems', 'runs', 'table1'):
        print(table, connection.execute('SELECT COUNT(*) FROM ' + table).fetchone()[0], 'rows')
    if len(sys.argv) > 1:
        run_file(connection, sys.argv[1])