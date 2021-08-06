import sqlite3

import click
from flask import current_app
from flask.cli import with_appcontext

import psycopg2
import time

CONNECTION_STR = "dbname='test' user='postgres' host='localhost' password='postgres'"
# CONNECTION_STR = "dbname='test' user='postgres' host='localhost' password='postgres'"

def get_db():

    conn = None
    try:
        conn = psycopg2.connect(CONNECTION_STR)
    except Exception as e:
        print "Connection Failed %s" % e

    return conn


def close_db(conn):
    """If this request connected to the database, close the
    connection.
    """
    conn.close()


def get_txn_id(token):
    connection = get_db()
    p_query = "select txid FROM query_sequence where token='{}'".format(token)
    cursor = connection.cursor()
    cursor.execute(p_query)
    try:
        return cursor.fetchall()[0][0]
    except:
        return None
