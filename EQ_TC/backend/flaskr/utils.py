from db import *
from collections import defaultdict
from process import Query, Results, Plan
from datetime import datetime, timedelta


TABLES = ["query_stats", "query_accuracy", "query_plan", "query_benefit", "query_perf", "query_thresholds",
          "output_response_tbl", "output_proba_tbl", "wifi_prog", "images_prog", "tweets_prog", "query_epoch"]


def cleanup():
    conn = get_db()

    cursor = conn.cursor()

    # cursor.execute("update wifi_state set state_bitmap = '{1,0}';")
    # cursor.execute("update wifi_state set output[2:2][:] = '{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}';")
    # cursor.execute("update wifi set location = NULL;")

    cursor.execute("Select txid FROM query_sequence")

    ans = cursor.fetchall()

    for row in ans:

        cursor.execute("DROP VIEW IF EXISTS {}_{}".format("group_query_result", row[0]))
        cursor.execute("DROP MATERIALIZED VIEW IF EXISTS {}_{}".format("query_result", row[0]))
        cursor.execute("DROP VIEW IF EXISTS {}_{}".format("single_row_view", row[0]))
        for tbl in TABLES:
            print (row,"DROP TABLE IF EXISTS {}_{}".format(tbl, row[0]))
            cursor.execute("DROP TABLE IF EXISTS {}_{}".format(tbl, row[0]))

    cursor.execute("Delete From query_sequence")
    cursor.close()
    conn.commit()
    conn.close()

cleanup()