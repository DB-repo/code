from db import get_db

RUNNING = 0
PAUSE = 1
STOP = 2


def pause(token):
    conn = get_db()

    pause_query = "UPDATE query_sequence SET status={} WHERE token='{}'".format(PAUSE, token)

    cursor = conn.cursor()
    cursor.execute(pause_query)
    rows = cursor.fetchall()

    cursor.close()
    conn.close()



