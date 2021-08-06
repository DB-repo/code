from db import *
from collections import defaultdict
from process import Query, Results, Plan
from datetime import datetime, timedelta
from utils import cleanup

import sqlparse
import threading

NUM_SPLITS = 10
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
TIME_GROUP = 300


class WiFiApproximation(threading.Thread):

    def __init__(self, query, delay, epochs, group):
        threading.Thread.__init__(self)
        cleanup()
        self.query = query
        self.delay = delay
        self.epochs = epochs
        self.group = group

        self.threads = {}
        self.results = {}
        self.queries = {}
        self.split_query()

    def split_query(self):

        sel, from_, where, groupby = self.query.split("@")

        where = where.split(' ', 1)[1].split(" AND ")

        other_predicates = []
        start = end = None
        for predicate in where:
            predicate = predicate.strip()
            if "timestamp" in predicate:
                lhs, op, rhs = predicate.split(' ', 2)
                if op == ">":
                    start = datetime.strptime(rhs.strip("'"), DATE_FORMAT)
                else:
                    end = datetime.strptime(rhs.strip("'"), DATE_FORMAT)
            else:
                other_predicates.append(predicate)

        count = 1
        delta = (end - start)/NUM_SPLITS
        for i in range(NUM_SPLITS):

            end = start + delta
            predicate_str = "{} AND wifi.timestamp > ''{}'' AND wifi.timestamp < ''{}''".format(
                ' AND '.join(other_predicates), start.strftime(DATE_FORMAT), end.strftime(DATE_FORMAT)
            )

            query = "{select} @{from_} @WHERE {whr} @{groupby}".format(
                select=sel, from_=from_, whr=predicate_str, groupby=groupby
            )
            self.queries[str(count)] = query
            count += 1
            start = end

    def run(self):
        for token, query in self.queries.items():
            query = Query(query, self.delay, self.epochs, self.group, 1, token)
            self.threads[token] = query
            query.start()

            result =  Results(self.group, token)
            self.results[token] = result

    def reload(self):
        ans = []
        columns = []
        for token, result in self.results.items():
            ans.extend(result.fetch())
            columns = result.get_columns()

        if self.group == 1:
            ans = sorted(ans, key=lambda x: x[0][0])
        if self.group == 0:
            ans = sorted(ans, key=lambda x: x[1], reverse=True)

        # ans = map(lambda x: ((datetime.fromtimestamp(x[0][0]*TIME_GROUP).strftime(DATE_FORMAT),x[0][1]), x[1]), ans)
        return columns, ans

    def plan(self):

        txids = self.get_txids()
        epoch_query = "SELECT epoch FROM query_epoch_{}"
        conn = get_db()
        cursor = conn.cursor()

        epochs = {}
        for txid in txids:
            cursor.execute(epoch_query.format(str(txid)))
            epochs[str(txid)] = cursor.fetchall()[0][0]

        cursor.close()
        conn.close()
        return epochs

    def stop(self):
        for token, query in self.threads.items():
            if query.is_alive():
                query.close()
            del self.threads[token]

    def get_txids(self):
        txids = []
        for token in self.queries:
            txid = get_txn_id(token)
            if txid is not None:
                txids.append(get_txn_id(token))

        return txids


class ExplainWiFiApproximation:

    def __init__(self, query):
        self.query = query

    def explain(self):

        explain = "-- Number of Splits : {}\n\n".format(NUM_SPLITS)

        sel, from_, where, groupby = self.query.split("@")
        where = where.split(' ', 1)[1].split(" AND ")

        other_predicates = []
        start = end = None
        for predicate in where:
            predicate = predicate.strip()
            if "timestamp" in predicate:
                lhs, op, rhs = predicate.split(' ', 2)
                if op == ">":
                    start = datetime.strptime(rhs.strip("'"), DATE_FORMAT)
                else:
                    end = datetime.strptime(rhs.strip("'"), DATE_FORMAT)
            else:
                other_predicates.append(predicate)

        count = 1
        delta = (end - start) / NUM_SPLITS
        for i in range(NUM_SPLITS):
            end = start + delta
            predicate_str = "{} AND wifi.timestamp > ''{}'' AND wifi.timestamp < ''{}''".format(
                ' AND '.join(other_predicates), start.strftime(DATE_FORMAT), end.strftime(DATE_FORMAT)
            )

            query = "{select} @{from_} @WHERE {whr} @{groupby}".format(
                select=sel, from_=from_, whr=predicate_str, groupby=groupby
            )

            explain += "\n\n-- Token {}\n".format(count)
            explain += sqlparse.format(query, reindent=True, keyword_case='upper')
            count += 1
            start = end

        return explain
