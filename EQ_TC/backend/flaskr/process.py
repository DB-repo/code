""" Module responsible for running queries and fetching data
    from the database
"""
from db import *
from collections import defaultdict

import threading


class Baseline1Query():

    def __init__(self, query):
        self.connection = get_db()
        self.query = query
        self.connection.autocommit = True
        self.exception = None

    def run(self):
        p_query = "select madlib.all_functions_rewrite_query('{query}')".format(query=self.query)
        cursor = self.connection.cursor()
        cursor.execute(p_query)
        r_query = cursor.fetchall()[0][0]

        cursor.execute(r_query)
        cols = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()
        return (cols, data)

    def close(self):
        self.connection.close()


class ProgressiveBaselinesQuery(threading.Thread):

    def __init__(self, query, delay, epochs, group, id, baseline, token):
        threading.Thread.__init__(self)
        self.connection = get_db()
        self.query = query
        self.delay = delay
        self.epochs = epochs
        self.group = group
        self.id = id
        self.baseline = baseline
        self.connection.autocommit = True
        self.exception = None
        self.token = token

    def run(self):
        if self.group == 0:
            p_query = "call madlib.progressive_baselines_exec_driver_udf('{query}', {epochs}, {delay}, {baseline}, '{token}')".format(
                query=self.query, delay=self.delay, epochs=self.epochs, baseline=self.baseline, token=self.token)
        else:
            p_query = "call madlib.progressive_baselines_groupby_exec_driver_udf('{query}', {epochs}, {delay}, {baseline}, '{token}')".format(
                query=self.query, delay=self.delay, epochs=self.epochs, baseline=self.baseline, token=self.token)

        cursor = self.connection.cursor()
        try:
            cursor.execute(p_query)
        except Exception as e:
            self.exception = e
            print(e)
        finally:
            cursor.close()
            self.connection.close()

    def close(self):
        conn = get_db()
        c_query = "UPDATE query_sequence SET status = 2 WHERE token='{}'".format(self.token)
        cursor = conn.cursor()
        cursor.execute(c_query)
        conn.close()
        self.connection.close()


class Query(threading.Thread):

    def __init__(self, query, delay, epochs, group, id, token):
        threading.Thread.__init__(self)
        self.connection = get_db()
        self.query = query
        self.delay = delay
        self.epochs = epochs
        self.group = group
        self.id = id
        self.token = token
        self.connection.autocommit = True
        self.exception = None

    def run(self):
        if self.group == 0:
            p_query = "call madlib.progressive_exec_driver_udf('{query}', {epochs}, {delay}, '{token}')".format(
                query=self.query, delay=self.delay, epochs=self.epochs, token=self.token)
        else:
            p_query = "call madlib.progressive_groupby_exec_driver_udf('{query}', {epochs}, {delay}, '{token}')".format(
                query=self.query, delay=self.delay, epochs=self.epochs, token=self.token)

        cursor = self.connection.cursor()
        try:
            cursor.execute(p_query)
        except Exception as e:
            self.exception = e
            print(e)
        finally:
            cursor.close()
            self.connection.close()

    def close(self):
        conn = get_db()
        c_query = "UPDATE query_sequence SET status = 2 WHERE token='{}'".format(self.token)
        cursor = conn.cursor()
        cursor.execute(c_query)
        conn.close()
        self.connection.close()


class Results:

    def __init__(self, group, token):
        self.group = group
        self.old = set()
        self.epoch = 0
        self.token = token

    def fetch(self):
        self.txid = get_txn_id(self.token)

        self.connection = get_db()

        if self.group == 0:
            f_query = "SELECT * FROM query_result_{}".format(self.txid)
        else:
            f_query = "SELECT * FROM group_query_result_{}".format(self.txid)

        cursor = self.connection.cursor()
        cursor.execute(f_query)

        rows = cursor.fetchall()

        ans = []
        for row in rows:
            if row not in self.old:
                ans.append((row, 1))
            else:
                ans.append((row, 0))

        if self.group == 0:
            for row in self.old:
                if row not in rows:
                    ans.append((row, 2))

        if self.group == 1:
            ans = sorted(ans, key=lambda x: x[0][0])
        if self.group == 0:
            ans = sorted(ans, key=lambda x: x[1], reverse=True)

        self.old = set(rows)
        cursor.close()
        self.connection.close()
        return ans

    def get_columns(self):
        self.txid = get_txn_id(self.token)
        self.connection = get_db()

        if self.group == 0:
            c_query = "SELECT * FROM query_result_{}  LIMIT 0".format(self.txid)
        else:
            c_query = "SELECT * FROM group_query_result_{}  LIMIT 0".format(self.txid)

        cursor = self.connection.cursor()
        cursor.execute(c_query)

        columns = [desc[0] for desc in cursor.description]
        cursor.fetchall()

        cursor.close()
        self.connection.close()
        return columns


class Plan:

    def __init__(self, token):
        self.thresholds = []
        self.last_epoch_plan = []
        self.db_epoch = 0
        self.acc_result = defaultdict(list)
        self.token = token

    def fetch_epoch_plan(self):
        self.txid = get_txn_id(self.token)
        self.connection = get_db()

        epoch_query = "SELECT epoch FROM query_epoch_{}".format(self.txid)
        th_query = "SELECT tbl_name, attr_name, threshold FROM query_thresholds_{}".format(self.txid)
        pl_query = "SELECT tbl_name, attr_name, fid, count(*) as count_ FROM query_plan_{} GROUP BY tbl_name, attr_name, fid".format(self.txid)

        cursor = self.connection.cursor()
        cursor.execute(epoch_query)

        rows = cursor.fetchall()
        cur_epoch = rows[0][0]

        if cur_epoch == self.db_epoch:
            cursor.close()
            self.connection.close()
            return (self.db_epoch, self.thresholds, self.last_epoch_plan, self.acc_result)

        self.db_epoch = cur_epoch

        cursor.execute(th_query)
        rows = cursor.fetchall()
        self.thresholds = rows

        cursor.execute(pl_query)
        rows = cursor.fetchall()
        self.last_epoch_plan = rows
        cursor.close()
        self.connection.close()

        self.acc_result = self.get_accuracy()

        return (self.db_epoch, self.thresholds, self.last_epoch_plan, self.acc_result)

    def get_accuracy(self):
        self.txid = get_txn_id(self.token)
        self.connection = get_db()

        epoch_query = "SELECT tbl,epoch, prec,recall,2*prec*recall/(prec+recall) as f1 FROM query_accuracy_{}".format(self.txid)

        cursor = self.connection.cursor()
        cursor.execute(epoch_query)
        rows = cursor.fetchall()

        acc_result = defaultdict(list)
        for row in rows:
            acc_result[row[0]].append(row[1:])

        cursor.close()
        self.connection.close()
        return acc_result


class Explain:

    def __init__(self, query, algo):
        self.query = query
        self.algo = algo
        self.explain = ""
        self.alog_map = {
            "pro1": 0,
            "ba1": 1,
            "ba2": 2,
            "ba3": 3,
        }

    def fetch_explain(self):

        self.connection = get_db()

        ex_query = "select madlib.explain('{}', {})".format(self.query, self.alog_map[self.algo])
        cursor = self.connection.cursor()
        cursor.execute(ex_query)

        rows = cursor.fetchall()
        self.explain = rows[0][0]

        cursor.close()
        self.connection.close()

        return self.explain


class State:

    def __init__(self, type_, id):
        self.tbl = type_
        self.st_tbl = type_ + "_state"
        self.id = id

    def fetch_state(self):

        self.connection = get_db()

        st_query = "SELECT attribute, state_bitmap, output FROM {} WHERE id={}".format(self.st_tbl, self.id)
        f_query = """SELECT fc.name FROM progressive_functions f, progressive_function_classes fc 
                     WHERE f.fcid = fc.id AND f.bitmap_index={} AND f.tbl_name='{}' AND f.attr_name='{}'"""
        cursor = self.connection.cursor()
        cursor.execute(st_query)

        rows = cursor.fetchall()

        ans = {}
        for row in rows:
            ans[row[0]] = []
            # print(row[1])
            # for i in range(len(row[1])):
            #     if row[1][i] == 0:
            #         continue
            #
            #     cursor.execute(f_query.format(i+1, self.tbl, row[0]))
            #     cursor.fetchall()
            for i in range(len(row[2])):
                row[2][i] = map(lambda x: round(x, 2), row[2][i])
            ans[row[0]].append("State: {}<br/>Probabilities: {}".format(row[1], row[2]))

        cursor.close()
        self.connection.close()

        return ans


class Schema:

    def __init__(self):
        pass

    def fetch_function_table(self):

        self.connection = get_db()

        ft_query = """SELECT fid, fcid, tbl_name, attr_name, bitmap_index, cost, quality FROM progressive_functions"""
        cursor = self.connection.cursor()
        cursor.execute(ft_query)

        rows = cursor.fetchall()

        cursor.close()
        self.connection.close()

        return rows

    def fetch_decision_table(self):

        self.connection = get_db()

        ft_query = """SELECT tbl_name, attr_name, state, uncertainty_ranges, next_func, delta_uncertainty FROM progressive_decision"""
        cursor = self.connection.cursor()
        cursor.execute(ft_query)

        rows = cursor.fetchall()

        cursor.close()
        self.connection.close()

        return rows

    def fetch_function_classes(self):

        self.connection = get_db()

        ft_query = """SELECT id, name FROM progressive_function_classes"""
        cursor = self.connection.cursor()
        cursor.execute(ft_query)

        rows = cursor.fetchall()

        cursor.close()
        self.connection.close()

        return rows

    def fetch_tables_and_attrs(self):

        self.connection = get_db()

        ft_query = """SELECT tbl_name, attr_name, attr_type, num_labels FROM progressive_attrs"""
        cursor = self.connection.cursor()
        cursor.execute(ft_query)

        rows = cursor.fetchall()

        cursor.close()
        self.connection.close()

        return rows


class Performance:

    def __init__(self, token):
        self.token = token
        self.txid = get_txn_id(token)

    def fetch_perf(self):

        self.connection = get_db()

        ft_query = """SELECT * FROM query_perf_{}""".format(self.txid)
        cursor = self.connection.cursor()
        cursor.execute(ft_query)

        rows = cursor.fetchall()

        cursor.close()
        self.connection.close()

        return rows


def insert(stmt):
    pass
