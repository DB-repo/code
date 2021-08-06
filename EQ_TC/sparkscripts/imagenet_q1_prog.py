"""
@file functions.py_in

@namespace progressive
"""
'''
from state import update_state, insert_state
from constants import *
from location_prediction import location_wrapper
from algorithm import generateCombinedProbability,generateCombinedProbabilityUPI
'''
import time
#import plpy
#import uuid
import re
import sys
import numpy as np
sys.path.insert(0, '/home/ubuntu/supermadlib/SuperMADLib/backend/load')

#import clf_image_net
#import clf_multipie
from sklearn.externals import joblib

sys.path.insert(0, '/home/ubuntu/supermadlib/SuperMADLib/backend/load')
import pyspark
from pyspark import SQLContext
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType
from pyspark.sql.functions import udf
from pyspark.sql import Row
from pyspark.sql.functions import col
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark import SparkConf
import pyspark.sql.functions as f
import pickle

url = 'jdbc:postgresql://localhost:5432?user=postgres&password=postgres'
table = 'tweets'
conf = SparkConf()
conf.setMaster("local[*]")
conf.setAppName('pyspark')

properties = {
    'user': 'postgres',
    'password': 'postgres',
    'driver': 'org.postgresql.Driver',
    'spark.jars':'org.postgresql:postgresql:42.2.12'
}


 
sc = pyspark.SparkContext.getOrCreate()
#spark = SQLContext(sc)


spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.jars","/home/ubuntu/java/postgresql-42.2.18.jar") \
    .getOrCreate()

df = spark.read \
    .format("jdbc") \
    .option("url", "jdbc:postgresql://localhost:5432/test") \
    .option("dbtable", "imagenet") \
    .option("user", "postgres") \
    .option("password", "postgres") \
    .load()
#.option("driver", "org.postgresql.Driver") \

#clf_lda = pickle.load(open('/home/ubuntu/supermadlib/SuperMADLib/backend/load/good_clfs/object_lda_calibrated.p', 'rb')) # 5K features
clf_gnb = pickle.load(open('/home/ubuntu/supermadlib/SuperMADLib/backend/load/good_clfs/25k_feature/object_gnb_calibrated_25K.p', 'rb')) # 5K features
clf_knn = pickle.load(open('/home/ubuntu/supermadlib/SuperMADLib/backend/load/good_clfs/object_knn_calibrated.p', 'rb')) # 5K features
#clf_mlp = pickle.load(open('/home/ubuntu/supermadlib/SuperMADLib/backend/load/good_clfs/object_mlp_calibrated.p', 'rb')) # 5K features
#clf_mlp_big = pickle.load(open('/home/ubuntu/supermadlib/SuperMADLib/backend/load/good_clfs/object_mlp_calibrated.p', 'rb')) # 8K features
clf_sgd_25k = pickle.load(open('/home/ubuntu/supermadlib/SuperMADLib/backend/load/good_clfs/25k_feature/object_sgd_calibrated_25K.p', 'rb'))
#clf_dt_25k = pickle.load(open('/home/ubuntu/supermadlib/SuperMADLib/backend/load/good_clfs/25k_feature/object_dt_calibrated_25K.p', 'rb'))


def execute_lda(rl):
        #rl.extend([0]*4500)
        #if len(rl) >5000:
        #       rl = rl[:5000]
        #rl = [rl]
        cProb = clf_lda.predict_proba(rl)
        return cProb[0]

def execute_gnb(rl):
        #if len(rl) >5000:
        #       rl = rl[:5000]
        cProb = clf_gnb.predict_proba(rl)
        return cProb[0]

def execute_knn(rl):
        if len(rl) >5000:
                rl = rl[:5000]
        cProb = clf_knn.predict_proba(rl)
        return cProb[0]

def execute_mlp(rl):
        if len(rl) <= 5000:
                cProb = clf_mlp.predict_proba(rl)
        else:
                cProb = clf_mlp_big.predict_proba(rl)
        return cProb[0]

def execute_sgd(rl):
    cProb = clf_sgd_25k.predict_proba(rl)
    return cProb[0]

def execute_dt(rl):
    cProb = clf_dt_25k.predict_proba(rl)
    return cProb[0]

def run_function(schema_madlib, fid, oid, **kwargs):
    ret = plpy.execute("""SELECT fc.name as fcname, f.tbl_name as tbl, f.model_tbl_name as model_tbl,
                                f.bitmap_index as index, f.attr_name as attr, f.parameters as params
                            FROM {fctbl} as fc, {ftbl} as f
                            WHERE f.fid={fid} AND f.fcid=fc.id
                        """.format(fctbl=FUNC_CLASS_TBL, ftbl=FUNC_TBL, fid=fid))
    model_tbl = ret[0]['model_tbl']
    tbl = ret[0]['tbl']
    fcname = ret[0]['fcname']
    index = int(ret[0]['index'])
    attr = ret[0]['attr']
    parameters = ret[0]['params']

    result, proba = _response(fcname, tbl, model_tbl, oid, attr, parameters)
    with plpy.subtransaction():
        update_state(oid, tbl, attr, index, proba, result)
    # plpy.notice(result)
    return result


def run_function_multi(schema_madlib, req_attr, attr_fids, oid, label, **kwargs):

    attr_fid_split = attr_fids.split(",")
    fid = None
    for af in attr_fid_split:
        attr, func = af.split(":")
        if attr == req_attr:
            fid = func
            break

    if fid is None:
        return -1

    ret = plpy.execute("""SELECT fc.name as fcname, f.tbl_name as tbl, f.model_tbl_name as model_tbl,
                                f.bitmap_index as index, f.attr_name as attr, f.parameters as params
                            FROM {fctbl} as fc, {ftbl} as f
                            WHERE f.fid={fid} AND f.fcid=fc.id
                        """.format(fctbl=FUNC_CLASS_TBL, ftbl=FUNC_TBL, fid=fid))
    model_tbl = ret[0]['model_tbl']
    tbl = ret[0]['tbl']
    fcname = ret[0]['fcname']
    index = int(ret[0]['index'])
    attr = ret[0]['attr']
    parameters = ret[0]['params']

    ret = plpy.execute("""SELECT num_labels FROM {attr_tbl} WHERE tbl_name='{tbl}' AND attr_name='{attr}'
                        """.format(attr_tbl=ATTR_TABLE, tbl=tbl, attr=attr))

    epoch_threshold = plpy.execute("""SELECT threshold FROM {tbl} 
                                        WHERE tbl_name='{tbl_name}' AND attr_name='{attr_name}'
                                        """.format(tbl=THRESHOLD_TBL, tbl_name=tbl, attr_name=attr))[0]['threshold']

    result, proba = _response(fcname, tbl, model_tbl, oid, attr, int(ret[0]['num_labels']), parameters)
    max_prob_label = update_state(oid, tbl, attr, index, proba, epoch_threshold)

    # Logging Performance
    ep_exec = plpy.execute("select current_setting('epoch.executed')::int as ep_exec")[0]['ep_exec']
    ep_succ = plpy.execute("select current_setting('epoch.success')::int as ep_succ")[0]['ep_succ']
    plpy.execute("set epoch.executed = {}".format(ep_exec+1))

    if max_prob_label == label:
        # Logging Performance
        plpy.execute("set epoch.success = {}".format(ep_succ + 1))
        return label
    return None


def run_function_multi_select(schema_madlib, req_attr, attr_fids, oid, **kwargs):

    attr_fid_split = attr_fids.split(",")
    fid = None
    for af in attr_fid_split:
        attr, func = af.split(":")
        if attr == req_attr:
            fid = func
            break

    if fid is None:
        return -1

    ret = plpy.execute("""SELECT fc.name as fcname, f.tbl_name as tbl, f.model_tbl_name as model_tbl,
                                f.bitmap_index as index, f.attr_name as attr, f.parameters as params
                            FROM {fctbl} as fc, {ftbl} as f
                            WHERE f.fid={fid} AND f.fcid=fc.id
                        """.format(fctbl=FUNC_CLASS_TBL, ftbl=FUNC_TBL, fid=fid))
    model_tbl = ret[0]['model_tbl']
    tbl = ret[0]['tbl']
    fcname = ret[0]['fcname']
    index = int(ret[0]['index'])
    attr = ret[0]['attr']
    parameters = ret[0]['params']

    ret = plpy.execute("""SELECT num_labels FROM {attr_tbl} WHERE tbl_name='{tbl}' AND attr_name='{attr}'
                        """.format(attr_tbl=ATTR_TABLE, tbl=tbl, attr=attr))

    epoch_threshold = plpy.execute("""SELECT threshold FROM {tbl} 
                                        WHERE tbl_name='{tbl_name}' AND attr_name='{attr_name}'
                                        """.format(tbl=THRESHOLD_TBL, tbl_name=tbl, attr_name=attr))[0]['threshold']

    result, proba = _response(fcname, tbl, model_tbl, oid, attr, int(ret[0]['num_labels']), parameters)
    max_prob_label = update_state(oid, tbl, attr, index, proba, epoch_threshold)

    # Logging Performance
    ep_exec = plpy.execute("select current_setting('epoch.executed')::int as ep_exec")[0]['ep_exec']
    ep_succ = plpy.execute("select current_setting('epoch.success')::int as ep_succ")[0]['ep_succ']
    plpy.execute("set epoch.executed = {}".format(ep_exec+1))

    if max_prob_label is not None:
        plpy.execute("set epoch.success = {}".format(ep_succ + 1))

    return max_prob_label


def run_function_in(fid, oid):
    run_function(None, fid, oid)


def run_function_in_insert(fid, oid, num_labels):
    ret = plpy.execute("""SELECT fc.name as fcname, f.tbl_name as tbl, f.model_tbl_name as model_tbl, 
                                f.bitmap_index as index, f.attr_name as attr, f.parameters as params 
                            FROM {fctbl} as fc, {ftbl} as f
                            WHERE f.fid={fid} AND f.fcid=fc.id
                        """.format(fctbl=FUNC_CLASS_TBL, ftbl=FUNC_TBL, fid=fid))
    model_tbl = ret[0]['model_tbl']
    tbl = ret[0]['tbl']
    fcname = ret[0]['fcname']
    index = int(ret[0]['index'])
    attr = ret[0]['attr']
    parameters = ret[0]['params']

    result, proba = _response(fcname, tbl, model_tbl, oid, attr, num_labels, parameters)
    insert_state(oid, tbl, attr, index, proba)

    return result, max(proba)

def generateCombinedProbability(functionBitmap, probability2dArr):
    """ Combines probability vectors from different function execution and generates a
    combined probability vector E.g.

    functionBitmap = [1,0,0,1,1,0]
    probability2dArr = [
        [0.6,0,0,0.7,0.9,0],
            [0,0,0,0,0,0],
        [0,0,0,0,0,0],
        [0.7,0,0,0.75,0.95,0],
        [0.6,0,0,0.7,0.8,0],
        [0,0,0,0,0,0]
    ]
    Output: [0.6,0,0,0.18,0.28,0]
    """
    output_arr = [0] * len(probability2dArr[0])
    num_possible_tag = len(probability2dArr[0])

    #weights = [0,1,6,1]
    if num_possible_tag == 2:
        weights = [0, 1, 6, 1] # multipie
    else:
        weights = [0,1, 1 ,6]
    #weights = [0, 4, 6, 8]
    #weights =[0, 1, 1, 6] # tweet works.
    #weights = [0,1,1,6]
    #weights = [0,2,1,4]
    for i in range(num_possible_tag):
        sum_val = 0.0
        count_val = 0
        #for j in range(len(probability2dArr)):
        for j in range(len(functionBitmap)):
            if functionBitmap[j] == 1:
                sum_val += weights[j]*probability2dArr[j][i]
                count_val += weights[j]
        if count_val > 0:
            output_arr[i] = 1.0 * (sum_val / count_val)
        else:
            output_arr[i] = 0

    return output_arr

def _response(input_features):
    fcname = 'imagenet_object_all'
    #plpy.execute("""DROP VIEW IF EXISTS {name}
    #                        """.format(name=VIEW))
    #plpy.execute("""CREATE OR REPLACE TEMP VIEW {name} AS SELECT * FROM {tbl_name} WHERE id={id}
    #                    """.format(name=VIEW, tbl_name=tbl, id=oid))
    #plpy.execute("""DROP TABLE IF EXISTS {name}
    #                        """.format(name=OUTPUT_RESPONSE))
    #plpy.execute("""DROP TABLE IF EXISTS {name}
    #                            """.format(name=OUTPUT_PROBA))

    response = None
    proba = None
    #print 'start'

    if fcname == 'imagenet_object_all':
        bitmap = [1,1,1,1]
        prob2DArray =[]

        features = input_features
        proba = execute_gnb([features])
        prob2DArray.append(proba)

        proba = execute_gnb([features])
        prob2DArray.append(proba)

        proba = execute_knn([features[:5000]])
        prob2DArray.append(proba)

        proba = execute_sgd([features])
        prob2DArray.append(proba)

        output = generateCombinedProbability(bitmap, prob2DArray)

        proba = output
        response = 0
        
        #print output
    #return list(output)
    #print type(output)
    res = [round(v,4) for v in output]
    max_val = max(res)
    label = res.index(max_val)
    return label+1


df.printSchema()
t1 = time.time() 
_select_sql = "(select id,object,feature from imagenet where id > 194000 and id < 194500) as my_table"


if __name__ == "__main__":

    start_id = 191500
    end_id = 191701
    _select_sql = "(select i1.id,i1.feature, i2.object from imagenet i1, imagenet_full i2 where i1.id = i2.id and i1.id >" + str(start_id)+ " and i1.id <" + str(end_id) +  " and i2.id >" + str(start_id) +" and i2.id  <" + str(end_id) + ") as my_table"
    df_select = spark.read.jdbc(url="jdbc:postgresql://localhost:5432/test",table=_select_sql,properties=properties)
    df_select.show()
    truth_list = df_select.select('object').rdd.flatMap(lambda x: x).collect()
    all_f1 = []
    truth = 0
    for i in range(len(truth_list)):
        if truth_list[i] == 6:
                truth+=1
    prev_recall =0
    num_epoch = 25
    epoch_size = 8

    output_udf_float = udf(_response, IntegerType())
    for i in range(25):
        t1 = time.time()
        start_id = 191500 + i * epoch_size
        end_id = start_id + epoch_size
        _select_sql = "(select i1.id,i1.feature,i2.object from imagenet i1, imagenet_full i2 where i1.id = i2.id and i1.id >" + str(start_id)+ " and i1.id <" + str(end_id) +  " and i2.id >" + str(start_id) +" and i2.id  <" + str(end_id) + ") as my_table"
        print _select_sql
        df_select = spark.read.jdbc(url="jdbc:postgresql://localhost:5432/test",table=_select_sql,properties=properties)
        df_select.show()
        output_udf_float = udf(_response, IntegerType())

        df4 = df_select.withColumn('exec',output_udf_float('feature').alias('exec_output'))
        df4.show()
        pred_list = df4.select('exec').rdd.flatMap(lambda x: x).collect()
        #truth_list = df4.select('sentiment').rdd.flatMap(lambda x: x).collect()

        count = 0
        correct = 0
        for j in range(len(pred_list)):
            if pred_list[j] == 6:
                count+=1
            #print '(j + i * epoch_size) = %d'%(j + i*epoch_size)
            #print 'len true list = %d'%(len(truth_list))
            if pred_list[j] == truth_list[j + i*epoch_size] and pred_list[j] == 6:
                correct+=1

        print pred_list
        print truth_list
        if count !=0:
            prec = correct*1.0/count
        else:
            prec =0

        recall = correct* 1.0 /truth
        total_recall = prev_recall + recall
        prev_recall = total_recall
        if (prec + total_recall) != 0:
            f1 = 2* prec * total_recall / (prec + total_recall)
        else:
            f1 = 0
        print f1
        all_f1.append(f1)
        print all_f1

    print all_f1
