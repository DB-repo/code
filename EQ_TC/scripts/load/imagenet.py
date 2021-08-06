import psycopg2
import time, pickle
import sqlparse
from datetime import datetime
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import re
import random
import mysql.connector
import json

DB_USER = "postgres"
DB_PWD = "postgres"
DB_HOST = "localhost"
DB_PORT = "5432"
DB = "test"

import sys
import os
from skimage.feature import hog
from skimage import io
from skimage import data, color, exposure

try:
    from cPickle import dumps, loads
except ImportError:
    from pickle import dumps, loads


def hogx(fileName):
    ri = io.imread(fileName)
    ii = color.rgb2gray(ri)
    fd = hog(ii)
    return fd


def lst2pgarr(alist):
    return '{' + ','.join(alist) + '}'


def get_connection():
    connection = psycopg2.connect(user=DB_USER,
                                  password=DB_PWD,
                                  host=DB_HOST,
                                  port=DB_PORT,
                                  database=DB)
    connection.autocommit = True
    return connection


def addImageBlobDataToLocalDB():
    connection = get_connection()
    dirPath = '/home/ubuntu/imagenet_val'
    gtPath = '/home/ubuntu/imagenet_labels/ILSVRC2012_validation_ground_truth.txt'
    imgList = os.listdir(dirPath)
    imgList.sort()

    gt = map(int, open(gtPath, 'r').readlines())


    try:
        cursor = connection.cursor()

        i = 0
        for imgName in imgList:
            if gt[i] > 10:
                continue

            imgPath = os.path.join(dirPath, imgName)

            # feature extraction of image using the path to image.
            feature = hogx(imgPath)

            imageID = str(i + 1)
            feature2 = [str(k) for k in feature[:500]]
            label = str(gt[i])

            data = (imageID, lst2pgarr(feature2), label)
            data = (imageID, lst2pgarr(feature2))

            query = "INSERT INTO imagenet_train (id,feature,object) VALUES (%s, %s, %s);"
            query = "select madlib.enriched_insert('INSERT INTO imagenet (id,feature,object) VALUES (%s, ''%s'', NULL)');" % data

            cursor.execute(query, data)
    except (Exception, psycopg2.Error) as error:
        print ("Error while connecting to PostgreSQL", error)
    finally:
        # closing database connection.
        if connection:
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")


def addImageNetTrainData():
    connection = get_connection()
    dirPath = '/home/ubuntu/imagenet_val'
    gtPath = '/home/ubuntu/imagenet_labels/ILSVRC2012_validation_ground_truth.txt'
    mapPath = '/home/ubuntu/imagenet_labels/ILSVRC2012_mapping.txt'
    imgList = os.listdir(dirPath)
    imgList.sort()

    gt = map(int, open(gtPath, 'r').readlines())
    mapping = {x[1]:x[0] for x in map(lambda x: x.split(), open(mapPath, 'r').readlines())}


    try:
        cursor = connection.cursor()

        i = 1000
        for imgName in imgList:
            label_name = imgName.split("_")[0]
            if mapping[label_name] > 10:
                continue

            imgPath = os.path.join(dirPath, imgName)

            # feature extraction of image using the path to image.
            feature = hogx(imgPath)

            imageID = str(i + 1)
            feature2 = [str(k) for k in feature[:500]]
            label = str(gt[i])

            data = (imageID, lst2pgarr(feature2), label)
            data = (imageID, lst2pgarr(feature2))

            query = "INSERT INTO imagenet_train (id,feature,object) VALUES (%s, %s, %s);"
            query = "select madlib.enriched_insert('INSERT INTO imagenet (id,feature,object) VALUES (%s, ''%s'', NULL)');" % data

            cursor.execute(query, data)
    except (Exception, psycopg2.Error) as error:
        print ("Error while connecting to PostgreSQL", error)
    finally:
        # closing database connection.
        if connection:
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")



if __name__ == '__main__':
    addImageBlobDataToLocalDB()

