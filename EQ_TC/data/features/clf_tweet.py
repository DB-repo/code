import psycopg2
import time, pickle
import sqlparse
from datetime import datetime
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import re
import random
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from scipy.spatial import distance
import csv
from sklearn.model_selection import StratifiedKFold
from scipy import interp
from itertools import cycle
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings("ignore")
from scipy.spatial import distance
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.svm import libsvm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn import tree
from sklearn.neural_network import MLPClassifier

DB_USER = "postgres"
DB_PWD = "postgres"
DB_HOST = "localhost"
DB_PORT = "5432"
DB = "test"

import sys
import os

clf_dt = pickle.load(open('/usr/local/tagdb/data/features/tweet_dt_sentiment_calibrated.p', 'rb')) # 1K features
clf_gnb = pickle.load(open('/usr/local/tagdb/data/features/tweet_gnb_sentiment_calibrated.p', 'rb')) # 1K features
clf_lda = pickle.load(open('/usr/local/tagdb/data/features/tweet_lda_sentiment_calibrated.p', 'rb')) # 1K features
clf_mlp = pickle.load(open('/usr/local/tagdb/data/features/tweet_mlp_sentiment_calibrated.p', 'rb')) # 1K features

def execute_dt(rl):
	if len(rl) >1000:
		rl = rl[:1000]
	gProb = clf_dt.predict_proba(rl)
	return gProb[0]

def execute_gnb(rl):
	if len(rl) >1000:
		rl = rl[:1000]
	cProb = clf_gnb.predict_proba(rl)
	return cProb[0]
	
def execute_lda(rl):
	if len(rl) >1000:
		rl = rl[:1000]
	gProb = clf_lda.predict_proba(rl)
	return gProb[0]
	
def execute_svm(rl):
	if len(rl) > 1000:
		rl = rl[:1000]
	gProb = clf_svm.predict_proba(rl)
	return gProb[0]

def execute_mlp(rl):
	if len(rl) > 1000:
		rl = rl[:1000]
	gProb = clf_mlp.predict_proba(rl)
	return gProb[0]

def execute_topic_gnb(rl):
    if len(rl) > 1000:
        rl = rl[:1000]
    gProb = clf_topic_gnb.predict_proba(rl)
    return gProb[0]

def execute_topic_lda(rl):
    if len(rl) > 1000:
        rl = rl[:1000]
    gProb = clf_topic_lda.predict_proba(rl)
    return gProb[0]

def execute_topic_lr(rl):
    if len(rl) > 1000:
        rl = rl[:1000]
    gProb = clf_topic_lr.predict_proba(rl)
    return gProb[0]

def execute_topic_knn(rl):
    if len(rl) > 1000:
        rl = rl[:1000]
    gProb = clf_topic_knn.predict_proba(rl)
    return gProb[0]
   
