import numpy as np
import sys
import math
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
import datetime, time

import psycopg2
import math
#from sklearn.metrics import mean_squared_error
from math import sqrt
import pickle
import joblib
#from sklearn.externals import joblib
import random

import sys
import multiprocessing
from multiprocessing import Queue

sys.path.append("/Users/dhrub/Documents/GitHub/MUCT Dataset/FeatureGender/")



#import seaborn.apionly as sns

CONN_STRING = "host='ec2-35-84-134-188.us-west-2.compute.amazonaws.com' user='postgres' password='postgres' dbname='test'"




#Q1, Q2
#FETCH_QUERY_MW = ("SELECT * FROM images where id < 10000")


#Q3
FETCH_QUERY_MW = ("SELECT * FROM tweets where id < 100000")





#conjunctive query
#FETCH_QUERY_MW = ("SELECT * FROM images where images.id < 10000 and images.gender = 1")

# Join query static table
#FETCH_QUERY_MW = ("SELECT * FROM images, tweets where images.gender = tweets.sentiment and images.id < 100 and tweets.location = 'L1' ")






# Join query Tweet

#SELECT  T1.id,T2.id, T1.sentiment, T2.gender  from  tweets_full T1,  images_full  T2  where  T1.sentiment  =  T2.gender  and T1.id<1000 and T2.id<1000

#FETCH_QUERY_MW = ("SELECT images.id, images.feature, tweets.id, tweets.object FROM images, tweets where images.gender = tweets.sentiment and images.id < 1000 and tweets.id < 1000 ")
#FETCH_QUERY_MW = ("SELECT id,feature FROM images where images.id < 1000")
#FETCH_QUERY_MW = ("SELECT id,feature FROM tweets where tweets.id < 1000")


# fetch from images: 

# Aggregation query 
#FETCH_QUERY_MW = ("SELECT gender, count(*) FROM images where id < 1000 group by gender")
#FETCH_QUERY_MW = ("SELECT gender,expression, count(*) FROM images where id < 10000 group by gender, expression")


#FETCH_QUERY_MW = ("SELECT sum(sentiment) FROM tweets where id < 100000")


#FETCH_QUERY_MW = ("SELECT sentiment, count(*) FROM tweets  where tweets.id < 100000 group by sentiment")

#FETCH_QUERY_MW = ("SELECT id, feature FROM imagenet where imagenet.id < 300000")


#July 4 queries
#FETCH_QUERY_MW  = ("SELECT * FROM images where images.id < 10000")
#FETCH_QUERY_MW  = ("SELECT * FROM tweets where tweets.id < 100000")
#FETCH_QUERY_MW  = ("SELECT sentiment, count(*) FROM tweets where tweets.id < 100000")


#FETCH_QUERY_MW = ("SELECT * FROM images, tweets where images.gender = tweets.sentiment and images.id < 100 and tweets.location = 'L1' ")




# 
# 
# 
# gender_gnb = joblib.load(open('/home/ubuntu/supermadlib/SuperMADLib/backend/load/multi_pie_clfs/gender_multipie_gnb_calibrated.p', 'rb'))
# 
# expression_dt = joblib.load(open('/home/ubuntu/supermadlib/SuperMADLib/backend/load/multi_pie_clfs/expression_multipie_dt_calibrated_10k.p', 'rb'))

# 
# 
# dl,nl = pickle.load(open('/Users/dhrub/Documents/GitHub/MUCT Dataset/FeatureGender/MuctTestGender6_XY.p','rb'))
# 
# dl4,nl4 = pickle.load(open('/Users/dhrub/Documents/GitHub/MUCT Dataset/FeatureGender/MuctTestGender6_XY.p','rb'))



# 
sentiment_dt =joblib.load(open('./classifiers/tweet_gnb_sentiment_calibrated.p', 'rb'))
# sentiment_rf = pickle.load(open('/Users/dhrub/Documents/GitHub/SandersTweet/twitter-sentiment-classifier/sentiment_rf.p', 'rb'))
# sentiment_svm = joblib.load(open('/Users/dhrub/Documents/GitHub/SandersTweet/twitter-sentiment-classifier/sentiment_svm.p', 'rb'))

topic_gnb = pickle.load(open('./classifiers/tweet_gnb_topic_calibrated.p', 'rb'))


def genderPredicate1(rl):
	gProb = gender_gnb.predict_proba(rl)
	gProbSmile = gProb[:,1]
	return gProbSmile
	
def genderPredicate2(rl):
	gProb = gender_extraTree.predict_proba(rl)
	gProbSmile = gProb[:,1]
	return gProbSmile
	
	
def genderPredicate3(rl):
	gProb = gender_rf.predict_proba(rl)
	gProbSmile = gProb[:,1]
	return gProbSmile

def genderPredicate4(rl):
	gProb = gender_svm.predict_proba(rl)
	gProbSmile = gProb[:,1]
	return gProbSmile
	
def genderPredicate5(rl):
	gProb = gender_lr.predict_proba(rl)
	gProbSmile = gProb[:,1]
	return gProbSmile
	
def genderPredicate6(rl):
	gProb = gender_dt.predict_proba(rl)
	gProbSmile = gProb[:,1]
	return gProbSmile
	
def genderPredicate7(rl):
	gProb = gender_knn.predict_proba(rl)
	gProbSmile = gProb[:,1]
	return gProbSmile
	
def genderPredicate8(rl):
	gProb = gender_lda.predict_proba(rl)
	gProbSmile = gProb[:,1]
	return gProbSmile
	
def genderPredicate9(rl):
	gProb = gender_sgd.predict_proba(rl)
	gProbSmile = gProb[:,1]
	return gProbSmile
	
def genderPredicate10(rl):
	gProb = gender_nusvc.predict_proba(rl)
	gProbSmile = gProb[:,1]
	return gProbSmile
	
def genderPredicate17(rl):
	gProb = gender_dt_new.predict_proba(rl)
	gProbSmile = gProb[:,1]
	return gProbSmile
	
def genderPredicate18(rl):
	gProb = gender_mlp.predict_proba(rl)
	gProbSmile = gProb[:,1]
	return gProbSmile
	
	
	
###
###### Expression Classifiers############	
def expressionPredicate1(rl):
	gProb = expression_gnb.predict_proba(rl)
	gProbGlass = gProb[:,1]
	return gProbGlass
'''	
def expressionPredicate2(rl):
	gProb = expression_extraTree.predict_proba(rl)
	gProbGlass = gProb[:,1]
	return gProbGlass
'''	
	
def expressionPredicate3(rl):
	gProb = expression_rf.predict_proba(rl)
	#gProb = ethnicity_rf.predict_proba(rl)
	gProbGlass = gProb[:,1]
	return gProbGlass

def expressionPredicate4(rl):
	gProb = expression_sgd.predict_proba(rl)
	gProbGlass = gProb[:,1]
	return gProbGlass	
	
def expressionPredicate6(rl):
	gProb = expression_dt.predict_proba(rl)
	#gProb = ethnicity_dt.predict_proba(rl)
	gProbGlass = gProb[:,1]
	return gProbGlass

def expressionPredicate7(rl):
	gProb = expression_mlp.predict_proba(rl)
	#gProb = ethnicity_knn.predict_proba(rl)
	gProbGlass = gProb[:,1]
	return gProbGlass
	
######
# Sentiment classifiers ##

def sentimentPredicate1(rl):
	gProb = sentiment_gnb.predict_proba(rl)
	return gProb[:,1]	
	
def sentimentPredicate3(rl):
	gProb = sentiment_rf.prob_classify(rl)
	gProbSmile = gProb.prob('1')
	#print gProbSmile
	return gProbSmile

def sentimentPredicate4(rl):
	gProb = sentiment_mlp.predict_proba(rl)
	return gProb[:,1]

def sentimentPredicate6(rl):
	gProb = sentiment_dt.prob_classify(rl)	
	gProbSmile = gProb.prob('1')
	return gProbSmile
	
def sentimentPredicate7(rl):
	gProb = sentiment_knn.prob_classify(rl)
	gProbSmile = gProb.prob('1')
	return gProbSmile

def sentimentPredicate10(rl):
	gProb = sentiment_svm.prob_classify(rl)
	gProbSmile = gProb.prob('1')
	return gProbSmile

def sentimentPredicate12(rl):
	gProb = sentiment_sgd.prob_classify(rl)
	gProbSmile = gProb.prob('1')
	return gProbSmile
	
def sentimentPredicate14(rl):
	gProb = sentiment_maxent.prob_classify(rl)
	gProbSmile = gProb.prob('1')
	return gProbSmile
	
def sentimentPredicate16(rl):
	gProb = sentiment_et.prob_classify(rl)
	gProbSmile = gProb.prob('1')
	return gProbSmile
	
def sentimentPredicate18(rl):
	gProb = sentiment_knn_reduced.prob_classify(rl)
	gProbSmile = gProb.prob('1')
	return gProbSmile
	
def topicPredicate1(rl):
        gProb = topic_gnb.predict_proba(rl)
        return gProb[:,1]

def topicPredicate2(rl):
        gProb = topic_lr.predict_proba(rl)
        return gProb[:,1]

def findQuality(currentProbability):
	probabilitySet = []
	probDictionary = {}
	#t1_q=time.time()
	for i in range(len(dl)):		
		combinedProbability = combineProbability(currentProbability[i])
		probabilitySet.append(combinedProbability)

		value = combinedProbability
		probDictionary[i] = [value]
	
	#t2_q=time.time()
	#print 'time init 1: %f'%(t2_q - t1_q)
	
	#probabilitySet.sort(reverse=True)
	#t1_s=time.time()
	sortedProbSet = probabilitySet[:]
	sortedProbSet.sort(reverse=True)
	#t2_s=time.time()
	#print 'time sort 1: %f'%(t2_s - t1_s)
	
	#t1_th=time.time()
	totalSum = sum(sortedProbSet[0:len(sortedProbSet)])
	prevF1 = 0
	precision =0
	recall = 0
	f1Value = 0
	probThreshold = 0
	sumOfProbability =0
	
	for i in range(len(sortedProbSet)):
		sizeOfAnswer = i
		sumOfProbability = sumOfProbability + sortedProbSet[i]
		
		if i>0:
			precision = float(sumOfProbability)/(i)
			if totalSum >0:
				recall = float(sumOfProbability)/totalSum
			else:
				recall = 0 
			if (precision+recall) >0 :
				f1Value = 2*precision*recall/(precision+recall)
			else:
				f1Value = 0
			#f1Value = 2*float(sumOfProbability)/(totalSum +i)
		
		if f1Value < prevF1 :
			break
		else:
			prevF1 = f1Value
	indexSorted = i
	#print sortedProbSet
	probThreshold = sortedProbSet[indexSorted]
	#print 'indexSorted value : %d'%(indexSorted)
	#print 'threshold probability value : %f'%(probThreshold)
	
	#t2_th=time.time()
	#print 'time threshold 1: %f'%(t2_th - t1_th)
	
	returnedImages = []
	outsideImages = []
	
	
	#t1_ret=time.time()
	for i in range(len(probabilitySet)):
		if probabilitySet[i] > probThreshold:
			returnedImages.append(i)
		else:
			outsideImages.append(i)
			
	#t2_ret=time.time()
	#print 'time return 1: %f'%(t2_ret - t1_ret)
	
	return [prevF1,precision, recall, returnedImages, outsideImages]


def findRealF1(imageList):
	sizeAnswer = len(imageList)	
	sizeDataset = len(nl)
	num_ones = np.count_nonzero(nl == 1)
	
	
	temp_nl = list(nl)
	num_ones =  temp_nl.count(1)  
	
	count = 0
	for i in imageList:
		if nl[i]==1: 
			count+=1
	
	if sizeAnswer > 0 :
		precision = float(count)/sizeAnswer
	else:
		precision = 0
	if num_ones > 0:
		recall = float(count)/num_ones
	else:
		recall = 0
	
	if precision !=0 or recall !=0:
		f1Measure = float(2*precision*recall)/(precision+recall)
	else:
		f1Measure = 0
	
	return f1Measure
Q = Queue()
def execEnrichmment(enrichmentFunction, objectToEnrich):
	Q.put(enrichmentFunction(objectToEnrich))



def ExecuteInLC():
	
	f1 = open('queryTestMW_1.txt','w+')
	set = [sentimentPredicate6, topicPredicate1]
	dl = ExecuteProbeQuery()
	print('number of images = %f'%(len(dl)))	
	currentUncertainty = [1]*len(dl)
	currentProbability = {}
	for i in range(len(dl)):
			key = i
			value = [-1,-1,-1,-1]
			if key in currentProbability:
				currentProbability[key].append(value)
			else:
				currentProbability[key] = [value]	
				
	prevClassifier = {}
	for i in range(len(dl)):
		key = i
		value = [0,0,0,0]
		if key in prevClassifier:
			prevClassifier[key].append(value)
		else:
			prevClassifier[key] = [value]
				
	
	t1 = time.time()
	image_list = []
	print('execution of enrichment start')
	for i in range(len(workflow)):
		operator = workflow[i]
		print(len(dl))
		probValues = operator(dl)
		#print probValues
		for j in range(len(dl)):
		    #print(probValues[j])
			imageProb = probValues[j]
			rocProb = imageProb
			averageProbability = 0;
				
			#index of classifier
			indexClf = set.index(operator)
			tempProb = currentProbability[j][0]
			tempProb[indexClf] = rocProb

		print('round %d completed'%(round))
		set.remove(operator)
		round = round + 1
		t2 = time.time()
		timeElapsed = t2-t1
		#print('Enrichment time = %f'%(timeElapsed))
	print('execution of enrichment end')	
	#print(currentProbability)	
	t2 = time.time()
	timeElapsed = t2-t1
	#print('Enrichment completed')
	print('Enrichment time = %f'%(timeElapsed))



def findF1TwoFeatures(imageList):
	sizeAnswer = len(imageList)	
	sizeDataset = len(nl)
	
	#num_ones = (nl==1).sum()
	num_ones=0
	for j in range(len(nl)):
		if (nl[j]==4 and nl4[j] ==4):
			num_ones+=1
	count = 0
	#print 'number of ones=%f'%(num_ones)
	for i in imageList:
		if (nl[i]==4 and nl4[i]==4):
			#print 'image number=%d'%(i)
			count+=1
	#print 'correct number of ones=%f'%(count)
	
	precision = float(count)/sizeAnswer
	if num_ones >0:
		recall = float(count)/num_ones
	else:
		recall =0
	if precision !=0 and recall !=0:
		f1Measure = (2*precision*recall)/(precision+recall)
	else:
		f1Measure = 0
	#print 'precision:%f, recall : %f, f1 measure: %f'%(precision,recall,f1Measure)
	return f1Measure





def ExecuteProbeQuery():
	
	num_iteration = 1
	print(CONN_STRING)
	
	t1 = time.time()
	for  i in range(num_iteration):
		
	
		cnx = psycopg2.connect(CONN_STRING)
	
	
	
		cur = cnx.cursor()
		cur.execute(FETCH_QUERY_MW)
		print(FETCH_QUERY_MW)
		print(cur)
		count = 0
		feature_list = []
		for row in cur:
			feature_list.append(row[2])
			#print(row[0],row[1])
			count +=1
	
		print(count)
	
	t2 = time.time()
	elapsedTime = t2 - t1
	
	print('DB server time = %f'%(elapsedTime/num_iteration))
	return feature_list
	

	

def EQ_LC_NetworkLatency():
	num_iteration = 1
	print(CONN_STRING)

	for  i in range(num_iteration):
		t1 = time.time()
	
		cnx = psycopg2.connect(CONN_STRING)
	
	
	
		cur = cnx.cursor()
		cur.execute(FETCH_QUERY_DB)
		print(FETCH_QUERY_DB)
		print(cur)
		count = 0
		feature_list = []
		for row in cur:
			#print(row)
			feature_list.append(row[1])
			#print(row[0],row[1])
			count +=1
	
		print(count)
	
	t2 = time.time()
	elapsedTime = t2 - t1
	print(len(feature_list))
	
	print('DB server time = %f'%(elapsedTime/num_iteration))

	


if __name__ == "__main__":
	
        
    ExecuteInLC()     
    #EQ_LC_NetworkLatency()
	#compareWithMWProgressive()
	#compareWithMWProgressiveConjunctive()
	#compareWithMWProgressiveConjunctiveTweet()
	
	
