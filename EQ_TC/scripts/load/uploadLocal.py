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


# dl2, nl2 = pickle.load(
#     open('/home/Downloads/MuctTestGender6_XY.p', 'rb'))

# dl2, nl2 = pickle.load(
#     open('/home/Downloads/MultiPieTestGenderEnrDB_XY.p', 'rb'))

# dl2, nl2 = pickle.load(
#     open('/home/SuperMADLib/experiments/MultiPieTestGenderEnrDB_XY.p', 'rb'))
#

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


def get_connection_with_params(user, pwd, host, port, db):
    connection = psycopg2.connect(user=user,
                                  password=pwd,
                                  host=host,
                                  port=port,
                                  database=db)
    connection.autocommit = True
    return connection


def addImageBlobDataToLocalDB():
    connection = get_connection()

    try:
        cursor = connection.cursor()
        for i in range(300):
            imageID = str(i + 1)
            feature = dl2[i].flatten().tolist()
            feature2 = [str(k) for k in feature[:500]]
            label = str(nl2[i])
            dt = datetime.now()

            data = (imageID, lst2pgarr(feature2), label, dt, 'L' + str(i), i % 20)
            # data = (imageID, lst2pgarr(feature2), dt, 'L' + str(i), i % 20)

            query = "INSERT INTO image_train (id,feature,gender,timestamp,location,cameraid) VALUES (%s, %s, %s, %s, %s, %s);"
            # query = "select madlib.enriched_insert('INSERT INTO images (id,feature,gender,timestamp,location,cameraid) VALUES (%s, ''%s'', NULL, ''%s'', ''%s'', %s)');" % data

            cursor.execute(query, data)
    except (Exception, psycopg2.Error) as error:
        print ("Error while connecting to PostgreSQL", error)
    finally:
        # closing database connection.
        if connection:
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")


def addTweetBlobDataToLocalDB():
    # read all tweets and labels
    fp = open('sentiment.csv', 'rb')
    # fp = open( 'StanfordDataset/trainingandtestdata/training.1600000.processed.noemoticon.csv', 'rb' )
    reader = csv.reader(fp, delimiter=',', quotechar='"', escapechar='\\')
    tweets = []
    labels = []
    index = 0
    for row in reader:
        tweets.append(row[2])
        labels.append(row[1])
        index += 1

        if index == 10002:
            break

    processed_tweets = []

    for tweet in range(1500, len(tweets)):
        # Remove all the special characters
        processed_tweet = re.sub(r'\W', ' ', str(tweets[tweet]))

        # remove all single characters
        processed_tweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_tweet)

        # Remove single characters from the start
        processed_tweet = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_tweet)

        # Substituting multiple spaces with single space
        processed_tweet = re.sub(r'\s+', ' ', processed_tweet, flags=re.I)

        # Removing prefixed 'b'
        processed_tweet = re.sub(r'^b\s+', '', processed_tweet)

        # Converting to Lowercase
        processed_tweet = processed_tweet.lower()

        processed_tweets.append(processed_tweet)

    tfidfconverter = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.7)
    X = tfidfconverter.fit_transform(processed_tweets).toarray()

    connection = get_connection()
    connection.autocommit = False

    try:
        cursor = connection.cursor()

        for i in range(len(X)):
            tweetID = str(i + 1)
            feature = X[i]
            feature2 = [str(k) for k in feature]

            label = str(labels[i])
            dt = datetime.now()
            print(len(feature))
            # data = (tweetID, tweets[i].replace("'", ""), lst2pgarr(feature2), i%4, label, dt, 'L' + str(i))
            data = (tweetID, tweets[i].replace("'", ""), lst2pgarr(feature2), dt, 'L' + str(i))

            # print data
            # query = "INSERT INTO tweets_train (id, tweet_object, feature,topic,sentiment,timestamp ,location) VALUES (%s, %s, %s, %s, %s, %s, %s)"
            query = "select madlib.enriched_insert('INSERT INTO tweets (id, tweet_object, feature,topic,sentiment,timestamp ,location) VALUES (%s, ''%s'', ''%s'', NULL, NULL, ''%s'', ''%s'')');" % data

            print(query)
            # cursor.execute(query, data)
            cursor.execute(query)
            connection.commit()


    except (Exception, psycopg2.Error) as error:
        print ("Error while connecting to PostgreSQL", error)
    finally:
        # closing database connection.
        if connection:
            connection.commit()
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")


if __name__ == '__main__':
    addImageBlobDataToLocalDB()
    addTweetBlobDataToLocalDB()
