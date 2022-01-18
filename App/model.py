import pandas as pd
import numpy as np

# Load Data Viz
import seaborn as sns
# Load Text Cleaning 
import neattext.functions as nfx

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Transformers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# File handling
from google.cloud import storage
import gcsfs
import joblib
import os
from dotenv import load_dotenv

load_dotenv()



def preprocessing(df):
    """
    Remove userHandles (@Gary => Gary)
    Remove stopwords
    """
    # User handles
    df['Clean_Text'] = df['Text'].apply(nfx.remove_userhandles)

    # Stopwords
    df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_stopwords)

    # Features & Labels
    Xfeatures = df['Clean_Text']
    Ylabels = df['Emotion']

    return Xfeatures, Ylabels


def pipeline(Xfeatures,Ylabels):
    """
    Split dataset (30% test_set, 70% trainset)
    Pipeline for training (CounVectorizer and LogisticRegression)
    """

    #  Split Data
    x_train,x_test,y_train,y_test = train_test_split(Xfeatures,Ylabels,test_size=0.3)

    # LogisticRegression Pipeline
    pipe_lr = Pipeline(steps=[('cv',CountVectorizer()),('lr',LogisticRegression(max_iter=1500))])

    # Train and Fit Data
    pipe_lr.fit(x_train,y_train)

    #Check accuracy
    #pipe_lr.score(x_test,y_test)
    return pipe_lr

def upload_model(pipe_lr):
    """
    Upload file to GCS(Google Cloud Storage)
    """
    pipeline_file = open(os.environ.get("model_file"),"wb")
    joblib.dump(pipe_lr,pipeline_file)
    pipeline_file.close()
    client = storage.Client()
    bucket = client.get_bucket(os.environ.get("bucket_name"))
    blob = bucket.blob(os.environ.get("model_file"))
    blob.upload_from_filename(os.environ.get("model_file"))
    pipeline_file.close()

def download_data(): 
    temp = pd.read_csv('gs://'+'04abbd2b-c18d-4799-b9f3-762020f4c180' +'/'+'data'+'.csv', encoding='utf-8')
    return temp


def main():
    df = download_data()
    Xfeatures, Ylabels = preprocessing(df)
    pipe_lr = pipeline(Xfeatures, Ylabels)
    upload_model(pipe_lr)

main()