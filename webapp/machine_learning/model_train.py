import pandas
import math
from sklearn.preprocessing import LabelEncoder, normalize, scale
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score
from random import choice

from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

from sklearn.preprocessing import LabelEncoder, normalize, scale
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


DIR = ''
if __name__ != '__main__':
    DIR = 'machine_learning/'


def new_model():
    train = pandas.read_csv(DIR + 'twitter_data_public.csv')
    train = train.drop("Unnamed: 0", axis=1)
    test = train.truncate(before=50000)
    print(test)

    return 'ok'


def start_learning():
    model = None
    try:
        with open('model.dat', 'rb') as file:
            model = pickle.load(file)
    except OSError:
        model = new_model()
    return model
