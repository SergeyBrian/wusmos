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


nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

embeddings_dict = {}

with open('glove.6B.50d.txt', 'r', encoding="utf-8") as f:
  for line in f:
    values = line.split()
    word = values[0]
    vector = np.asarray(values[1:], "float32")
    embeddings_dict[word] = vector

DIR = ''
if __name__ != '__main__':
    DIR = 'machine_learning/'

features = train.drop("Unnamed: 0", axis=1)
test = features.truncate(before='50001')
features = features.truncate(after='50000')


def find_closest_embeddings(embedding):
    return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))


def vectorize(tokens):
    word_vectors_list = []
    empty_vector = [0 for i in range(50)]
    rtokens = [i for i in tokens if not i in stop_words and i not in stop_symbols]
    for token in rtokens:
        # print(token)
        token = lemmatizer.lemmatize(token, 'v')
        token = re.sub(r'(.)\1+', r'\1', token)
        token = re.sub('[^A-Za-z0-9]+', '', token)
        token = lemmatizer.lemmatize(token, 'v')
        # print("Final: " + token)

        if token in embeddings_dict.keys():
            word_vectors_list.append(embeddings_dict[token])
        else:
            word_vectors_list.append(empty_vector)
    return word_vectors_list


def preprocess(x):
    try:
        x = x.lower().strip()
        x = x.split(' ')
    except Exception:
        print("PROBLEM PHRASE")
        print(x)
        raise Exception

    return x


def preproc(features):
    conv = features
    conv[['text']] = conv.text.apply(preprocess)
    return conv


print("==============")
print("Preprocessing...\t[1/2]")
features = preproc(features)
print("Preprocessing...\t[2/2]")
test = preproc(test)

print("Vectorizing...\t[1/2]")
features[['text']] = features.text.apply(lambda x: vectorize(x))
print("Vectorizing...\t[2/2]")
test[['text']] = test.text.apply(lambda x: vectorize(x))
features.head()

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
