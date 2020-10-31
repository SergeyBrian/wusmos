from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.pipeline import Pipeline

import numpy


def create_vectorizer():
    return CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None,
                                 max_features=18000, min_df=2, ngram_range=(1, 3))


def create_bag_of_words(data_frame):
    pipe = Pipeline([('vect', create_vectorizer())])
    bag_of_words = pipe.fit_transform(data_frame['clean'])

    bw_dict = vectorizer.get_feature_names()
    res = pandas.DataFrame(numpy.sum(bag_of_words, axis=0), columns=bw_dict)
    return res


def create_model(train):
    kfold = StratifiedKFold()
    svc = LinearSVC()

    params = {
        'loss': ['hinge'],
        'class_weight': [{1: 1}],
        'C': [0.01]
    }

    model = GridSearchCV(svc, param_grid=[params], verbose=1, cv=kfold, n_jobs=-1, scoring='roc_auc')
    model.fit(train_bw, train['target'])
    return model


# test should be result of create_bag_of_words function
# model should be result of create_model function
def predict(test, model):
    return model.predict(test)
