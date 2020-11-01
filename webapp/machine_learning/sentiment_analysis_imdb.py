

import pandas as pd
import numpy as np
import re

train = pd.read_csv("twitter_data_public.csv")
test = pd.read_csv("twitter_data_private_nolabels.csv")



from bs4 import BeautifulSoup

def cleaning(raw_text):
    import nltk
    
    # 1. Remove HTML.
    html_text = BeautifulSoup(raw_text,"html.parser").get_text()
    
    # 2. Remove non-letters.
    letters = re.sub("[^a-zA-Z]", " ", html_text)
    
    # 3. Convert to lower case.
    letters = letters.lower()
    
    # 4. Tokenize.
    tokens = nltk.word_tokenize(letters)
    
    # 5. Convert the stopwords list to "set" data type.
    stops = set(nltk.corpus.stopwords.words("english"))
    
    # 6. Remove stop words. 
    words = [w for w in tokens if not w in stops]
    
    # 7. Stemming
    words = [nltk.stem.SnowballStemmer('english').stem(w) for w in words]
    
    # 8. Join the words back into one string separated by space, and return the result.
    return " ".join(words)

test['clean'] = test['text'].apply(cleaning)

train.head()

test.head()


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer = "word", 
                             tokenizer = None, 
                             preprocessor = None, 
                             stop_words = None, 
                             max_features = 18000,
                             min_df = 2,
                             ngram_range = (1,3)
                            )


from sklearn.pipeline import Pipeline

pipe = Pipeline( [('vect', vectorizer)] )


train_bw = pipe.fit_transform(train['clean'])


test_bw = pipe.transform(test['clean'])



lexi = vectorizer.get_feature_names()



train_sum = pd.DataFrame(np.sum(train_bw, axis=0), columns = lexi)

train_sum.head()


from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier

kfold = StratifiedKFold( n_splits = 5, random_state = 2018 )


# LinearSVC

sv = LinearSVC(random_state=2018)

param_grid2 = {
    'loss':['hinge'],
    'class_weight':[{1:1}],
    'C': [0.01]
}

gs_sv = GridSearchCV(sv, param_grid = [param_grid2], verbose = 1, cv = kfold, n_jobs = -1, scoring = 'roc_auc' )
gs_sv.fit(train_bw, train['target'])
gs_sv_best = gs_sv.best_estimator_
print(gs_sv.best_params_)

# {'C': 0.01, 'class_weight': {1: 1}, 'loss': 'hinge'} - 0.88104

submission1 = gs_sv.predict(test_bw)

print(gs_sv.best_score_)

"""### 6.2 Bernoulli Naive Bayes Classifier <a id='bnb'></a>"""

bnb = BernoulliNB()
gs_bnb = GridSearchCV(bnb, param_grid = {'alpha': [0.03],
                                         'binarize': [0.001]}, verbose = 1, cv = kfold, n_jobs = -1, scoring = 'roc_auc')
gs_bnb.fit(train_bw, train['target'])
gs_bnb_best = gs_bnb.best_estimator_
print(gs_bnb.best_params_)

# {'alpha': 0.1, 'binarize': 0.001} - 0.85240
# {'alpha': 0.03, 'binarize': 0.001} - 0.85240

submission2 = gs_bnb.predict(test_bw)

print(gs_bnb.best_score_)

"""### 6.4 Logistic Regression <a id='logi'></a>"""

lr = LogisticRegression(random_state = 2018)


lr2_param = {
    'penalty':['l2'],
    'dual':[False],
    'C':[0.05],
    'class_weight':['balanced']
    }

lr_CV = GridSearchCV(lr, param_grid = [lr2_param], cv = kfold, scoring = 'roc_auc', n_jobs = -1, verbose = 1)
lr_CV.fit(train_bw, train['target'])
print(lr_CV.best_params_)
logi_best = lr_CV.best_estimator_


# {'C': 0.1, 'class_weight': 'balanced', 'dual': False, 'penalty': 'l2'} - 0.87868
# {'C': 0.05, 'class_weight': 'balanced', 'dual': False, 'penalty': 'l2'} - 0.88028

submission4 = lr_CV.predict(test_bw)

print(lr_CV.best_score_)