import pandas
import numpy as np
import pickle


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
