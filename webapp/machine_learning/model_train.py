import pandas as ps
import numpy as np
import pickle


def new_model():
    return "HELLO!!!"


def start_learning():
    model = None
    try:
        with open('model.dat', 'rb') as file:
            model = pickle.load(file)
    except OSError:
        model = new_model()
    return model
