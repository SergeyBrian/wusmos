import os

from flask import Flask, render_template
import machine_learning.model_train as mt

app = Flask(__name__)


@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/machinelearning')
def ml():
    return mt.start_learning()


if __name__ == '__main__':
    app.run(host='0.0.0.0')
