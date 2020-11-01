import os

from flask import Flask, render_template
import machine_learning.machine_learning as mt

app = Flask(__name__)


@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/machinelearning')
def ml():

    return 'ok'


if __name__ == '__main__':
    app.run(host='0.0.0.0')
