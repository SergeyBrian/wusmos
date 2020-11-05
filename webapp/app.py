import os

from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/machinelearning')
def ml():
    return 'ok'

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/graph')
def graph():
    return render_template('graph.html', res=request.args.get('res'))


if __name__ == '__main__':
    app.run(host='0.0.0.0')
