from flask import Flask, render_template, url_for, redirect, request
from flask_wtf import FlaskForm
from wtforms import IntegerField, SelectField
from random import randint, seed
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'asdfghjk'


@app.route('/', methods=['GET'])
def index():
    '''
    Main index endpoint
    '''
    return render_template('index.html')



app.run(host='0.0.0.0', port='8098', debug=True)
