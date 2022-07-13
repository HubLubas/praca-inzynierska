from flask import Flask, render_template, url_for, redirect, request, jsonify 
from flask_wtf import FlaskForm
from wtforms import IntegerField, SelectField
from random import randint, seed
import time
import yfinance as yf
import datetime
from datetime import date, timedelta
from technical import get_prices

app = Flask(__name__)
app.config['SECRET_KEY'] = 'asdfghjk'


@app.route('/', methods=['GET'])
def index():
    '''
    Main index endpoint
    '''
    return render_template('index.html')

@app.route('/technical/', methods=['GET'])
def technical():
    '''Classification endpoint'''
    
    #response = yf.download("AZN.L", start=datetime.datetime(2010, 1, 1), 
    #                                  end=datetime.datetime(2019, 12, 31), group_by='tickers')
    #response.headers.add('Access-Control-Allow-Origin', '*')
    
    response  = get_prices()
    
    return response



app.run(host='0.0.0.0', port='8098', debug=True)
    