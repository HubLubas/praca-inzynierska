from flask import Flask, render_template, url_for, redirect, request, jsonify 
from flask_wtf import FlaskForm
from wtforms import IntegerField, SelectField
from random import randint, seed
from technical_my import sma

app = Flask(__name__)
app.config['SECRET_KEY'] = 'asdfghjk'


@app.route('/', methods=['GET'])
def index():
    '''
    Main index endpoint
    '''
    return render_template('index.html')

#@app.route('/technical/', methods=['GET'])
#def technical():
#    '''Classification endpoint'''
#        
#    response  = sma(20, '2019-01-01','2019-12-31')
#    
#    return response



app.run(host='0.0.0.0', port='8098', debug=True)
    