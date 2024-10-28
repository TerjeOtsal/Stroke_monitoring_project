
from flask import Flask, redirect, url_for, request, render_template, jsonify

app = Flask(__name__)
app.config['DEBUG'] = True  # Enable debug mode

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/activities')
def activities():
    return render_template('activities.html')