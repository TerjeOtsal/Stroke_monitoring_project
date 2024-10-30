
from flask import Flask, redirect, url_for, request, render_template, jsonify
import csv

app = Flask(__name__)
app.config['DEBUG'] = True  # Enable debug mode

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/activities')
def activities():
    return render_template('activities.html')

@app.route('/display')
def display():
    return render_template('data.html')

@app.route('/data')
def data():
    data = read_csv_data()
    return jsonify(data)  # Return data as JSON


# Define the path to the CSV file on the microSD card
CSV_FILE_PATH = 'D:/test.csv'  # Update this path as needed

def read_csv_data():
    try:
        with open(CSV_FILE_PATH, mode='r') as file:
            csv_reader = csv.DictReader(file)
            data = [row for row in csv_reader]  # List of dictionaries for each row
        return data
    except FileNotFoundError:
        return {"error": "CSV file not found at specified path."}
    except Exception as e:
        return {"error": str(e)}