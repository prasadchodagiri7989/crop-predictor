from flask import Flask, render_template, request
import pickle
import numpy as np


label=['rice' ,'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans',
 'mungbean', 'blackgram', 'lentil', 'pomegranate', 'banana', 'mango', 'grapes',
 'watermelon', 'muskmelon', 'apple', 'orange', 'papaya', 'coconut', 'cotton',
 'jute', 'coffee']

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/disease', methods=['GET'] )
def disease():
    return render_template('disease.html')

@app.route('/solution', methods=['Get'])
def solution():
    return render_template('solution.html')

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/home' ,methods=['GET'])
def crop():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    data6 = request.form['f']
    data7 = request.form['g']
    arr = np.array([[int(data1), int(data2), float(data3), int(data4), int(data5), int(data6), int(data7)]])
    pred = model.predict(arr)
    print(pred)
    res = label[int(pred)]
    image_path = "/static/assets/"+res+".jpg"
    return render_template('after.html', data=res, image_path=image_path)


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')















