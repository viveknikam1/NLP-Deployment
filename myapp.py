# -*- coding: utf-8 -*-
from flask import Flask,request,render_template
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app= Flask(__name__)
@app.route('/')
def home():
    return render_template('Main.html')

@app.route('/predict',methods=['POST'])
def predict():
    PS = PorterStemmer()
    text = request.form['Feedback']
    RF_model = pickle.load(open('RF_model.pkl','rb'))
    CV_model = pickle.load(open('CV.pkl','rb'))
    msg = re.sub('[^a-zA-Z]',' ',text).lower().split()
    msg = [PS.stem(word) for word in msg if not word in stopwords.words('english')]
    msg = ' '.join(msg)
    X = CV_model.transform([msg]).toarray()
    output = RF_model.predict(X)[0]
    if output==1:
        return request.form['Name'] + ' is Liked the restaurant'
    else:
        return request.form['Name'] + ' is Didnt Liked the restaurant'
    

if __name__ =='__main__':
    app.run(debug = True)
