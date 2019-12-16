# -*- coding: utf-8 -*-
from flask import Flask,request,render_template
import pickle

app= Flask(__name__)
@app.route('/')
def home():
    return render_template('Main.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    model = pickle.load(open('model.pkl','rb'))
    
    text = "wow nice"  #request.form['Feedback']
    
    cv = pickle.load(open('cv.pkl','rb'))
    
    x = cv.transform([text])
    
    x = x.toarray()
    
    str1= str(model.predict(x))
    '''
    if str1=='[1]':
        return ('<h1>' + request.form['Name']+" liked the restaurant" + '</h1>')
    else:
        return('<h1>' + request.form['Name']+" Not liked the restaurant" + '</h1>')
     
    '''
    return "Hi Welcome to nlp "+str1
    

if __name__ =='__main__':
    app.run()
