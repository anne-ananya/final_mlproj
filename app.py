import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open("regmodel.pkl", "rb"))

@app.route('/')
def html_page():
    return render_template('home.html')

@app.route('/submit', methods=['POST','GET'])
def products():
    return render_template('result.html', pred = predict())

def predict():
    # my_dictionary = {'Artist':0,'Banker':1,'Business Owner':2,'Construction Engineer':3, 'Designer':4,'Doctor':5, 'Game Developer':6,
    #                  'Government Officer':7, 'Lawyer':8,'Real Estate Developer':9,'Scientist':10, 'Software Engineer':11,'Stock Investor':12, 
    #                  'Teacher':13,'Unknown':14, 'Writer':15}
    float_features = []
    for x in request.form.values() :
        float_features.append(float(x))
        # if(x=='female'):
        #     float_features.append(1.0)
        # elif(x=='do'):
        #     float_features.append(1.0)
        # elif(x=='donot do'):
        #     float_features.append(0.0)
        # elif(x=='involve'):
        #     float_features.append(1.0)
        # elif(x=='not involve'):
        #     float_features.append(0.0)
        # elif(x=='Artist' or x=='Banker' or x=='Business Owner' or x=='Construction Engineer' or x=='Designer' or x=='Doctor' or x=='Game Developer'or x==
        #              'Government Officer'or x=='Lawyer'or x=='Real Estate Developer' or x=='Scientist'or x== 'Software Engineer'or x=='Stock Investor'or x==
        #              'Teacher'or x=='Unknown'or x=='Writer') :

        #     for i in range(16):
        #         if(i==my_dictionary[x]):
        #             float_features.append(1.0)
        #         else:
        #             float_features.append(0.0)
        # elif 0 <= int(x) <= 100:
        #     float_features.append(float(x))
                        
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return prediction

#if __name__== "__main__":
 #   app.run(debug=True)
