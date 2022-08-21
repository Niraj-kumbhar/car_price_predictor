from flask import Flask, render_template, request, redirect
from flask_cors import CORS, cross_origin
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__)

cors=CORS(app)
model=pickle.load(open('model/xgboost_model.pkl','rb'))
car=pd.read_csv('data/Cleaned_car_data.csv')

@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    car['year'] = datetime.now().year - car['age']
    years = sorted(car['year'].unique(), reverse=True)
    fuel = car['fuel'].unique()
    transmissions = car['transmission'].unique()
    owners = car['owner'].unique()
    return render_template('index.html', companies=companies,car_models=car_models, years=years,fuel=fuel, transmissions=transmissions,owners=owners)

@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():

    company=request.form.get('company')

    car_model=request.form.get('car_models')
    year=request.form.get('year')
    #print(year)
    age = datetime.now().year - int(year)
    fuel_type=request.form.get('fuel_type')
    driven1=request.form.get('kilo_driven')
    driven = int(driven1)
    owner=request.form.get('owner')
    transmission = request.form.get('transmission')
    print(np.array([car_model,driven,fuel_type,transmission,owner,age,company]).reshape(1, 7))



    prediction=model.predict(pd.DataFrame(columns=['name','km_driven','fuel','transmission','owner','age','company'],
                              data=np.array([car_model,driven,fuel_type,transmission,owner,age,company]).reshape(1, 7)))
    print(prediction)

    #'name', 'km_driven', 'fuel', 'transmission','owner', 'age', 'company

    return str(np.round(prediction[0],2))

if __name__=="__main__":
    app.run(debug=True)