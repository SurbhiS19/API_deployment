
import pandas as pd
from flask import Flask, request, jsonify
import pickle


app = Flask(__name__)


@app.route('/', methods=['GET','POST'])

def home():
    if(request.method=='GET'):
        
        data = "hello world"
        return jsonify({'data': data})


@app.route('/predict')

def price_predict():
    model = pickle.load(open('model.pkl', 'rb'))
    income = request.args.get('income')
    house_age = request.args.get('house_age')
    rooms = request.args.get('rooms')
    bedrooms = request.args.get('bedrooms')
    population = request.args.get('population')
    
    test_df =pd.DataFrame({'Income':[income],'House Age':[house_age],'Rooms':[rooms],'Bedrooms':[bedrooms],'Population':[population]})
    
    
    predict_price= model.predict(test_df)
    return jsonify({'House Price': str(predict_price)})
 
#http://127.0.0.1:5000/predict?income=10&house_age=2&rooms=2&bedrooms=3&population=6   
    

if __name__=="__main__":
    app.run(debug=True)