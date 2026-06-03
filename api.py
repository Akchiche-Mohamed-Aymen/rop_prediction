from pickle import load
from pandas import  DataFrame
import json
from util import create_features
with open("stats.json", "r") as f:
    data = json.load(f)
    cols1 = data['init_features']
    cols2 = data['features']

model = load(open('model.pkl', 'rb'))
kmeans = load(open('kmeans.pkl', 'rb'))
def predict(input_data):
    try:
        df = DataFrame(input_data)
        df = df[cols1]
        df = create_features(df)
        df ['cluster'] = kmeans.predict(df)
        df = df[cols2]
        prediction = model.predict(df)
        print(prediction)
        return prediction[0]
    except :
        return -10000000000