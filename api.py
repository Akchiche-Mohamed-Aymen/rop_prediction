from pickle import load
from pandas import  DataFrame
import json
with open("stats.json", "r") as f:
    cols = json.load(f)['features']

model = load(open('model.pkl', 'rb'))
def predict(input_data):
    try:
        df = DataFrame(input_data)
        df = df[cols]
        prediction = model.predict(df)
        return prediction[0]
    except :
        return -10000000000