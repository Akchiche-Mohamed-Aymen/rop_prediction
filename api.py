from pickle import load
from pandas import read_csv , DataFrame
from util import create_features 
data = read_csv('test.csv')
c1 = data.drop(columns=['rop']).columns
c2 = data.drop(columns=['is_anomaly' , 'rop']).columns
model = load(open('model_lightgbm.pkl', 'rb'))
anomaly = load(open('isolation_forest_model.pkl', 'rb'))
def predict(input_data):
    try:
        df = DataFrame(input_data)
        df = create_features(df)[c2]
        df['is_anomaly'] = anomaly.predict(df)
        print(df['is_anomaly'])
        prediction = model.predict(df)
        return prediction[0]
    except :
        return -10000000000