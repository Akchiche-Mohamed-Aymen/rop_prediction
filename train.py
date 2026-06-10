
from sklearn.metrics import mean_squared_error , r2_score , mean_absolute_error   , mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from util import create_features
from sklearn.cluster import KMeans
import pandas as pd

#====================================================================
def evaluate_model(model,X, y_actual):
    y_pred = model.predict(X)
    mse = mean_squared_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
    mae = mean_absolute_error(y_actual, y_pred)
    mape = mean_absolute_percentage_error(y_actual, y_pred)
    print(f'RMSE : {round(mse**0.5, 2)}')
    print(f'R^2 Score: {round(100*r2, 2)} %')
    print(f'MAE: {round(mae, 2)}')
    print(f'MAPE: {round(mape, 2)}')
def remove_outlier(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df
# Load the cleaned dataset
df = pd.read_csv('cleaned_drilling.csv' , usecols=['rpm', 'wob', 'flow_in','rop' , 'depth_tmd' , 'depth_tvd' ])
init_features = list(df.columns)
init_features.remove('rop')
df = create_features(df)
df["rop"] = df["rop"].clip( df["rop"].quantile(0.01),df["rop"].quantile(0.99))
X = df.drop(columns=['rop'])
y = df['rop']
kmeans = KMeans(n_clusters=7, random_state=42)
kmeans.fit(X)
X['cluster'] = kmeans.labels_
model = RandomForestRegressor(n_estimators=35,min_samples_leaf=5,max_depth=12, random_state=42 )
model.fit(X , y)
importances = list(model.feature_importances_)
feature_names = list(X.columns)
evaluate_model(model , X , y)
import json 
with open('stats.json', "w", encoding="utf-8") as f:
    json.dump({
    "features":feature_names,
    "init_features":init_features,
    "importances": importances
}, f, ensure_ascii=False, indent=4)

import pickle
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("kmeans.pkl", "wb") as f:
    pickle.dump(kmeans, f)
