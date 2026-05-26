import pandas as pd
from sklearn.ensemble import IsolationForest 

df = pd.read_csv('cleaned_drilling.csv')
iso = IsolationForest(
    n_estimators=100,
    contamination=0.02,
    random_state=42,
    n_jobs=-1
)

df['is_anomaly'] = iso.fit_predict(df.drop(columns=['rop']))
df['is_anomaly'] = (df['is_anomaly'] == -1).astype(int)

df.to_csv('drilling_with_anomalies.csv', index=False)
from pickle import dump
dump(iso, open('isolation_forest_model.pkl', 'wb'))
print(iso.feature_names_in_)
#py check_anomalies.py