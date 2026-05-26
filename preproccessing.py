from pandas import read_csv
from sklearn.preprocessing import LabelEncoder

df = read_csv('drilling.csv')
def convert_mohs(value):
    try:
        if len(value) > 1:
            low, high = value.replace(',', '.').split('-')
            return (float(low) + float(high)) / 2
        else:
            return float(value.replace(',', '.'))
    except Exception as e:
        print(f"Error converting '{value}': {e}")
df['Mohes Index'] = [convert_mohs(val) for val in df['Mohes Index']]
print(df.shape)
df.rename(columns={'Depth: TMD [m]': 'depth_tmd',
                   'Depth: TVD [m]': 'depth_tvd',
                    'RPM  [rpm]': 'rpm',
                    'WOB  [t]': 'wob',
                    'Flow IN  [l/min]': 'flow_in',
                    'ROP  [m/h]': 'rop',
                    'Hardness index 100': 'hardness_index',
                    'Mohes Index Formation': 'mif' , 'Formation': 'formation'}, inplace=True)

le = LabelEncoder()
df= df.drop(columns=['hardness_index'])
df['formationId'] = le.fit_transform(df['formation'])

df.to_csv('cleaned_drilling.csv' , index=False)

#cls ; py preproccessing.py