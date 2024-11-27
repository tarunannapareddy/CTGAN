import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import time

# Outcome mapping for the labels
outcome_mapping = {
    'normal': 0,
    'dos': 1,
    'probe': 2,
    'r2l': 3,
    'u2r': 4,
}

def load_data(train_path, test_path):
    data_train = pd.read_csv(train_path)
    data_test = pd.read_csv(test_path)
    return data_train, data_test

# Scaling function using RobustScaler
def scale_numeric_features(df_num, cols):
    std_scaler = RobustScaler()
    std_scaler_temp = std_scaler.fit_transform(df_num)
    return pd.DataFrame(std_scaler_temp, columns=cols)

# Function to preprocess the data
def preprocess(dataframe, cat_cols):
    df = dataframe.copy()
    df_num = df.drop(cat_cols, axis=1)
    num_cols = df_num.columns
    scaled_df = scale_numeric_features(df_num, num_cols)

    df.drop(labels=num_cols, axis="columns", inplace=True)
    df[num_cols] = scaled_df[num_cols]
    
    # Map label column based on the outcome_mapping
    df['label'] = df['label'].map(outcome_mapping)
    
    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=['protocol_type', 'service', 'flag'], drop_first=True)
    
    return df

data_train, data_test = load_data('NSL-KDD/KDDTrainSynthatic.CSV', 'NSL-KDD/KDDTest.CSV')
cat_cols = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login', 'label']

data_test = data_test.head(2)
# Preprocess datasets and align columns
scaled_train = preprocess(data_train, cat_cols)
scaled_test = preprocess(data_test, cat_cols)
scaled_test = scaled_test.reindex(columns=scaled_train.columns, fill_value=0)

# Separate features and labels
x_train = scaled_train.drop(['label'], axis=1).values
y_train = scaled_train['label'].values
x_test = scaled_test.drop(['label'], axis=1).values
y_test = scaled_test['label'].values


# API endpoint
url = 'http://localhost:5000/predict'

# Make a POST request
start_time = time.time() 
response = requests.post(url, json={'x_test_point': x_test.tolist()})
end_time = time.time()

print(f"API call took {end_time - start_time:.4f} seconds")

# Print the response
if response.status_code == 200:
    print("Predictions:", response.json()['predictions'])
else:
    print("Error:", response.json())
