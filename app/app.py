from pathlib import Path
import joblib

import numpy as np
import pandas as pd
from pandas import DataFrame
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import streamlit as st


BASE_DIR = Path(__file__).parent

def prepare_data(df:DataFrame):
    if 'contract' not in df.columns:
        df['contract'] = df['reamining_contract'].apply(lambda x: 0 if pd.isna(x) else 1)
        df.insert(5, 'contract', df.pop('contract'))
        df['reamining_contract'] = df['reamining_contract'].fillna(0)
    return df

def get_potential_lost_customer(data:DataFrame, model, scaler):
    data = prepare_data(data)
    features = data.drop(columns=['id', 'churn'])
    ids = data['id']
    features = scaler.transform(features)
    pred = model.predict(features)
    if len(np.unique(pred)) > 2:
        pred = [1 if p > 0.5 else 0 for p in pred]
    customers = [ids[i] for i in range(len(ids)) if pred[i] == 1]
    return pd.DataFrame(customers, columns=['ID'])

def final_testing(data:DataFrame):
    data = prepare_data(data)
    models = [joblib.load(file) for file in Path(BASE_DIR.parent / 'models').iterdir()]
    scalers = [joblib.load(file) for file in Path(BASE_DIR.parent / 'scalers').iterdir()]
    data = data.drop(columns=['churn'])
    features = data.drop(columns=['id'])
    target = data['churn']

    y_pred_lr = models[0].predict(scalers[0].transform(features))
    y_pred_svc = models[1].predict(scalers[0].transform(features))
    y_pred_rf = models[2].predict(scalers[0].transform(features))
    y_pred_nn =[1 if p > 0.5 else 0 for p in models[3].predict(scalers[1].transform(features))]

    pred = {'LR':y_pred_lr, 'SVC':y_pred_svc, 'RF':y_pred_rf, 'NN':y_pred_nn}

    plt.figure(figsize=(15, 3))
    for i, (m, p) in enumerate(pred.items()):
        cm = confusion_matrix(target, p)
        plt.subplot(1, 4, i+1)
        sns.heatmap(cm, square=True, annot=True, fmt='d', cbar=False, cmap='coolwarm', vmin=0, vmax=100)
        plt.title(f'Confusion matrix for {m} model')
        plt.ylabel('Real')
        plt.xlabel('Predicted');

def model_info(data, model, scaler):
    data = prepare_data(data)
    features = data.drop(columns=['id', 'churn'])
    target = data['churn']
    features = scaler.transform(features)
    pred = model.predict(features)
    if len(np.unique(pred)) > 2:
        pred = [1 if p > 0.5 else 0 for p in pred]
    
    fig = plt.figure(figsize=(15, 3))
    cm = confusion_matrix(target, pred)
    sns.heatmap(cm, square=True, annot=True, fmt='d', cbar=False, cmap='coolwarm', vmin=0, vmax=100)
    plt.ylabel('Real')
    plt.xlabel('Predicted')
    st.pyplot(fig)


client_data = st.file_uploader(label="Choose a CSV file", type='csv')
model_file = st.file_uploader(label="Choose a MODEL", type='pkl')
scaler_file = st.file_uploader(label="Choose a SCALER", type='pkl')

if client_data and model_file and scaler_file:
    df = pd.read_csv(client_data)
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    
    potential_lost_customers = get_potential_lost_customer(df, model, scaler)
    
    st.title("IDs of Potential Lost Customers")
    st.write(potential_lost_customers['ID'].tolist())

    if not potential_lost_customers.empty:
        potential_lost_customers.to_csv(BASE_DIR / 'result.csv', index=False, encoding='utf8')
        with open(BASE_DIR / 'result.csv', 'r') as f:
            st.download_button("Download CSV", f, mime ='text/csv', key=0)
        b0 = st.button("Show analyzing results", key=1)
        if b0:
            model_info(df, model, scaler)