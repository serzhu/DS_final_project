from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent

def get_potential_lost_customer(data, model, scaler):
    features = data.drop(columns=['id', 'churn'])
    ids = data['id']
    features = scaler.transform(features)
    pred = model.predict(features)
    if len(np.unique(pred)) > 2:
        pred = [1 if p > 0.5 else 0 for p in pred]
    customers = [ids[i] for i in range(len(ids)) if pred[i] == 1]
    return customers

client_data = st.file_uploader(label="Choose a CSV file", type='csv')
model_file = st.file_uploader(label="Choose a MODEL", type='pkl')
scaler_file = st.file_uploader(label="Choose a scaler", type='pkl')

if client_data and model_file and scaler_file:
    df = pd.read_csv(client_data)
    
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    
    potential_lost_customers = get_potential_lost_customer(df, model, scaler)
    
    st.title("IDs of Potential Lost Customers")
    st.write(potential_lost_customers)

    if potential_lost_customers:
        if st.button("Save results"):
            df_result = pd.DataFrame(potential_lost_customers, columns=['ID'])
            df_result.to_csv(BASE_DIR / 'result.csv', index=False)
            st.success("Results saved successfully!")