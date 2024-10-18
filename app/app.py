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

def prepare_data(df: DataFrame) -> DataFrame:
    if 'contract' not in df.columns:
        df['contract'] = df['reamining_contract'].apply(lambda x: 0 if pd.isna(x) else 1)
        df.insert(5, 'contract', df.pop('contract'))
        df['reamining_contract'] = df['reamining_contract'].fillna(0)
        df.dropna(subset=['download_avg', 'upload_avg'], inplace=True, ignore_index=True)
    return df

def get_potential_lost_customers(df: DataFrame, model, scaler):
    data = prepare_data(df)
    features = data.drop(columns=['id', 'churn'])
    ids = data['id']
    features = scaler.transform(features)
    pred = model.predict(features)

    pred = [1 if p > 0.5 else 0 for p in pred] if len(np.unique(pred)) > 2 else pred
    customers = [ids[i] for i in range(len(ids)) if pred[i] == 1]
    return pd.DataFrame(customers, columns=['ID']), pred

def plot_confusion_matrix(data, pred):
    data = prepare_data(data)
    target = data['churn']
    cm = confusion_matrix(target, pred)

    plt.figure(figsize=(2, 2))
    sns.heatmap(cm, square=True, annot=True, fmt='d', cbar=False, cmap='coolwarm', vmin=0, vmax=len(target)/2)
    plt.ylabel('Real')
    plt.xlabel('Predicted')
    st.pyplot(plt)

def upload_files():
    client_data = st.file_uploader("Choose a CSV file", type='csv')
    model_file = st.file_uploader("Choose a MODEL", type='pkl')
    scaler_file = st.file_uploader("Choose a SCALER", type='pkl')
    return client_data, model_file, scaler_file

def display_potential_lost_customers(cstm, ids):
    num_columns = 3
    columns = st.columns(num_columns)
    
    for col_num in range(num_columns):
        start_index = col_num * (len(cstm) // num_columns)
        end_index = start_index + (len(cstm) // num_columns)
        cstm_col = cstm[start_index:end_index] 

        colored_customers = [
            f"<span style='color:red'>{id}</span>" if id in ids else str(id)
            for id in cstm_col
        ]
        columns[col_num].markdown('  \n'.join(colored_customers), unsafe_allow_html=True)

def main():
    client_data, model_file, scaler_file = upload_files()

    if client_data and model_file and scaler_file:
        df = pd.read_csv(client_data)
        ids = df[(df['reamining_contract'] >= 0) & (df['reamining_contract'] < 0.1) & (df['churn'] == 0)]['id'].to_list()

        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
        
        cstm, pred = get_potential_lost_customers(df, model, scaler)
        cstm_ids = cstm['ID'].tolist()

        st.title("IDs of Potential Lost Customers")
        display_potential_lost_customers(cstm_ids, ids)

        st.markdown("#### <span style='color:red'>Red ID</span> indicates that active customer has a contract ending in ~1 month. Model predicted it as a potential lost customer!", unsafe_allow_html=True)

        if cstm_ids:
            result_df = pd.DataFrame(cstm_ids, columns=['ID'])
            result_file = BASE_DIR / 'result.csv'
            result_df.to_csv(result_file, index=False, encoding='utf8')

            with open(result_file, 'r') as f:
                st.download_button("Download CSV", f, mime='text/csv')

            if st.button("Show analyzing results"):
                plot_confusion_matrix(df, pred)

if __name__ == "__main__":
    main()
