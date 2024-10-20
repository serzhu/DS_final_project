# DS_final_project
### Description:
Application predicts potentialy lost clients on data from the billing service.  
Data shuld be in CSV format and contains next features:
```text
'id' - client ID
'is_tv_subscriber' -  has client IPTV subscription
'is_movie_package_subscriber' - has client VOD subscription
'subscription_age' - how long client use service
'bill_avg' - average bill amount for 3 month
'reamining_contract' - time in years to end of contract
'service_failure_count' - number of service failures
'download_avg' - input traffic in GB for 1 month
'upload_avg' - output traffic in GB for 1 month
'download_over_limit' - input traffic in GB over the limit
```
### Run app from source:
```bash
streamlit run app/app.py
```
### Run app from Docker container:
```bash
docker build  -t lost-clients-prediction . 
docker run -p 8501:8501 lost-clients-prediction
```
### Open in browser:
### <http://localhost:8501>

### Select test data in CSV format:  
test_data_100_clients.csv
### Select model from avaliable models:  
LR_model_Serhii.pkl  
LSVC_model_Serhii.pkl  
SVC_rbf_model_Serhii.pkl  
RF_model_Serhii.pkl  
NN_model_Serhii.pkl  
### Select scaler for current model:  
scaler_Serhii.pkl  
NN_scaler_Serhii.pkl  

<b>App will return list of IDs potentially lost clients  
<b>List of clients can be downloaded in CSV format