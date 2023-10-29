# -*- coding: utf-8 -*-
import pymongo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from pymongo import MongoClient
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


client = MongoClient("mongodb://localhost:27017/")
database = client["local"]
columnOrder = database["CoffeBean"]
columnCostumers = database["CoffeBeanCustomers"]
columnProducts = database["CoffeBeanProducts"]
 
dataOrders = pd.DataFrame(list(columnOrder.find()))
dataOrders = dataOrders.drop("_id",axis=1)

dataCustomers = pd.DataFrame(list(columnCostumers.find()))
dataCustomers = dataCustomers.drop("_id", axis=1)
dataCustomers = dataCustomers.drop("Postcode", axis=1)

dataProducts = pd.DataFrame(list(columnProducts.find()))
dataProducts = dataProducts.drop("_id", axis=1)


#Customers Communication Informations
dataCustomers.head()

customersWithoutEmail = []
customersWithoutPhone = []
customersNoComm = []
customersFullComm = []

for index, row in dataCustomers.iterrows():
    email_value = row['Email']
    phone_value = row['Phone Number']
    name = row['Customer Name']
    
    if pd.isnull(email_value) and not pd.isnull(phone_value):
        customer_info = {
            'Name': name,
            'Phone Number': phone_value
        }
        customersWithoutEmail.append(customer_info)
    if pd.isnull(phone_value) and not pd.isnull(email_value):
        customer_info = {
            'Name': name,
            'Email': email_value
        }
        customersWithoutPhone.append(customer_info)
    if pd.isnull(email_value) and pd.isnull(phone_value):
        customer_info = {
            'Name': name,
        }
        customersNoComm.append(customer_info)
    if not pd.isnull(email_value) and not pd.isnull(phone_value):
        customer_info = {
            'Name': name,
            'Phone Number': phone_value,
            'Email': email_value
        }
        customersFullComm.append(customer_info)

#-------------------------------------------------------------------


#Preprocessing for Total Sales Predictions
product_order_counts = dataOrders.groupby('Product ID')['Order ID'].count().reset_index()
total_order_counts = product_order_counts.groupby('Product ID')['Order ID'].sum().reset_index()

product_counts = dataOrders['Product ID'].value_counts().reset_index()
product_counts.columns = ['Product ID', 'Sales Count']

product_prices = dataProducts[['Product ID', 'Unit Price']]

merged_data = product_counts.merge(product_prices, on='Product ID', how='left')
merged_data['Unit Price'] = merged_data['Unit Price'].str.replace(',', '.').astype(float)
merged_data['Total Sales'] = merged_data['Sales Count'] * merged_data['Unit Price']
merged_data['Total Sales'] = merged_data['Total Sales'].astype(float)
merged_data['Total Sales'] = merged_data['Total Sales'].map('{:,.2f}'.format)

merged_data = merged_data.set_index('Product ID')

#Model

X = merged_data[['Sales Count' , 'Unit Price']]
Y = merged_data['Total Sales']


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


regressor = LinearRegression()
regressor.fit(x_train,y_train)


cross_val_scores = cross_val_score(regressor, X, Y, cv=5)
average_r2 = cross_val_scores.mean()


y_pred = regressor.predict(x_test)
print('Linear Regressor R Squared ' , regressor.score(x_test,y_test))
print(f"Ortalama R-squared değeri: {average_r2:.2f}")


#------------------------------------------------------------------


