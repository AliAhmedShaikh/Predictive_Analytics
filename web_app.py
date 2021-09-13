# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 18:00:29 2021

@author: DELL
"""

import streamlit as st
import datetime
import pandas as pd
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

sales_2020 = pd.read_csv('jalia20/sales_2020-11-23_2021-09-03.csv')

sales_grouped_products = sales_2020.groupby(['product_title']).count().reset_index()
sales_all_data_products = sales_grouped_products[sales_grouped_products['month']>9]['product_title'].values

sales_filtered_df = sales_2020[sales_2020['product_title']==sales_all_data_products[0]]
sales_filtered_products_df = []
for title in sales_all_data_products:
    sales_filtered_products_df.append(sales_2020[sales_2020['product_title']==title])
sales_filtered_product_df = pd.concat(sales_filtered_products_df, ignore_index=True)

sales_filtered_product_df = sales_filtered_product_df.drop(['product_title', 'year','shipping_region','hour_of_day','day','month','quarter'], axis=1)

sales_filtered_product_df.replace('\d\d\d\d-W',"",regex=True, inplace = True)

X_train, X_test, y_train, y_test = train_test_split(sales_filtered_product_df[['week','day_of_week']], sales_filtered_product_df['ordered_item_quantity'], test_size=0.2, random_state=23)


le_DoW = preprocessing.LabelEncoder()
le_SC = preprocessing.LabelEncoder()

le_DoW.fit(X_train['day_of_week'])
X_train['day_of_week']   = le_DoW.transform(X_train['day_of_week'])

X_test['day_of_week']    = le_DoW.transform(X_test['day_of_week'])

#fitting the polynomial regression model to the dataset
poly_reg=PolynomialFeatures(degree=2)
X_poly=poly_reg.fit_transform(X=X_train)
poly_reg.fit(X_poly,y_train)
lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y_train)
y_pred_poly_train = lin_reg2.predict(X_poly)

product_name = st.text_input("Product Name", "Marie Hachis parmentier 600g")

today = datetime.date.today()
day = st.date_input('day', today)
hour = st.text_input("Hour", "00")
city = st.text_input("City", "London")
region = st.text_input("Region", "England")
day_of_week = st.text_input("Day of Week", "Monday")
week_no = st.text_input("Week No", "22")

day_of_week = le_DoW.transform(day_of_week)
predict = [week_no,day_of_week]
X_pred = poly_reg.transform(predict)
prediction = lin_reg2.predict(X_pred)

pred = "The prediction is: " + str(prediction)
st.write(pred)