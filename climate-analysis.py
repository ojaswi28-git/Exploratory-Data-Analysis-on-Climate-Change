#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import math
import fire

data = pd.read_csv("data/GlobalLandTemperaturesByCity.csv")
data.sort_values(by=['dt'])
#us_data = data[data['Country'] == 'United States']
#la_data = us_data[us_data['City'] == 'Los Angeles']

def get_yearly_avg():
    current_year = data['dt'].iloc[0][:4]
    prev_year = current_year
    monthly_avg_temp = 0
    possible_error = 0
    yearly_avg_temp = []
    possible_error_pos = []
    possible_error_neg = []
    num_months = 0
    year_list = []
        
    for index, month in data.iterrows(): 
        num_months += 1
        current_year = month['dt'][:4]
        if current_year > prev_year:
            year_avg_temp = monthly_avg_temp / num_months
            yearly_avg_temp.append(year_avg_temp)
            possible_error = possible_error / num_months
            possible_error_pos.append(year_avg_temp+possible_error)
            possible_error_neg.append(year_avg_temp-possible_error)
            monthly_avg_temp = 0
            possible_error = 0
            year_list.append(prev_year)
            prev_year = current_year
            num_months = 1     
        monthly_avg_temp += month['AverageTemperature']
        possible_error += month['AverageTemperatureUncertainty']
    return (year_list, yearly_avg_temp, possible_error_pos, possible_error_neg)

def get_yearly_avg_city(city, country):
    if country not in data['Country'].unique():
        raise Exception('Country not found in data set!')
    dataset = data[data['Country'] == country]
    if city not in dataset['City'].unique():
        raise Exception('City not found in data set!')
    dataset = dataset[dataset['City'] == city]
    current_year = dataset['dt'].iloc[0][:4]
    prev_year = current_year
    monthly_avg_temp = 0
    possible_error = 0
    yearly_avg_temp = []
    possible_error_pos = []
    possible_error_neg = []
    num_months = 0
    year_list = []
        
    for index, month in dataset.iterrows(): 
        num_months += 1
        current_year = month['dt'][:4]
        if current_year > prev_year:
            year_avg_temp = monthly_avg_temp / num_months
            yearly_avg_temp.append(year_avg_temp)
            possible_error = possible_error / num_months
            possible_error_pos.append(year_avg_temp+possible_error)
            possible_error_neg.append(year_avg_temp-possible_error)
            monthly_avg_temp = 0
            possible_error = 0
            year_list.append(prev_year)
            prev_year = current_year
            num_months = 1     
        monthly_avg_temp += month['AverageTemperature']
        possible_error += month['AverageTemperatureUncertainty']
    return (year_list, yearly_avg_temp, possible_error_pos, possible_error_neg)

def get_yearly_avg_country(country):
    if country not in data['Country'].unique():
        raise Exception('Country not found in data set!')
    dataset = data[data['Country'] == country]
    current_year = dataset['dt'].iloc[0][:4]
    prev_year = current_year
    monthly_avg_temp = 0
    possible_error = 0
    yearly_avg_temp = []
    possible_error_pos = []
    possible_error_neg = []
    num_months = 0
    year_list = []
        
    for index, month in dataset.iterrows(): 
        num_months += 1
        current_year = month['dt'][:4]
        if current_year > prev_year:
            year_avg_temp = monthly_avg_temp / num_months
            yearly_avg_temp.append(year_avg_temp)
            possible_error = possible_error / num_months
            possible_error_pos.append(year_avg_temp+possible_error)
            possible_error_neg.append(year_avg_temp-possible_error)
            monthly_avg_temp = 0
            possible_error = 0
            year_list.append(prev_year)
            prev_year = current_year
            num_months = 1     
        monthly_avg_temp += month['AverageTemperature']
        possible_error += month['AverageTemperatureUncertainty']
    return (year_list, yearly_avg_temp, possible_error_pos, possible_error_neg)


def make_graph_city(city, country):
    x, y, error1, error2 = get_yearly_avg_city(city, country)
    x, y, error1, error2 = remove_gaps(x, y, error1, error2)
    fig, ax = plt.subplots(figsize=(10,4))
    
    #Set the graph
    ax.set_xlabel('Year')
    ax.set_ylabel('Average Temperature(°C)')
    ax.set_title('Yearly Average Temperatures in ' + city +', ' + country)
    ax.set_xticklabels(x, rotation = 90)
    ax.plot(x, y, label="Average Temperature")
    ax.fill_between(x, error1, error2,  facecolor='blue', alpha=0.1, label = "Margin of Error")  
    
    #Compute linear regression to show relationship of temperature in respect to time
    X = np.array(x).astype(np.int64)[:, np.newaxis]
    reg = LinearRegression()
    reg.fit(X, np.array(y).astype(np.int64))
    y_pred = reg.predict(X)
    ax.plot(x, y_pred, label="Predicted Behavior of Temperature over Time\n(Linear Regression)")
    
    #Show the graph
    ax.legend(loc=4)
    plt.xticks(x[::10], x[::10])
    plt.show()
    plt.close()
    
def make_graph_country(country):
    x, y, error1, error2 = get_yearly_avg_country(country)
    x, y, error1, error2 = remove_gaps(x, y, error1, error2)
    
    #Set the graph
    fig, ax = plt.subplots(figsize=(10,4))
    ax.set_xlabel('Year')
    ax.set_ylabel('Average Temperature(°C)')
    ax.set_title('Yearly Average Temperatures in ' + country)
    ax.set_xticklabels(x, rotation = 90)
    ax.plot(x, y, label="Average Temperature")
    ax.fill_between(x, error1, error2,  facecolor='blue', alpha=0.1, label = "Margin of Error")
    
    #Compute linear regression to show relationship of temperature in respect to time
    X = np.array(x).astype(np.int64)[:, np.newaxis]
    reg = LinearRegression()
    reg.fit(X, np.array(y).astype(np.int64))
    y_pred = reg.predict(X)
    ax.plot(x, y_pred, label="Predicted Behavior of Temperature over Time\n(Linear Regression)")
    
    #Show the graph
    plt.xticks(x[::10], x[::10])
    ax.legend(loc=4)
    plt.show()
    
def make_graph_world():
    x, y, error1, error2 = get_yearly_avg()
    x, y, error1, error2 = remove_gaps(x, y, error1, error2)
    
    #Set the graph
    fig, ax = plt.subplots(figsize=(10,4))
    ax.set_xlabel('Year')
    ax.set_ylabel('Average Temperature(°C)')
    ax.set_title('World Yearly Average Temperatures')
    ax.set_xticklabels(x, rotation = 90)
    ax.plot(x, y, label="Average Temperature")
    ax.fill_between(x, error1, error2,  facecolor='blue', alpha=0.1, label = "Margin of Error")
    
    #Compute linear regression to show relationship of temperature in respect to time
    X = np.array(x).astype(np.int64)[:, np.newaxis]
    reg = LinearRegression()
    reg.fit(X, np.array(y).astype(np.int64))
    y_pred = reg.predict(X)
    ax.plot(x, y_pred, label="Predicted Behavior of Temperature over Time\n(Linear Regression)")
    
    #Show the graph
    plt.xticks(x[::10], x[::10])
    ax.legend(loc=4)
    plt.show()

#Remove gaps between years for the linear regression model to work.
#In this case, we start at the firt year where there is no gap in any years after it.
def remove_gaps(x, y, z1, z2):
    start = 0
    for i in range(len(x)):
        if math.isnan(y[i]):
            start = i+1
    return (x[start:], y[start:], z1[start:], z2[start:])

if __name__ == '__main__':
  fire.Fire()