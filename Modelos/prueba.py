# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 11:11:21 2023

@author: Sergio
"""

import streamlit as st
import pandas as pd
import altair as alt
from PIL import Image

import json
import logging
import numpy as np

from urllib.error import URLError

import sys
sys.path.append('C:/Users/Sergio/Documents/valencia-ia4covid-xprize-master/valencia-ia4covid-xprize-master')

logo = 'https://ellisalicante.org/assets/xprize/images/logo_oscuro.png'
carpeta = 'C:/Users/Sergio/Documents/valencia-ia4covid-xprize-master/valencia-ia4covid-xprize-master/'

st.set_page_config(layout = 'wide')

@st.cache
def get_UN_data():
    paises = pd.read_csv("countries_regions.csv")
    return paises.set_index("CountryName")

def get_data_rule(DATA):
    x1 = DATA[0]
    date1 = DATA.index[0]
    xs=[x1]
    dates=[date1]
    for i in range(1, len(data)):
        c = DATA[i]
        if c!= x1:
            x1 = c
            xs.append(x1)
            dates.append(DATA.index[i])
    dates.append(DATA.index[-1])        
    return xs,dates


try:
    col1, col2, col3 = st.columns([1,2,1])

    with col1:
        st.write("")
        
    with col2:
        st.image(logo)
    
    with col3:
       st.write("")

    st.markdown('# Visualización de los modelos sum')
    
    
    paises = get_UN_data()
    data = pd.read_csv("data/OxCGRT_latest.csv")
    data['Date'] = pd.to_datetime(data['Date'], format = '%Y%m%d')    

####################################################
####################################################
    
    
    st.markdown('# Visualización de los modelos nuevos')
    paises = get_UN_data()
    data = pd.read_csv("data/OxCGRT_latest.csv")
    data['Date'] = pd.to_datetime(data['Date'], format = '%Y%m%d')
        
    cols = st.columns((1,5))
    with cols[0]:
        paises_list = list(paises.index.unique())
        paises_list.insert(0, "Europe")
        paises_list.insert(0, "Overall")
        country2 = st.selectbox(
            "Choose countries ",paises_list
        )
        
        months_list = ["January","February","March","April","May","June","July"]
        months_dates = ["2020-12-31","2021-01-31","2021-01-31","2021-02-28","2021-02-28","2021-03-31","2021-03-31","2021-04-30",
                        "2021-04-30","2021-05-31","2021-05-31","2021-06-30","2021-06-30","2021-07-31",]
        months_list_short = ["jan","feb","mar","apr","may","jun","jul"]

        month = st.selectbox('Choose a month   ', months_list)
       
        H7_SUS_2_bool = st.checkbox("H7_SUS_1")
        H7_SUS_3_bool = st.checkbox("H7_SUS_2")
        H7_SUS_4_bool = st.checkbox("H7_SUS_3")
        smooth_bool = st.checkbox("Smooth")
        
        
    new_data = {}
    overall_data = {}
    europe_data = {}
   
    start_date = pd.to_datetime(months_dates[months_list.index(month)*2],format = '%Y-%m-%d')
    
    if smooth_bool:
        # Añadido el -7 para que empiece 7 dias antes [NEW]
        start_date = start_date - np.timedelta64(7,'D')
    end_date = pd.to_datetime(months_dates[months_list.index(month)*2+1],format = '%Y-%m-%d')
    
    
    
    test = pd.read_csv("data/OxCGRT_latest.csv",parse_dates=["Date"])#parse
    #pd.to_datetime(test["Date"], unit='d')
    paises_utiles= test[test["CountryName"].isin(list(["Albania","Andorra","Austria","Azerbaijan","Belarus","Belgium","Bosnia and Herzegovina","Bulgaria","Croatia","Cyprus","Czech Republic",
                   "Denmark","Estonia","Finland","France","Georgia","Germany","Greece","Hungary","Iceland","Ireland","Italy","Kazakhstan","Latvia","Liechtenstein",
                   "Lithuania","Luxembourg","Malta","Moldova","Monaco","Netherlands","Norway","Poland","Portugal","Romania","San Marino","Serbia","Slovakia","Slovenia",
                   "Spain","Sweden","Switzerland","Turkey","Ukraine","United Kingdom"]))]
    
    if smooth_bool:
        #[NEW]
        # Cogemos los datos de la semana anterior al mes que queremos
        if months_list_short[months_list.index(month)] == "jan":
            star_day = "2020-12-24"
            star_day = pd.to_datetime(star_day, format = '%Y-%m-%d')
            end_day = "2020-12-31"
            end_day = pd.to_datetime(end_day, format = '%Y-%m-%d')
            data = pd.read_csv("data/OxCGRT_latest.csv")
            data['Date'] = pd.to_datetime(data['Date'], format = '%Y%m%d')
            semana_anterior = data[(data.Date >= star_day - np.timedelta64(1, 'D')) & (data.Date <= end_day + np.timedelta64(1, 'D'))]
            semana_anterior["ConfirmedCases"] = semana_anterior["ConfirmedCases"].diff()
            semana_anterior = semana_anterior[(semana_anterior.Date >= star_day) & (semana_anterior.Date <=end_day)]
            st.write(semana_anterior)
        else:            
            end_day = "2021-0"+str(months_list.index(month)+1)+"-01" # 2021-0+"mes"+
            st.write(end_day)
            end_day = pd.to_datetime(end_day, format = '%Y-%m-%d')
            star_day = end_day - np.timedelta64(7, 'D')
            st.write("hola")
            star_day = pd.to_datetime(star_day, format = '%Y-%m-%d')
            st.write("hola")
            data = pd.read_csv("data/OxCGRT_latest.csv")
            st.write("hola")
            data['Date'] = pd.to_datetime(data['Date'], format = '%Y%m%d')
            semana_anterior = data[(data.Date >= star_day - np.timedelta64(1, 'D')) & (data.Date <= end_day + np.timedelta64(1, 'D'))]
            st.write("hola")
            semana_anterior["ConfirmedCases"] = semana_anterior["ConfirmedCases"].diff()
            semana_anterior = semana_anterior[(semana_anterior.Date >= star_day) & (semana_anterior.Date <= end_day)]
            st.write("hola")
                
        # Lo mismo para los datos de la semana anterior[NEW]
        paises_utiles_semana_anterior = semana_anterior[semana_anterior["CountryName"].isin(list(["Albania","Andorra","Austria","Azerbaijan","Belarus","Belgium","Bosnia and Herzegovina","Bulgaria","Croatia","Cyprus","Czech Republic",
                     "Denmark","Estonia","Finland","France","Georgia","Germany","Greece","Hungary","Iceland","Ireland","Italy","Kazakhstan","Latvia","Liechtenstein",
                     "Lithuania","Luxembourg","Malta","Moldova","Monaco","Netherlands","Norway","Poland","Portugal","Romania","San Marino","Serbia","Slovakia","Slovenia",
                     "Spain","Sweden","Switzerland","Turkey","Ukraine","United Kingdom"]))]
        paises_utiles = paises_utiles.append(paises_utiles_semana_anterior) #[NEW]
        st.write("hola")
    
    suma1 =paises_utiles[["Date","CountryName","ConfirmedCases"]].fillna(0).groupby("Date").sum().reset_index()
    
    #data= data[(data.Date >= start_date)&(data.Date <= end_date)&(data.CountryName == country)]
    suma1 =suma1[(data.Date >= start_date)&(data.Date <= end_date)]
    data= data[(data.Date >= start_date)&(data.Date <= end_date)]
                           
    overall_data = data.groupby(["Date"])[["ConfirmedCases"]].sum().sort_values(by=["Date"])
    #europe_data =paises_utiles[["Date","CountryName","ConfirmedCases"]].fillna(0).groupby("Date").sum()
    data = data.set_index("Date")
    data = data[data.CountryName == country2]
    overall_data["ConfirmedCases"] = overall_data["ConfirmedCases"].diff()
    if smooth_bool:
        # Ahora hacemos la media a 7 dias de los datos de la semana anterior [NEW]
        suma1["ConfirmedCases"] = suma1["ConfirmedCases"].rolling(7).mean()
        # Ahora hacemos la media a 7 dias de los datos de la semana anterior [NEW]
        overall_data["ConfirmedCases"] = overall_data["ConfirmedCases"].rolling(7).mean()
    #europe_data["ConfirmedCases"] = europe_data["ConfirmedCases"].diff()
    new_data["Ground_truth"] = data["ConfirmedCases"].diff()
    import plotly.graph_objects as go
    fig = go.Figure()
    
    if H7_SUS_2_bool:
        overall_H7_SUS_2_data={}
        H7_SUS_2_data = pd.read_csv("Modelos/predicciones_ahmed/robojudge_test_H7_SUS_"
                              +months_list_short[months_list.index(month)] + ".csv")
        H7_SUS_2_data['Date'] = pd.to_datetime(H7_SUS_2_data['Date'], format = '%Y-%m-%d')
        
        paises_utiles_H7_SUS_2= H7_SUS_2_data[H7_SUS_2_data["CountryName"].isin(list(["Albania","Andorra","Austria","Azerbaijan","Belarus","Belgium","Bosnia and Herzegovina","Bulgaria","Croatia","Cyprus","Czech Republic",
                   "Denmark","Estonia","Finland","France","Georgia","Germany","Greece","Hungary","Iceland","Ireland","Italy","Kazakhstan","Latvia","Liechtenstein",
                   "Lithuania","Luxembourg","Malta","Moldova","Monaco","Netherlands","Norway","Poland","Portugal","Romania","San Marino","Serbia","Slovakia","Slovenia",
                   "Spain","Sweden","Switzerland","Turkey","Ukraine","United Kingdom"]))]
        if smooth_bool:
            #[NEW]
            # Cogemos los datos de la semana anterior al mes que queremos
            if months_list_short[months_list.index(month)] == "jan":
                star_day = "2020-12-24"
                star_day = pd.to_datetime(star_day, format = '%Y-%m-%d')
                end_day = "2020-12-31"
                end_day = pd.to_datetime(end_day, format = '%Y-%m-%d')
                data = pd.read_csv("data/OxCGRT_latest.csv")
                data['Date'] = pd.to_datetime(data['Date'], format = '%Y%m%d')
                semana_anterior = data[(data.Date >= start_date - np.timedelta64(1, 'D')) & (data.Date <= end_date + np.timedelta64(1, 'D'))]
                semana_anterior["ConfirmedCases"] = semana_anterior["ConfirmedCases"].diff()
                semana_anterior = semana_anterior[(semana_anterior.Date >= start_date) & (semana_anterior.Date <= end_date)]
                st.write(semana_anterior)
            else:            
                end_date = "2021-0"+str(months_list_short.index(months_list_short[months_list.index(month)])+1)+"-01" # 2021-0+"mes"+
                end_date = pd.to_datetime(end_date, format = '%Y-%m-%d')
                start_date = end_date - np.timedelta64(7, 'D')
                start_date = pd.to_datetime(start_date, format = '%Y-%m-%d')
                data = pd.read_csv("data/OxCGRT_latest.csv")
                data['Date'] = pd.to_datetime(data['Date'], format = '%Y%m%d')
                semana_anterior = data[(data.Date >= start_date - np.timedelta64(1, 'D')) & (data.Date <= end_date + np.timedelta64(1, 'D'))]
                semana_anterior["ConfirmedCases"] = semana_anterior["ConfirmedCases"].diff()
                semana_anterior = semana_anterior[(semana_anterior.Date >= start_date) & (semana_anterior.Date <= end_date)]
                
            semana_anterior["PredictedDailyNewCases"] = semana_anterior["ConfirmedCases"]
            # Lo mismo para los datos de la semana anterior[NEW]
            paises_utiles_semana_anterior = semana_anterior[semana_anterior["CountryName"].isin(list(["Albania","Andorra","Austria","Azerbaijan","Belarus","Belgium","Bosnia and Herzegovina","Bulgaria","Croatia","Cyprus","Czech Republic",
                     "Denmark","Estonia","Finland","France","Georgia","Germany","Greece","Hungary","Iceland","Ireland","Italy","Kazakhstan","Latvia","Liechtenstein",
                     "Lithuania","Luxembourg","Malta","Moldova","Monaco","Netherlands","Norway","Poland","Portugal","Romania","San Marino","Serbia","Slovakia","Slovenia",
                     "Spain","Sweden","Switzerland","Turkey","Ukraine","United Kingdom"]))]
            paises_utiles_H7_SUS_3 = paises_utiles_H7_SUS_3.append(paises_utiles_semana_anterior) #[NEW]
        
        suma1_H7_SUS_2 =paises_utiles_H7_SUS_2[["Date","CountryName","PredictedDailyNewCases"]].fillna(0).groupby("Date").sum()
        
        if smooth_bool:
             Ahora hacemos la media a 7 dias de los datos de la semana anterior [NEW]
            suma1_H7_SUS_2["PredictedDailyNewCases"] = suma1_H7_SUS_2["PredictedDailyNewCases"].rolling(7).mean()
            H7_SUS_2_data["PredictedDailyNewCases"] = H7_SUS_2_data["PredictedDailyNewCases"].rolling(7).mean()
        
        #suma1_H7_SUS["PredictedDailyNewCases"] = suma1_H7_SUS["PredictedDailyNewCases"].diff()
        fig.add_trace(go.Scatter(x = suma1_H7_SUS_2.index, y = suma1_H7_SUS_2.PredictedDailyNewCases,name = "H7_SUS_1",visible = (country2 == "Europe"),line=dict(color="purple")))
        
        overall_H7_SUS_2_data= H7_SUS_2_data.groupby(["Date"])[["PredictedDailyNewCases"]].sum().sort_values(by=["Date"])
        H7_SUS_2_data = H7_SUS_2_data[H7_SUS_2_data.CountryName == country2]
        H7_SUS_2_data = H7_SUS_2_data.set_index("Date")
        new_data["H7_SUS_2"] = H7_SUS_2_data["PredictedDailyNewCases"]
        new_data = pd.DataFrame(new_data)
        fig.add_trace(go.Scatter(x = new_data.index, y = new_data.H7_SUS_2,name = "H7_SUS_1",line=dict(color='blue')))
        fig.add_trace(go.Scatter(x = overall_H7_SUS_2_data.index, y = overall_H7_SUS_2_data.PredictedDailyNewCases,name = "H7_SUS_1",visible = (country2 == "Overall"),line=dict(color='purple')))
    
    
    if H7_SUS_3_bool:
        overall_H7_SUS_3_data={}
        H7_SUS_3_data = pd.read_csv("Modelos/predicciones_oscar/robojudge_test_H7_SUS_"
                              +months_list_short[months_list.index(month)] + ".csv")
        H7_SUS_3_data['Date'] = pd.to_datetime(H7_SUS_3_data['Date'], format = '%Y-%m-%d')
        
        paises_utiles_H7_SUS_3= H7_SUS_3_data[H7_SUS_3_data["CountryName"].isin(list(["Albania","Andorra","Austria","Azerbaijan","Belarus","Belgium","Bosnia and Herzegovina","Bulgaria","Croatia","Cyprus","Czech Republic",
                   "Denmark","Estonia","Finland","France","Georgia","Germany","Greece","Hungary","Iceland","Ireland","Italy","Kazakhstan","Latvia","Liechtenstein",
                   "Lithuania","Luxembourg","Malta","Moldova","Monaco","Netherlands","Norway","Poland","Portugal","Romania","San Marino","Serbia","Slovakia","Slovenia",
                   "Spain","Sweden","Switzerland","Turkey","Ukraine","United Kingdom"]))]
        
        if smooth_bool:
            #[NEW]
            # Cogemos los datos de la semana anterior al mes que queremos
            if months_list_short[months_list.index(month)] == "jan":
                star_day = "2020-12-24"
                star_day = pd.to_datetime(star_day, format = '%Y-%m-%d')
                end_day = "2020-12-31"
                end_day = pd.to_datetime(end_day, format = '%Y-%m-%d')
                data = pd.read_csv("data/OxCGRT_latest.csv")
                data['Date'] = pd.to_datetime(data['Date'], format = '%Y%m%d')
                semana_anterior = data[(data.Date >= start_date - np.timedelta64(1, 'D')) & (data.Date <= end_date + np.timedelta64(1, 'D'))]
                semana_anterior["ConfirmedCases"] = semana_anterior["ConfirmedCases"].diff()
                semana_anterior = semana_anterior[(semana_anterior.Date >= start_date) & (semana_anterior.Date <= end_date)]
                st.write(semana_anterior)
            else:            
                end_date = "2021-0"+str(months_list_short.index(months_list_short[months_list.index(month)])+1)+"-01" # 2021-0+"mes"+
                end_date = pd.to_datetime(end_date, format = '%Y-%m-%d')
                start_date = end_date - np.timedelta64(7, 'D')
                start_date = pd.to_datetime(start_date, format = '%Y-%m-%d')
                data = pd.read_csv("data/OxCGRT_latest.csv")
                data['Date'] = pd.to_datetime(data['Date'], format = '%Y%m%d')
                semana_anterior = data[(data.Date >= start_date - np.timedelta64(1, 'D')) & (data.Date <= end_date + np.timedelta64(1, 'D'))]
                semana_anterior["ConfirmedCases"] = semana_anterior["ConfirmedCases"].diff()
                semana_anterior = semana_anterior[(semana_anterior.Date >= start_date) & (semana_anterior.Date <= end_date)]
                
            semana_anterior["PredictedDailyNewCases"] = semana_anterior["ConfirmedCases"]
            # Lo mismo para los datos de la semana anterior[NEW]
            paises_utiles_semana_anterior = semana_anterior[semana_anterior["CountryName"].isin(list(["Albania","Andorra","Austria","Azerbaijan","Belarus","Belgium","Bosnia and Herzegovina","Bulgaria","Croatia","Cyprus","Czech Republic",
                     "Denmark","Estonia","Finland","France","Georgia","Germany","Greece","Hungary","Iceland","Ireland","Italy","Kazakhstan","Latvia","Liechtenstein",
                     "Lithuania","Luxembourg","Malta","Moldova","Monaco","Netherlands","Norway","Poland","Portugal","Romania","San Marino","Serbia","Slovakia","Slovenia",
                     "Spain","Sweden","Switzerland","Turkey","Ukraine","United Kingdom"]))]
            paises_utiles_H7_SUS_3 = paises_utiles_H7_SUS_3.append(paises_utiles_semana_anterior) #[NEW]
        
        suma1_H7_SUS_3 =paises_utiles_H7_SUS_3[["Date","CountryName","PredictedDailyNewCases"]].fillna(0).groupby("Date").sum()
        if smooth_bool:
            # Ahora hacemos la media a 7 dias de los datos de la semana anterior [NEW]
            suma1_H7_SUS_3["PredictedDailyNewCases"] = suma1_H7_SUS_3["PredictedDailyNewCases"].rolling(7).mean()
            H7_SUS_3_data["PredictedDailyNewCases"] = H7_SUS_3_data["PredictedDailyNewCases"].rolling(7).mean()
        #suma1_H7_SUS["PredictedDailyNewCases"] = suma1_H7_SUS["PredictedDailyNewCases"].diff()
        fig.add_trace(go.Scatter(x = suma1_H7_SUS_3.index, y = suma1_H7_SUS_3.PredictedDailyNewCases,name = "H7_SUS_2",visible = (country2 == "Europe"),line=dict(color='green')))
        
        overall_H7_SUS_3_data= H7_SUS_3_data.groupby(["Date"])[["PredictedDailyNewCases"]].sum().sort_values(by=["Date"])
        H7_SUS_3_data = H7_SUS_3_data[H7_SUS_3_data.CountryName == country2]
        H7_SUS_3_data = H7_SUS_3_data.set_index("Date")
        new_data["H7_SUS_3"] = H7_SUS_3_data["PredictedDailyNewCases"]
        new_data = pd.DataFrame(new_data)
        fig.add_trace(go.Scatter(x = new_data.index, y = new_data.H7_SUS_3,name = "H7_SUS_2",line=dict(color='green')))
        fig.add_trace(go.Scatter(x = overall_H7_SUS_3_data.index, y = overall_H7_SUS_3_data.PredictedDailyNewCases,name = "H7_SUS_2",visible = (country2 == "Overall"),line=dict(color='green')))
    
        
    if H7_SUS_4_bool:
        overall_H7_SUS_4_data={}
        H7_SUS_4_data = pd.read_csv("Modelos/07_03_23/robojudge_test_H7_SUS_"
                              +months_list_short[months_list.index(month)] + ".csv")
        H7_SUS_4_data['Date'] = pd.to_datetime(H7_SUS_4_data['Date'], format = '%Y-%m-%d')
        
        paises_utiles_H7_SUS_4= H7_SUS_4_data[H7_SUS_4_data["CountryName"].isin(list(["Albania","Andorra","Austria","Azerbaijan","Belarus","Belgium","Bosnia and Herzegovina","Bulgaria","Croatia","Cyprus","Czech Republic",
                   "Denmark","Estonia","Finland","France","Georgia","Germany","Greece","Hungary","Iceland","Ireland","Italy","Kazakhstan","Latvia","Liechtenstein",
                   "Lithuania","Luxembourg","Malta","Moldova","Monaco","Netherlands","Norway","Poland","Portugal","Romania","San Marino","Serbia","Slovakia","Slovenia",
                   "Spain","Sweden","Switzerland","Turkey","Ukraine","United Kingdom"]))]
        
        if smooth_bool:
            #[NEW]
            # Cogemos los datos de la semana anterior al mes que queremos
            if months_list_short[months_list.index(month)] == "jan":
                star_day = "2020-12-24"
                star_day = pd.to_datetime(star_day, format = '%Y-%m-%d')
                end_day = "2020-12-31"
                end_day = pd.to_datetime(end_day, format = '%Y-%m-%d')
                data = pd.read_csv("data/OxCGRT_latest.csv")
                data['Date'] = pd.to_datetime(data['Date'], format = '%Y%m%d')
                semana_anterior = data[(data.Date >= star_day - np.timedelta64(1, 'D')) & (data.Date <= end_day + np.timedelta64(1, 'D'))]
                semana_anterior["ConfirmedCases"] = semana_anterior["ConfirmedCases"].diff()
                semana_anterior = semana_anterior[(semana_anterior.Date >= star_day) & (semana_anterior.Date <=end_day)]
                st.write(semana_anterior)
            else:            
                end_day = "2021-0"+str(months_list.index(month)+1)+"-01" # 2021-0+"mes"+
                st.write(end_day)
                end_day = pd.to_datetime(end_day, format = '%Y-%m-%d')
                star_day = end_day - np.timedelta64(7, 'D')
                star_day = pd.to_datetime(star_day, format = '%Y-%m-%d')
                data = pd.read_csv("data/OxCGRT_latest.csv")
                data['Date'] = pd.to_datetime(data['Date'], format = '%Y%m%d')
                semana_anterior = data[(data.Date >= star_day - np.timedelta64(1, 'D')) & (data.Date <= end_day + np.timedelta64(1, 'D'))]
                semana_anterior["ConfirmedCases"] = semana_anterior["ConfirmedCases"].diff()
                semana_anterior = semana_anterior[(semana_anterior.Date >= star_day) & (semana_anterior.Date <= end_day)]
                
            st.write(star_day)
            st.write(end_day)
                
            semana_anterior["PredictedDailyNewCases"] = semana_anterior["ConfirmedCases"]
            # Lo mismo para los datos de la semana anterior[NEW]
            paises_utiles_semana_anterior = semana_anterior[semana_anterior["CountryName"].isin(list(["Albania","Andorra","Austria","Azerbaijan","Belarus","Belgium","Bosnia and Herzegovina","Bulgaria","Croatia","Cyprus","Czech Republic",
                     "Denmark","Estonia","Finland","France","Georgia","Germany","Greece","Hungary","Iceland","Ireland","Italy","Kazakhstan","Latvia","Liechtenstein",
                     "Lithuania","Luxembourg","Malta","Moldova","Monaco","Netherlands","Norway","Poland","Portugal","Romania","San Marino","Serbia","Slovakia","Slovenia",
                     "Spain","Sweden","Switzerland","Turkey","Ukraine","United Kingdom"]))]
            paises_utiles_H7_SUS_4 = paises_utiles_H7_SUS_4.append(paises_utiles_semana_anterior) #[NEW]
        
        suma1_H7_SUS_4 =paises_utiles_H7_SUS_4[["Date","CountryName","PredictedDailyNewCases"]].fillna(0).groupby("Date").sum()
        if smooth_bool:
            # Ahora hacemos la media a 7 dias de los datos de la semana anterior [NEW]
            suma1_H7_SUS_4["PredictedDailyNewCases"] = suma1_H7_SUS_4["PredictedDailyNewCases"].rolling(7).mean()
            H7_SUS_4_data["PredictedDailyNewCases"] = H7_SUS_4_data["PredictedDailyNewCases"].rolling(7).mean()
        #suma1_H7_SUS["PredictedDailyNewCases"] = suma1_H7_SUS["PredictedDailyNewCases"].diff()
        fig.add_trace(go.Scatter(x = suma1_H7_SUS_4.index, y = suma1_H7_SUS_4.PredictedDailyNewCases,name = "H7_SUS_3",visible = (country2 == "Europe"),line=dict(color='black')))
        
        overall_H7_SUS_4_data= H7_SUS_4_data.groupby(["Date"])[["PredictedDailyNewCases"]].sum().sort_values(by=["Date"])
        H7_SUS_4_data = H7_SUS_4_data[H7_SUS_4_data.CountryName == country2]
        H7_SUS_4_data = H7_SUS_4_data.set_index("Date")
        new_data["H7_SUS_4"] = H7_SUS_4_data["PredictedDailyNewCases"]
        new_data = pd.DataFrame(new_data)
        fig.add_trace(go.Scatter(x = new_data.index, y = new_data.H7_SUS_4,name = "H7_SUS_3",line=dict(color='black')))
        fig.add_trace(go.Scatter(x = overall_H7_SUS_4_data.index, y = overall_H7_SUS_4_data.PredictedDailyNewCases,name = "H7_SUS_3",visible = (country2 == "Overall"),line=dict(color='black')))
    
   
    
    with cols[1]:
        new_data = pd.DataFrame(new_data)
        #overall_data = pd.DataFrame(overall_data)
        #overall_data = overall_data.set_index("Date")
        fig.add_trace(go.Scatter(x = suma1.Date, y = suma1.ConfirmedCases,name = "Ground truth",visible = (country2 == "Europe"),line=dict(color='orange', width=4, dash='dash')))
        #fig.add_trace(go.Scatter(x = new_data.index, y = new_data.Ground_truth,name = "Ground truth",line=dict(color='orange', width=4, dash='dash')))
        fig.add_trace(go.Scatter(x = overall_data.index, y = overall_data.ConfirmedCases,name = "Ground truth",visible = (country2 == "Overall"),line=dict(color='orange', width=4, dash='dash')))
        #new_data = new_data.set_index("Date")
        fig.update_layout(
      margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(figure_or_data=fig,use_container_width=True)
        
    st.text("Modelo 1: Los mejores modelos de el entrenamiento que se hizo con el set de paises inicial y 100 iteraciones para cada mes" )
    st.text("Modelo 2: Los mejores modelos de el entrenamiento que se hizo con el set de paises nuevo (Europa con más de 2M y América con más de 5M) y 100 iteraciones" )
        
        
        
except URLError as e:
    st.error(
        """
            
        **This demo requires internet access.**

        Connection error: %s
        """

        % e.reason
    )
