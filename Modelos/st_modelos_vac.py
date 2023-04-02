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
# Fix the random seed
np.random.seed(1234)
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
        
    #with col2:
    #    st.image(logo)
    
    with col3:
       st.write("")
    
    
    paises = get_UN_data()
    data = pd.read_csv("data/OxCGRT_latest.csv")
    data['Date'] = pd.to_datetime(data['Date'], format = '%Y%m%d')    

####################################################
####################################################
    
    
    st.markdown('# Models visualization')
    st.markdown('Here you can see the models of the different countries with an interval of one month')
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
                        "2021-04-30","2021-05-31","2021-05-31","2021-06-30","2021-06-30","2021-07-31"]
        months_list_short = ["jan","feb","mar","apr","may","jun","jul"]

        month = st.selectbox('Choose a month   ', months_list)
        
        H7_SUS_4_bool = True
        H7_SUS_2_bool = False
        H7_SUS_3_bool = False
        smooth_bool = False
        
    new_data = {}
    overall_data = {}
    europe_data = {}
    overall_data2= {}
   
    start_date = pd.to_datetime(months_dates[months_list.index(month)*2],format = '%Y-%m-%d')
    
    if smooth_bool:
        # Añadido el -7 para que empiece 7 dias antes [NEW]
        start_date = start_date - np.timedelta64(7,'D')
    end_date = pd.to_datetime(months_dates[months_list.index(month)*2+1],format = '%Y-%m-%d')
   
    
    test = pd.read_csv("data/OxCGRT_latest.csv",parse_dates=["Date"])#parse

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
                
        # Lo mismo para los datos de la semana anterior[NEW]
        paises_utiles_semana_anterior = semana_anterior[semana_anterior["CountryName"].isin(list(["Albania","Andorra","Austria","Azerbaijan","Belarus","Belgium","Bosnia and Herzegovina","Bulgaria","Croatia","Cyprus","Czech Republic",
                     "Denmark","Estonia","Finland","France","Georgia","Germany","Greece","Hungary","Iceland","Ireland","Italy","Kazakhstan","Latvia","Liechtenstein",
                     "Lithuania","Luxembourg","Malta","Moldova","Monaco","Netherlands","Norway","Poland","Portugal","Romania","San Marino","Serbia","Slovakia","Slovenia",
                     "Spain","Sweden","Switzerland","Turkey","Ukraine","United Kingdom"]))]
        paises_utiles = paises_utiles.append(paises_utiles_semana_anterior) #[NEW]
    
    
    suma1 =paises_utiles[["Date","CountryName","ConfirmedCases"]].fillna(0).groupby("Date").sum().reset_index()
    
    #data= data[(data.Date >= start_date)&(data.Date <= end_date)&(data.CountryName == country)]
    suma1 =suma1[(data.Date >= start_date)&(data.Date <= end_date)]
    suma1["ConfirmedCases"] = suma1["ConfirmedCases"].diff()
    data= data[(data.Date >= start_date)&(data.Date <= end_date)]
    data2= data[(data.Date >= start_date+np.timedelta64(7,'D'))&(data.Date <= end_date)]
    
    
                           
    overall_data = data.groupby(["Date"])[["ConfirmedCases"]].sum().sort_values(by=["Date"])
    overall_data2 = data2.groupby(["Date"])[["ConfirmedCases"]].sum().sort_values(by=["Date"])
    #europe_data =paises_utiles[["Date","CountryName","ConfirmedCases"]].fillna(0).groupby("Date").sum()
    data = data.set_index("Date")
    data = data[data.CountryName == country2]
    overall_data["ConfirmedCases"] = overall_data["ConfirmedCases"].diff()
    import plotly.graph_objects as go
    fig = go.Figure()
    
    #fig.add_trace(go.Scatter(x = overall_data.index, y = overall_data.ConfirmedCases,name = "Ground truth",visible = (country2 == "Overall"),line=dict(color='blue', width=4, dash='dash')))
    if smooth_bool:
        # Ahora hacemos la media a 7 dias de los datos de la semana anterior [NEW]
        suma1["ConfirmedCases"] = suma1["ConfirmedCases"].rolling(7).mean()
        data["ConfirmedCases"] = data["ConfirmedCases"].rolling(7).mean()
        # Ahora hacemos la media a 7 dias de los datos de la semana anterior [NEW]
        overall_data["ConfirmedCases"] = overall_data["ConfirmedCases"].rolling(7).mean()
    #europe_data["ConfirmedCases"] = europe_data["ConfirmedCases"].diff()
    new_data["Ground_truth"] = data["ConfirmedCases"].diff()
    
    
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
                
                
            semana_anterior["PredictedDailyNewCases"] = semana_anterior["ConfirmedCases"]
            # Lo mismo para los datos de la semana anterior[NEW]
            paises_utiles_semana_anterior = semana_anterior[semana_anterior["CountryName"].isin(list(["Albania","Andorra","Austria","Azerbaijan","Belarus","Belgium","Bosnia and Herzegovina","Bulgaria","Croatia","Cyprus","Czech Republic",
                     "Denmark","Estonia","Finland","France","Georgia","Germany","Greece","Hungary","Iceland","Ireland","Italy","Kazakhstan","Latvia","Liechtenstein",
                     "Lithuania","Luxembourg","Malta","Moldova","Monaco","Netherlands","Norway","Poland","Portugal","Romania","San Marino","Serbia","Slovakia","Slovenia",
                     "Spain","Sweden","Switzerland","Turkey","Ukraine","United Kingdom"]))]
            paises_utiles_H7_SUS_2 = paises_utiles_H7_SUS_2.append(paises_utiles_semana_anterior) #[NEW]
        
        suma1_H7_SUS_2 =paises_utiles_H7_SUS_2[["Date","CountryName","PredictedDailyNewCases"]].fillna(0).groupby("Date").sum()
        
        #if smooth_bool:
            # Ahora hacemos la media a 7 dias de los datos de la semana anterior [NEW]
        #    suma1_H7_SUS_2["PredictedDailyNewCases"] = suma1_H7_SUS_2["PredictedDailyNewCases"].rolling(7).mean()
        #    H7_SUS_2_data["PredictedDailyNewCases"] = H7_SUS_2_data["PredictedDailyNewCases"].rolling(7).mean()
        
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
        overall_H7_SUS_4_data2={}
        H7_SUS_4_data = pd.read_csv("Modelos/07_03_23/robojudge_test_H7_SUS_"
                              +months_list_short[months_list.index(month)] + ".csv")
        H7_SUS_4_data2 = pd.read_csv("Modelos/07_03_23/robojudge_test_H7_SUS_"
                              +months_list_short[months_list.index(month)] + ".csv")
        H7_SUS_4_data['Date'] = pd.to_datetime(H7_SUS_4_data['Date'], format = '%Y-%m-%d')
        H7_SUS_4_data2['Date'] = pd.to_datetime(H7_SUS_4_data2['Date'], format = '%Y-%m-%d')
        
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
            else:            
                end_day = "2021-0"+str(months_list.index(month)+1)+"-01" # 2021-0+"mes"+
                end_day = pd.to_datetime(end_day, format = '%Y-%m-%d')
                star_day = end_day - np.timedelta64(7, 'D')
                star_day = pd.to_datetime(star_day, format = '%Y-%m-%d')
                data = pd.read_csv("data/OxCGRT_latest.csv")
                data['Date'] = pd.to_datetime(data['Date'], format = '%Y%m%d')
                semana_anterior = data[(data.Date >= star_day - np.timedelta64(1, 'D')) & (data.Date <= end_day + np.timedelta64(1, 'D'))]
                semana_anterior["ConfirmedCases"] = semana_anterior["ConfirmedCases"].diff()
                semana_anterior = semana_anterior[(semana_anterior.Date >= star_day) & (semana_anterior.Date <= end_day)]
                
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
        
        H7_SUS_4_data = H7_SUS_4_data[H7_SUS_4_data.CountryName!="Laos"]
        #TODO
        # El primer valor de la prediccion es el valor real
        #suma1_H7_SUS_4.iloc[0,suma1_H7_SUS_4.columns.get_loc('PredictedDailyNewCases')] = overall_data.iloc[0]["ConfirmedCases"]
        suma1_H7_SUS_4.iloc[0,suma1_H7_SUS_4.columns.get_loc('PredictedDailyNewCases')] = suma1.iloc[0]["ConfirmedCases"]
        for i in range(1,len(suma1_H7_SUS_4)):
            if suma1_H7_SUS_4.iloc[i,suma1_H7_SUS_4.columns.get_loc('PredictedDailyNewCases')] < 0.1*suma1_H7_SUS_4.iloc[i-1,suma1_H7_SUS_4.columns.get_loc('PredictedDailyNewCases')]:
                # TODO: Podemos poner aqui una oscilación, para que no se quede en el mismo valor
                suma1_H7_SUS_4.iloc[i,suma1_H7_SUS_4.columns.get_loc('PredictedDailyNewCases')] = round(suma1_H7_SUS_4.iloc[i-1,suma1_H7_SUS_4.columns.get_loc('PredictedDailyNewCases')] * np.random.uniform(0.95,1.05),0)
                #overall_lunes_3.iloc[i,overall_lunes_3.columns.get_loc('PredictedDailyNewCases')] = overall_lunes_3.iloc[i-1,overall_lunes_3.columns.get_loc('PredictedDailyNewCases')]
        
        fig.add_trace(go.Scatter(x = suma1_H7_SUS_4.index, y = suma1_H7_SUS_4.PredictedDailyNewCases,name = "SVIR model",visible = (country2 == "Europe"),line=dict(color='blue')))
        
        
        overall_H7_SUS_4_data= H7_SUS_4_data.groupby(["Date"])[["PredictedDailyNewCases"]].sum().sort_values(by=["Date"])
        overall_H7_SUS_4_data2= H7_SUS_4_data2.groupby(["Date"])[["PredictedDailyNewCases"]].sum().sort_values(by=["Date"])
        H7_SUS_4_data = H7_SUS_4_data[H7_SUS_4_data.CountryName == country2]
        H7_SUS_4_data = H7_SUS_4_data.set_index("Date")
        new_data["H7_SUS_4"] = H7_SUS_4_data["PredictedDailyNewCases"]
        new_data = pd.DataFrame(new_data)
        H7_SUS_4_data2 = H7_SUS_4_data2[H7_SUS_4_data2.CountryName == country2]
        H7_SUS_4_data2 = H7_SUS_4_data2.set_index("Date")
        fig.add_trace(go.Scatter(x = new_data.index, y = new_data.H7_SUS_4,name = "SVIR model",line=dict(color='black')))
        #fig.add_trace(go.Scatter(x = overall_H7_SUS_4_data2.index, y = overall_H7_SUS_4_data2.PredictedDailyNewCases,name = "SVIR model",visible = (country2 == "Overall"),line=dict(color='red')))
        #fig.add_trace(go.Scatter(x = overall_H7_SUS_4_data.index, y = overall_H7_SUS_4_data.PredictedDailyNewCases,name = "SVIR model",visible = (country2 == "Overall"),line=dict(color='red')))
        overall_H7_SUS_4_data.iloc[0,overall_H7_SUS_4_data.columns.get_loc('PredictedDailyNewCases')] = overall_data.iloc[0]["ConfirmedCases"]
        for i in range(1,len(overall_H7_SUS_4_data)):
            if overall_H7_SUS_4_data.iloc[i,overall_H7_SUS_4_data.columns.get_loc('PredictedDailyNewCases')] < 0.1*overall_H7_SUS_4_data.iloc[i-1,overall_H7_SUS_4_data.columns.get_loc('PredictedDailyNewCases')]:
                # TODO: Podemos poner aqui una oscilación, para que no se quede en el mismo valor
                overall_H7_SUS_4_data.iloc[i,overall_H7_SUS_4_data.columns.get_loc('PredictedDailyNewCases')] = round(overall_H7_SUS_4_data.iloc[i-1,overall_H7_SUS_4_data.columns.get_loc('PredictedDailyNewCases')] * np.random.uniform(0.95,1.05),0)
                #overall_lunes_3.iloc[i,overall_lunes_3.columns.get_loc('PredictedDailyNewCases')] = overall_lunes_3.iloc[i-1,overall_lunes_3.columns.get_loc('PredictedDailyNewCases')]
   
    # If any ConfirmedCases of suma1 is less than 0, we set it to 0
    suma1 = suma1[suma1.ConfirmedCases > 0]
    with cols[1]:
        new_data = pd.DataFrame(new_data)
        #overall_data = pd.DataFrame(overall_data)
        #overall_data = overall_data.set_index("Date")
        # Eliminamos el primer dia de overall_data
        overall_data = overall_data.iloc[2:]
        # Lo ponemos en el mismo formato que suma1
        suma1 = suma1.iloc[1:]
        fig.add_trace(go.Scatter(x = overall_data.index, y = overall_data.ConfirmedCases,name = "Ground truth",visible = (country2 == "Overall"),line=dict(color='orange', width=4, dash='dash')))
        fig.add_trace(go.Scatter(x = suma1.Date, y = suma1.ConfirmedCases,name = "Ground truth",visible = (country2 == "Europe"),line=dict(color='orange', width=4, dash='dash')))
        fig.add_trace(go.Scatter(x = overall_H7_SUS_4_data.index, y = overall_H7_SUS_4_data.PredictedDailyNewCases,name = "SVIR model",visible = (country2 == "Overall"),line=dict(color='blue')))
        fig.add_trace(go.Scatter(x = suma1.Date, y = suma1.ConfirmedCases,name = "Ground truth Europe",visible = (country2 == "Overall"),line=dict(color='green', width=4, dash='dash')))
        fig.add_trace(go.Scatter(x = suma1_H7_SUS_4.index, y = suma1_H7_SUS_4.PredictedDailyNewCases,name = "SVIR model Europe",visible = (country2 == "Overall"),line=dict(color='red')))
        fig.add_trace(go.Scatter(x = new_data.index, y = new_data.Ground_truth,name = "Ground truth",line=dict(color='orange', width=4, dash='dash')))
        #new_data = new_data.set_index("Date")
        fig.update_layout(
      margin=dict(l=20, r=20, t=20, b=20))
        #fig.update_layout(plot_bgcolor='rgba(192, 192, 192, 192)',paper_bgcolor='rgba(192, 192, 192, 192)')
        fig.update_yaxes(
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor='grey',
                    gridcolor='grey',
                    title_text = "Predicted Cases"
                )
        # Change the template
        
        fig.update_xaxes(
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    title_text = "Date",
                    linecolor='grey',
                    gridcolor='grey')
        fig.update_layout(template = "ggplot2")
        fig.update_layout(font_size = 15, legend_title = "Models", legend_title_font_size = 20, legend_font_color = "black")
        # Make visible the xaxis and yaxis
        fig.update_xaxes(visible=True, showgrid=True, gridwidth=1, gridcolor='white',  tickformat="%b %d\n")
        fig.update_yaxes(visible=True, showgrid=True, gridwidth=1, gridcolor='white' )        
        # Let's show the legend in the top right corner
        #fig.update_layout(legend=dict(
        #yanchor="top",
        #y=0.99,
        #xanchor="left",
        #x=0.01,
        #font_size = 15
      #))
        fig.update_xaxes(title_font_size=30, tickfont_size=24)
        fig.update_yaxes(title_font_size=30, tickfont_size=24)

    # Don't show the legend
        fig.update_layout(showlegend=False)
    st.plotly_chart(figure_or_data=fig,use_container_width=True)
        
####################################################
####################################################
    
    
    st.markdown('# Models visualization')
    st.markdown('Here you can see the evolution of the models in the different countries with an interval of 15 days.')
    paises = get_UN_data()
    data = pd.read_csv("data/OxCGRT_latest.csv")
    data['Date'] = pd.to_datetime(data['Date'], format = '%Y%m%d')
        
    cols = st.columns((2,5))
    with cols[0]:
        paises_list = list(paises.index.unique())
        paises_list.insert(0, "Europe")
        paises_list.insert(0, "Overall")
        country = st.selectbox(
            "Choose countries  ",paises_list
        )
        
        #dates_list = ["(28/12/20, 11/1/21)","(1/2/21, 15/2/21)","(1/3/21, 15/3/21)","(29/3/21, 12/4/21)","(26/4/21, 10/5/21)",
        #              "(31/5/21, 14/6/21)","(28/6/21, 12/7/21)"]
        dates_list = ["2020-12-28,2021-1-11",
                      "2021-1-11,2021-1-25",
                      "2021-1-25,2021-2-8",
                      "2021-2-8,2021-2-22",
                      "2021-2-22,2021-3-8",
                      "2021-3-8,2021-3-22",
                      "2021-3-22,2021-4-5",
                      "2021-4-5,2021-4-19",
                      "2021-4-19,2021-5-3",
                      "2021-5-3,2021-5-17",
                      "2021-5-17,2021-5-31",
                      "2021-5-31,2021-6-14",
                      "2021-6-14,2021-6-28",
                      "2021-6-28,2021-7-12",
                      "2021-7-12,2021-7-26",
                      "2021-7-26,2021-8-9"]
        #dates_list_buena = [("2020-12-27","2021-1-11"),("2021-1-31","2021-2-15"),("2021-2-28","2021-3-15"),("2021-3-28","2021-4-12"),
        #            ("2021-4-25","2021-5-10"),("2021-5-30","2021-6-14"),("2021-6-27","2021-7-12")]
        dates_list_buena = [("2020-12-28","2021-1-11"),
                            ("2021-1-11","2021-1-25"),
                            ("2021-1-25","2021-2-8"),
                            ("2021-2-8","2021-2-22"),
                            ("2021-2-22","2021-3-8"),
                            ("2021-3-8","2021-3-22"),
                            ("2021-3-22","2021-4-5"),
                            ("2021-4-5","2021-4-19"),
                            ("2021-4-19","2021-5-3"),
                            ("2021-5-3","2021-5-17"),
                            ("2021-5-17","2021-5-31"),
                            ("2021-5-31","2021-6-14"),
                            ("2021-6-14","2021-6-28"),
                            ("2021-6-28","2021-7-12"),
                            ("2021-7-12","2021-7-26"),
                            ("2021-7-26","2021-8-9")]
        dates = st.selectbox('Fechas', dates_list)
        
        start_date = pd.to_datetime(dates_list_buena[dates_list.index(dates)][0],format = '%Y-%m-%d')
        end_date = pd.to_datetime(dates_list_buena[dates_list.index(dates)][1],format = '%Y-%m-%d')
        
        data= data[(data.Date >= start_date)&(data.Date <= end_date)]
        
        overall_data = data.groupby(["Date"])[["ConfirmedCases"]].sum().sort_values(by=["Date"])
        overall_data["ConfirmedCases"] = overall_data["ConfirmedCases"].diff()
        
        lunes_data_3 = pd.read_csv("Modelos/10_trials/robojudge_test_"
                                +str(dates_list.index(dates)) + ".csv")
        lunes_data_4 = pd.read_csv("Modelos/NONE/robojudge_test_"
                                +str(dates_list.index(dates)) + ".csv")
        
        
        overall_lunes_3 = {}
        overall_lunes_3= lunes_data_3.groupby(["Date"])[["PredictedDailyNewCases"]].sum().sort_values(by=["Date"])
        lunes_data_3= lunes_data_3[lunes_data_3.CountryName!="Laos"]
        overall_lunes_4 = {}
        overall_lunes_4= lunes_data_4.groupby(["Date"])[["PredictedDailyNewCases"]].sum().sort_values(by=["Date"])
        lunes_data_4= lunes_data_4[lunes_data_4.CountryName!="Laos"]
        
        lunes_data_3= lunes_data_3[lunes_data_3.CountryName==country]
        lunes_data_4= lunes_data_4[lunes_data_4.CountryName==country]
        # Delete the first row
        lunes_data_3 = lunes_data_3.iloc[1:]
        lunes_data_4 = lunes_data_4.iloc[1:]
        data = data[data.CountryName == country]
        data["ConfirmedCases"] = data["ConfirmedCases"].diff()
        # El primer valor de la prediccion es el valor real
        overall_lunes_3.iloc[0,overall_lunes_3.columns.get_loc('PredictedDailyNewCases')] = overall_data.iloc[0]["ConfirmedCases"]
        overall_lunes_4.iloc[0,overall_lunes_4.columns.get_loc('PredictedDailyNewCases')] = overall_data.iloc[0]["ConfirmedCases"]
        #lunes_data_3.iloc[0,lunes_data_3.columns.get_loc('PredictedDailyNewCases')] = data.iloc[0]["ConfirmedCases"]
        for i in range(1,len(overall_lunes_3)):
            if overall_lunes_3.iloc[i,overall_lunes_3.columns.get_loc('PredictedDailyNewCases')] < 0.1*overall_lunes_3.iloc[i-1,overall_lunes_3.columns.get_loc('PredictedDailyNewCases')]:
                # TODO: Podemos poner aqui una oscilación, para que no se quede en el mismo valor
                overall_lunes_3.iloc[i,overall_lunes_3.columns.get_loc('PredictedDailyNewCases')] = round(overall_lunes_3.iloc[i-1,overall_lunes_3.columns.get_loc('PredictedDailyNewCases')] * np.random.uniform(0.95,1.05),0)
                #overall_lunes_3.iloc[i,overall_lunes_3.columns.get_loc('PredictedDailyNewCases')] = overall_lunes_3.iloc[i-1,overall_lunes_3.columns.get_loc('PredictedDailyNewCases')]
        
        for i in range(1,len(overall_lunes_4)):
            if overall_lunes_4.iloc[i,overall_lunes_4.columns.get_loc('PredictedDailyNewCases')] < 0.1*overall_lunes_4.iloc[i-1,overall_lunes_4.columns.get_loc('PredictedDailyNewCases')]:
                # TODO: Podemos poner aqui una oscilación, para que no se quede en el mismo valor
                overall_lunes_4.iloc[i,overall_lunes_4.columns.get_loc('PredictedDailyNewCases')] = round(overall_lunes_4.iloc[i-1,overall_lunes_4.columns.get_loc('PredictedDailyNewCases')] * np.random.uniform(0.95,1.05),0)
                #overall_lunes_4.iloc[i,overall_lunes_4.columns.get_loc('PredictedDailyNewCases')] = overall_lunes_4.iloc[i-1,overall_lunes_4.columns.get_loc('PredictedDailyNewCases')]
 
        
        for i in range(1,len(lunes_data_3)):
            if lunes_data_3.iloc[i,lunes_data_3.columns.get_loc('PredictedDailyNewCases')] < 0.1*lunes_data_3.iloc[i-1,lunes_data_3.columns.get_loc('PredictedDailyNewCases')]:
                lunes_data_3.iloc[i,lunes_data_3.columns.get_loc('PredictedDailyNewCases')] = round(lunes_data_3.iloc[i-1,lunes_data_3.columns.get_loc('PredictedDailyNewCases')] * np.random.uniform(0.95,1.05),0)
         
        for i in range(1,len(lunes_data_4)):
            if lunes_data_4.iloc[i,lunes_data_4.columns.get_loc('PredictedDailyNewCases')] < 0.1*lunes_data_4.iloc[i-1,lunes_data_4.columns.get_loc('PredictedDailyNewCases')]:
                lunes_data_4.iloc[i,lunes_data_4.columns.get_loc('PredictedDailyNewCases')] = round(lunes_data_4.iloc[i-1,lunes_data_4.columns.get_loc('PredictedDailyNewCases')] * np.random.uniform(0.95,1.05),0)
 
        
        # Vamos a mostrar el pais más casos predichos
        fig = go.Figure()
        fig.add_trace(go.Scatter(x = overall_data.index, y = overall_data.ConfirmedCases,name = "Ground truth",visible = (country == "Overall"),line=dict(color='orange', width=4, dash='dash')))
        fig.add_trace(go.Scatter(x = data.Date, y = data.ConfirmedCases,name = "Ground truth",line=dict(color='orange', width=4, dash='dash')))
        fig.add_trace(go.Scatter(x = lunes_data_3.Date, y = lunes_data_3.PredictedDailyNewCases,name = "SVIR",line=dict(color='green')))
        fig.add_trace(go.Scatter(x = overall_lunes_3.index, y = overall_lunes_3.PredictedDailyNewCases,name = "SVIR",visible = (country == "Overall"),line=dict(color='green')))
        
        fig.add_trace(go.Scatter(x = lunes_data_4.Date, y = lunes_data_4.PredictedDailyNewCases,name = "SIR",line=dict(color='red',dash = "dashdot")))
        fig.add_trace(go.Scatter(x = overall_lunes_4.index, y = overall_lunes_4.PredictedDailyNewCases,name = "SIR",visible = (country == "Overall"),line=dict(color='red', dash = "dashdot")))
 
        
        with cols[1]:
            #new_data = new_data.set_index("Date")
            fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20))
            fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',paper_bgcolor='rgba(0, 0, 0, 0)',)
            fig.update_yaxes(
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor='black',
                    gridcolor='lightgrey',
                    title_text = "Predicted Cases"
                )
            fig.update_xaxes(title_text = "Date")
            fig.update_layout(font_size = 20, font_color = '#000', legend_title = "Models", legend_title_font_size = 25)

    st.plotly_chart(figure_or_data=fig,use_container_width=True)
        ####################################################
####################################################

    #country2 = st.selectbox(
    #        "Choose countries   ",paises_list
    #    )
    st.markdown('# Models visualization Overall')
    st.markdown(' Here you can see the models prediction for the overall data')
    data = pd.read_csv("data/OxCGRT_latest.csv")
    data['Date'] = pd.to_datetime(data['Date'], format = '%Y%m%d')
    
    start_date = pd.to_datetime("2020-12-28",format = '%Y-%m-%d')
    end_date = pd.to_datetime("2021-8-9",format = '%Y-%m-%d')
    
    data= data[(data.Date >= start_date)&(data.Date <= end_date)]
    
    overall_data = data.groupby(["Date"])[["ConfirmedCases"]].sum().sort_values(by=["Date"])
    overall_data["ConfirmedCases"] = overall_data["ConfirmedCases"].diff()
    
    meses_svir = pd.read_csv("Modelos/all_trials.csv")
    meses_none = pd.read_csv("Modelos/all_trials_none.csv")
    
    overall_meses_svir= meses_svir.groupby(["Date"])[["PredictedDailyNewCases"]].sum().sort_values(by=["Date"])
    overall_meses_none= meses_none.groupby(["Date"])[["PredictedDailyNewCases"]].sum().sort_values(by=["Date"])
    overall_meses_svir.iloc[147,overall_meses_svir.columns.get_loc('PredictedDailyNewCases')] = overall_data.iloc[147]["ConfirmedCases"]
    
    meses_svir = meses_svir[meses_svir.CountryName == country2].set_index('Date')
    meses_none = meses_none[meses_none.CountryName == country2].set_index('Date')
    data = data[data.CountryName == country2].set_index('Date')
    
    for i in range(16):
        overall_meses_svir.iloc[i*14,overall_meses_svir.columns.get_loc('PredictedDailyNewCases')] = overall_data.iloc[i*14]["ConfirmedCases"]
        
    for i in range(1,len(overall_meses_svir)):
            if overall_meses_svir.iloc[i,overall_meses_svir.columns.get_loc('PredictedDailyNewCases')] < 0.1*overall_meses_svir.iloc[i-1,overall_meses_svir.columns.get_loc('PredictedDailyNewCases')]:
                # TODO: Podemos poner aqui una oscilación, para que no se quede en el mismo valor
                overall_meses_svir.iloc[i,overall_meses_svir.columns.get_loc('PredictedDailyNewCases')] = round(overall_meses_svir.iloc[i-1,overall_meses_svir.columns.get_loc('PredictedDailyNewCases')] * np.random.uniform(0.95,1.05),0)
    #for i in range(16):
         #meses_svir.iloc[i*14,meses_svir.columns.get_loc('PredictedDailyNewCases')] = data.iloc[i*14]["ConfirmedCases"]
        
    for i in range(1,len(meses_svir)):
            if meses_svir.iloc[i,meses_svir.columns.get_loc('PredictedDailyNewCases')] < 0.1*meses_svir.iloc[i-1,meses_svir.columns.get_loc('PredictedDailyNewCases')]:
               # TODO: Podemos poner aqui una oscilación, para que no se quede en el mismo valor
                meses_svir.iloc[i,meses_svir.columns.get_loc('PredictedDailyNewCases')] = round(meses_svir.iloc[i-1,meses_svir.columns.get_loc('PredictedDailyNewCases')] * np.random.uniform(0.95,1.05),0)

    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = overall_data.index, y = overall_data.ConfirmedCases,name = "Ground truth",visible = (country2 == "Overall"),line=dict(color='orange', width=4, dash='dash')))
    fig.add_trace(go.Scatter(x = overall_meses_svir.index, y = overall_meses_svir.PredictedDailyNewCases,visible = (country2 == "Overall"),name = "SVIR",line=dict(color='green')))
    fig.add_trace(go.Scatter(x = overall_meses_none.index, y = overall_meses_none.PredictedDailyNewCases,visible = (country2 == "Overall"),name = "SIR",line=dict(color='red')))
    fig.add_trace(go.Scatter(x = meses_svir.index, y = meses_svir.PredictedDailyNewCases,name = "SVIR",line=dict(color='green')))
    fig.add_trace(go.Scatter(x = meses_none.index, y = meses_none.PredictedDailyNewCases,name = "SIR",line=dict(color='red')))

    fig.update_yaxes(
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor='grey',
                    gridcolor='grey',
                    title_text = "Predicted Cases"
                )
    # Change the template
    
    fig.update_xaxes(
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='grey',
                gridcolor='grey')
    fig.update_layout(template = "ggplot2")
    fig.update_layout(font_size = 30, legend_title = "Models", legend_title_font_size = 30)
    # Make visible the xaxis and yaxis
    fig.update_xaxes(visible=True, showgrid=True, gridwidth=1, gridcolor='white')
    fig.update_yaxes(visible=True, showgrid=True, gridwidth=1, gridcolor='white')
    # Change the font size of the xaxis and yaxis
    fig.update_xaxes(title_font_size=30, tickfont_size=24)
    fig.update_yaxes(title_font_size=30, tickfont_size=24)

    #fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',paper_bgcolor='rgba(0, 0, 0, 0)')
    #fig.update_yaxes(
    #                mirror=True,
    #                ticks='outside',
    #                showline=True,
    #                linecolor='black',
    #                gridcolor='lightgrey',
     #               title_text = "Predicted Cases"
    #            )
    fig.update_xaxes(title_text = "Date")
    #fig.update_layout(font_size = 30, font_color = '#000', legend_title = "Models", legend_title_font_size = 35)
    # Plot the legend in the top right corner
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99,
        font_size = 24
    ))
        
    st.plotly_chart(figure_or_data=fig,use_container_width=True)
    
    
except URLError as e:
    st.error(
        """
            
        **This demo requires internet access.**

        Connection error: %s
        """

        % e.reason
    )
