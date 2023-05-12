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
import plotly.graph_objects as go
import sys
#sys.path.append('C:/Users/Sergio/Documents/valencia-ia4covid-xprize-master/valencia-ia4covid-xprize-master')

logo = 'https://ellisalicante.org/assets/xprize/images/logo_oscuro.png'
carpeta = 'C:/Users/Sergio/Documents/valencia-ia4covid-xprize-master/valencia-ia4covid-xprize-master/'

st.set_page_config(layout = 'wide')

@st.cache
def get_UN_data():
    paises = ['Argentina', 'Austria', 'Belgium', 'Bulgaria', 'Canada', 'Croatia',
       'Cyprus', 'Czech Republic', 'Denmark', 'Ecuador', 'Estonia', 'Finland',
       'France', 'Germany', 'Hungary', 'Ireland', 'Italy', 'Latvia',
       'Lithuania', 'Luxembourg', 'Netherlands',
       'Norway', 'Poland', 'Portugal', 'Slovak Republic', 'Slovenia', 'Spain',
       'Sweden', 'Switzerland', 'United States']
    return paises

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
        paises_list = list(paises)
        paises_list.insert(0, "Europe")
        country2 = st.selectbox(
            "Choose countries ",paises_list
        )
        
        months_list = ["January","February","March","April"]
        months_list_short = ['2021-01','2021-02','2021-03','2021-04','2021-05']
        months_dates = ["2020-12-31","2021-01-31","2021-01-31","2021-02-28","2021-02-28","2021-03-31","2021-03-31","2021-04-30",
                        "2021-04-30"]
        #months_list_short = ["jan","feb","mar","apr","may"]
        month = st.selectbox('Choose a month   ', months_list)
        # Nos quedamos con el indice del mes seleccionado
        lista_idx = [0,1,2,3]
        idx = months_list.index(month)
        h7_waning = pd.read_csv('Modelos/last_pred/h7_waning_pred.csv')
        h7_casos = pd.read_csv('Modelos/last_pred/h7_casos_pred.csv')
        none_waning = pd.read_csv('Modelos/last_pred/none_waning_pred.csv')
        none_casos = pd.read_csv('Modelos/last_pred/none_casos_pred.csv')
        xprize = pd.read_csv('Modelos/last_pred/xprizes_pred.csv')
        # Drop duplicated rows
        
        h7_waning = h7_waning[h7_waning['CountryName'] == country2]
        h7_waning['fecha'] = pd.to_datetime(h7_waning['fecha'])
        h7_waning['month'] = h7_waning['fecha'].dt.strftime('%Y-%m')
        st.write(" La selección es: ", months_list_short[idx])
        h7_waning = h7_waning[h7_waning['month'] == months_list_short[idx]]
        h7_waning = h7_waning.groupby(['CountryName','fecha']).mean().reset_index()
        h7_waning['pred'] = h7_waning['pred'].rolling(window=7, min_periods=1).mean()
        h7_waning['pred_sir'] = h7_waning['pred_sir'].rolling(window=7, min_periods=1).mean()
        h7_waning['truth'] = h7_waning['truth'].rolling(window=7, min_periods=1).mean()
        
        h7_casos = h7_casos[h7_casos['CountryName'] == country2]
        h7_casos['fecha'] = pd.to_datetime(h7_casos['fecha'])
        h7_casos['month'] = h7_casos['fecha'].dt.strftime('%Y-%m')
        h7_casos = h7_casos[h7_casos['month'] == months_list_short[idx]]
        h7_casos = h7_casos.groupby(['CountryName','fecha']).mean().reset_index()
        h7_casos['pred'] = h7_casos['pred'].rolling(window=7, min_periods=1).mean()
        h7_casos['pred_sir'] = h7_casos['pred_sir'].rolling(window=7, min_periods=1).mean()
        
        none_waning = none_waning[none_waning['CountryName'] == country2]
        none_waning['fecha'] = pd.to_datetime(none_waning['fecha'])
        none_waning['month'] = none_waning['fecha'].dt.strftime('%Y-%m')
        none_waning = none_waning[none_waning['month'] == months_list_short[idx]]
        none_waning = none_waning.groupby(['CountryName','fecha']).mean().reset_index()
        none_waning['pred'] = none_waning['pred'].rolling(window=7, min_periods=1).mean()
        none_waning['pred_sir'] = none_waning['pred_sir'].rolling(window=7, min_periods=1).mean()
        
        none_casos = none_casos[none_casos['CountryName'] == country2]
        none_casos['fecha'] = pd.to_datetime(none_casos['fecha'])
        none_casos['month'] = none_casos['fecha'].dt.strftime('%Y-%m')
        none_casos = none_casos[none_casos['month'] == months_list_short[idx]]
        none_casos = none_casos.groupby(['CountryName','fecha']).mean().reset_index()
        none_casos['pred'] = none_casos['pred'].rolling(window=7, min_periods=1).mean()
        none_casos['pred_sir'] = none_casos['pred_sir'].rolling(window=7, min_periods=1).mean()
        
        xprize = xprize[xprize['CountryName'] == country2]
        xprize['fecha'] = pd.to_datetime(xprize['fecha'])
        xprize['month'] = xprize['fecha'].dt.strftime('%Y-%m')
        xprize = xprize[xprize['month'] == months_list_short[idx]]
        xprize = xprize.groupby(['CountryName','fecha']).mean().reset_index()
        xprize['pred'] = xprize['pred'].rolling(window=7, min_periods=1).mean()
        xprize['pred_sir'] = xprize['pred_sir'].rolling(window=7, min_periods=1).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x = h7_waning['fecha'], y = h7_waning['pred'], name = "H7 & VacW SVIR", line = dict(color = 'red', width = 2)))
        fig.add_trace(go.Scatter(x = h7_waning['fecha'], y = h7_waning['pred_sir'], name = "H7 & VacW SIR", line = dict(color = 'green', width = 2)))
        fig.add_trace(go.Scatter(x = none_waning['fecha'], y = none_waning['pred'], name = "No H7 & VacW SVIR", line = dict(color = 'blue', width = 2)))
        fig.add_trace(go.Scatter(x = none_waning['fecha'], y = none_waning['pred_sir'], name = "No H7 & VacW SIR", line = dict(color = 'purple', width = 2)))
        fig.add_trace(go.Scatter(x = xprize['fecha'], y = xprize['pred'], name = "XPrize", line = dict(color = 'black', width = 2)))
        fig.add_trace(go.Scatter(x = h7_waning['fecha'], y = h7_waning['truth'], name = "Ground Truth Daily New Cases", line = dict(color = 'orange', width = 4, dash = 'dash')))
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
        fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        font_size = 15
      ))
        fig.update_xaxes(title_font_size=30, tickfont_size=24)
        fig.update_yaxes(title_font_size=30, tickfont_size=24)

    # Don't show the legend
        fig.update_layout(showlegend=False)
    st.plotly_chart(figure_or_data=fig,use_container_width=True)
    
    
except URLError as e:
    st.error(
        """
            
        **This demo requires internet access.**

        Connection error: %s
        """

        % e.reason
    )
