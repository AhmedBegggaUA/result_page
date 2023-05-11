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
        
        months_list = ["January","February","March","April","May"]
        months_dates = ["2020-12-31","2021-01-31","2021-01-31","2021-02-28","2021-02-28","2021-03-31","2021-03-31","2021-04-30",
                        "2021-04-30"]
        months_list_short = ["jan","feb","mar","apr","may"]    
except URLError as e:
    st.error(
        """
            
        **This demo requires internet access.**

        Connection error: %s
        """

        % e.reason
    )
