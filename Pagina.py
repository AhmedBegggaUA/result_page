# -*- coding: utf-8 -*-
"""
Created on Wed May 11 13:04:24 2022

@author: Sergio
"""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd
import altair as alt
from PIL import Image
import matplotlib.pyplot as plt
from urllib.error import URLError
import numpy as np

import sys
#sys.path.append('C:/Users/Sergio/Documents/valencia-ia4covid-xprize-master/valencia-ia4covid-xprize-master')

logo = 'https://ellisalicante.org/assets/xprize/images/logo_oscuro.png'
data_URL = "https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_nat_latest.csv"

#st.set_page_config(layout = 'wide')
st.set_page_config(page_title = 'V4C',page_icon='https://ellisalicante.org/assets/xprize/images/logo_oscuro.png',layout = 'wide')
@st.cache
def get_UN_data():
    paises = pd.read_csv("countries_regions.csv")
    return paises.set_index("CountryName")
@st.cache
def get_prescriptions_and_stringency():
    prescription = pd.read_csv("valencia.csv")
    stringency = pd.read_csv("stringency.csv")
    return prescription,stringency
def compute_pareto_set(objective1_list, objective2_list):
    """
    Return objective values for the subset of solutions that
    lie on the pareto front.
    """

    assert len(objective1_list) == len(objective2_list), \
        "Each solution must have a value for each objective."

    n_solutions = len(objective1_list)

    objective1_pareto = []
    objective2_pareto = []
    for i in range(n_solutions):
        is_in_pareto_set = True
        for j in range(n_solutions):
            if (objective1_list[j] < objective1_list[i]) and \
                    (objective2_list[j] < objective2_list[i]):
                is_in_pareto_set = False
        if is_in_pareto_set:
            objective1_pareto.append(objective1_list[i])
            objective2_pareto.append(objective2_list[i])

    return objective1_pareto, objective2_pareto
def plot_pareto_curve_plotly(objective1_list, objective2_list):
    """
    Plot the pareto curve given the objective values for a set of solutions.
    This curve indicates the area dominated by the solution set, i.e., 
    every point up and to the right is dominated.
    """
    
    objective1_pareto, objective2_pareto = compute_pareto_set(objective1_list, 
                                                              objective2_list)
    
    objective1_pareto, objective2_pareto = list(zip(*sorted(zip(objective1_pareto,
                                                                objective2_pareto))))
    
    xs = []
    ys = []
    
    xs.append(objective1_pareto[0])
    ys.append(objective2_pareto[0])
    
    for i in range(0, len(objective1_pareto)-1):
        
        # Add intermediate point between successive solutions
        xs.append(objective1_pareto[i+1])
        ys.append(objective2_pareto[i])
        
        # Add next solution on front
        xs.append(objective1_pareto[i+1])
        ys.append(objective2_pareto[i+1])
        
    return xs, ys
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
    st.sidebar.image(logo)
    pag = st.sidebar.radio("",("Pagina 1","Pagina 2"))
    
    if pag == "Pagina 1":
        col1, col2, col3 = st.columns([1,2,1])

        with col1:
            st.write("")

        with col2:
            st.image(logo)

        with col3:
           st.write("")

        st.markdown('# Who are we?')
        st.write('#### Introducction')
        cols = st.columns((2,1))
        cols[0].write('''We are a team of Spanish scientists who have been working since March 2020 in collaboration with the Valencian Government of Spain on using Data Science to help fight the SARS-CoV-2 pandemic. We have focused on 4 large areas of work: large-scale human mobility modeling via the analysis of aggregated, anonymized data derived from the mobile network infrastructure; computational epidemiological models; predictive models and citizen science by means of a large-scale citizen survey called COVID19impactsurvey which, with over 375,000 answers in Spain and around 150,000 answers from other countries is one of the largest COVID-19 citizen surveys to date. Our work has been awarded two competitive research grants. 
                      \n Since March, we have been developing two types of traditional computational epidemiological models: a metapopulation compartmental SEIR model and an agent-based model. However, for this challenge, we opted for a deep learning-based approach, inspired by the model suggested by the challenge organizers. Such an approach would enable us to build a model within the time frame of the competition with two key properties: be applicable to a large number of regions and be able to automatically learn the impact of the Non-Pharmaceutical Interventions (NPIs) on the transmission rate of the disease. The Pandemic COVID-19 XPRIZE challenge has been a great opportunity for our team to explore new modeling approaches and expand our scope beyond the Valencian region of Spain.''')
        cols[1].video('https://www.youtube.com/watch?v=RZ9wsSGH8U8')


        st.markdown('# Confirmed cases of Covid-19 and applied NPIs')

        cols = st.columns((1,1,5))
        paises = get_UN_data()
        data_ini = pd.read_csv("data/OxCGRT_latest.csv")
        data = data_ini
        with cols[0]:
            country = st.selectbox("Choose country",list(paises.index.unique()))

            data = data[data.CountryName == country]
            data['Date'] = pd.to_datetime(data['Date'], format = '%Y%m%d')

            today = min(data.Date)
            start_date = st.date_input('Start date', today)
            start_date = pd.to_datetime(start_date,format = '%Y-%m-%d')

            choose = st.radio("",("Confirmed Cases","Confirmed Deaths"))

        with cols[1]:
            reg = list(paises.index).count(country)==1
            region = " "
            regiones = list(paises[paises.index == country].RegionName.fillna(" "))
            region = cols[1].selectbox("Choose region", regiones)
            tomorrow = max(data.Date)
            end_date = st.date_input('End date', tomorrow)
            end_date = pd.to_datetime(end_date,format = '%Y-%m-%d')
            if start_date > end_date:
                st.error('Error: End date must fall after start date.')

        data = data[(data.Date >= start_date)&(data.Date <= end_date)]
        data = data.set_index("Date")
        data["ConfirmedCases7Days"] = data.groupby("CountryName")['ConfirmedCases'].rolling(7, center=False).mean().reset_index(0, drop=True)
        data["ConfirmedDeaths7Days"] = data.groupby("CountryName")['ConfirmedDeaths'].rolling(7, center=False).mean().reset_index(0, drop=True)


        with cols[2]:
            if choose == "Confirmed Cases":
                st.line_chart(data.ConfirmedCases7Days.diff().fillna(0))
            else:
                st.line_chart(data.ConfirmedDeaths7Days.diff().fillna(0))

        cols = st.columns((2,5))

        with cols[0]:
            rules = ["C1M_School closing","C2M_Workplace closing","C3M_Cancel public events",
                         "C4M_Restrictions on gatherings","C5M_Close public transport",
                         "C6M_Stay at home requirements","C7M_Restrictions on internal movement",
                         "C8EV_International travel controls","H1_Public information campaigns",
                         "H2_Testing policy","H3_Contact tracing","H6M_Facial Coverings"]
            rule = st.multiselect(
                    "Choose rule", rules,"C1M_School closing"
                )

            value_max = [3,3,2,4,2,3,2,4,2,3,2,4]

        with cols[1]:
            if len(rule)!=0:
                for j in range(len(rule)):
                    [xs,dates] = get_data_rule(data[rule[j]].fillna(0))
                    dataf = []
                    for k in range(value_max[j]):
                        dataf.append({rule[j]:str(k),"start":dates[0],"end":dates[0]})
                    for i in range(len(xs)):
                        dataf.append({rule[j]:str(int(xs[i])),"start":dates[i],"end":dates[i+1]})

                    data2 = pd.DataFrame(dataf)      

                    graf = alt.Chart(data2).mark_bar().encode(
                        x=alt.X('start',axis=alt.Axis(title='Date', labelAngle=-45, format = ("%b %Y"))),
                        x2='end',
                        y=rule[j],
                        color = alt.Color(rule[j],legend = None)
                    ).properties(width = 800)

                    st.altair_chart(graf,use_container_width=True)

        st.markdown("# Computational epidemiological models")
        cols = st.columns((5,2))

        with cols[0]:
            foto1 = Image.open("Foto1.png")
            st.image(foto1)

        with cols[1]:
            st.write("We have developed machine learning-based predictive models of the number"
                     "of hospitalizations and intensive care hospitalizations overall and for"
                     "SARS-CoV-2 patients. We have also developed a model to infer the prevalence"
                     "of the disease based on a few of the answers to our citizen survey "
                     "[https://covid19impactsurvey.org](https://covid19impactsurvey.org/)")




        st.markdown('# Predict cases of Covid-19')

        cols = st.columns((1,1,3))
        paises = get_UN_data()
        data_pred = pd.read_csv("predictions/robojudge_test.csv")
        with cols[0]:
            country_pred = st.selectbox("Choose countrys",list(paises.index.unique()))

            data_pred = data_pred[data_pred.CountryName == country_pred]
            data_pred['Date'] = pd.to_datetime(data_pred['Date'], format = '%Y-%m-%d')

            today_pred = min(data_pred.Date)
            start_date_pred = st.date_input('Start date', today_pred)
            start_date_pred = pd.to_datetime(start_date_pred,format = '%Y-%m-%d')

        with cols[1]:
            reg_pred = list(paises.index).count(country_pred)==1
            region = " "
            regiones = list(paises[paises.index == country_pred].RegionName.fillna(" "))
            region = cols[1].selectbox("Choose regions", regiones)
            tomorrow_pred = max(data_pred.Date)
            end_date_pred = st.date_input('End date', tomorrow_pred)
            end_date_pred = pd.to_datetime(end_date_pred,format = '%Y-%m-%d')
            if start_date_pred > end_date_pred:
                st.error('Error: End date must fall after start date.')

        with cols[2]:
            if not reg_pred:
                data_pred = data_pred[data_pred.RegionName == region]
            data_pred = data_pred[(data_pred.Date >= start_date_pred)&(data_pred.Date <= end_date_pred)]
            data_nopred = data_ini
            data_nopred['Date'] = pd.to_datetime(data_nopred['Date'], format = '%Y%m%d')
            data_nopred = data_nopred[(data_nopred.CountryName==country_pred)&(data_nopred.Date >= start_date_pred)&(data_nopred.Date <= end_date_pred)]
            data_nopred = data_nopred.set_index("Date")
            data_pred = data_pred.set_index("Date")

            chart_data_pred = pd.DataFrame({"pred": data_pred.PredictedDailyNewCases.diff().fillna(0),"nopred" :data_nopred.ConfirmedCases.diff().fillna(0)})
            #st.line_chart([data_pred.PredictedDailyNewCases.diff().fillna(0),data.ConfirmedCases.diff().fillna(0)],["1","2"])
            st.line_chart(chart_data_pred)

        st.markdown("# Prescriptor Models")
        cols = st.columns((2,5))

        with cols[0]:
            st.write("Our goal in the Prescription phase of the competition is to develop an"
                     "interpretable, data-driven and flexible prescription framework that would"
                     "be usable by non machine-learning experts, such as citizens and policy"
                     "makers in the Valencian Government. Our design principles are therefore"
                     "driven by developing interpretable and transparent models.")

            st.write("Given the intervention costs, it automatically generates up to 10"
                     "Pareto-optimal intervention plans. For each plan, it shows the resulting"
                     "number of cases and overall stringency, its position on the Pareto front"
                     "and the activation regime of each of the 12 types of interventions that"
                     "are part of the plan.")

        with cols[1]:
            foto2 = Image.open("Foto2.png")
            st.image(foto2)

    ############################################################################################################
    #                                         Prescriptor                                                      #
    ############################################################################################################
        st.markdown('# Prescript NPIs for Covid-19')
        cols = st.columns((1,3,3))
        paises = get_UN_data()
        data_pres = pd.read_csv("predictions/robojudge_test.csv")
        #The same as the predictor part
        with cols[0]:
            country_pres = st.selectbox("Choose the country",list(paises.index.unique()))
            data_pres = data_pres[data_pres.CountryName == country_pres]
            data_pres['Date'] = pd.to_datetime(data_pres['Date'], format = '%Y-%m-%d')
            today_pres = min(data_pres.Date)
            start_date_pres = st.date_input('Start date ', today_pres)
            start_date_pres = pd.to_datetime(start_date_pres,format = '%Y-%m-%d')

            tomorrow_pres = max(data_pres.Date)
            end_date_pres = st.date_input('End date ', tomorrow_pres)
            end_date_pres = pd.to_datetime(end_date_pres,format = '%Y-%m-%d')
            if start_date_pres > end_date_pres:
                st.error('Error: End date must fall after start date.')

        with cols[0]:
            reg_pres = list(paises.index).count(country_pres)==1
            region = " "
            regiones = list(paises[paises.index == country_pres].RegionName.fillna(" "))
            region = cols[0].selectbox("Choose the region ", regiones)
        with cols[0]:
            index = ["Index 0","Index 1","Index 2","Index 3","Index 4",
                    "Index 5","Index 6","Index 7","Index 8","Index 9"]
            value_max = [0,1,2,3,4,5,6,7,8,9]
            index = st.selectbox(
                    "Choose the level of stringency (0-9)", value_max,0
                )

            value_max = [0,1,2,3,4,5,6,7,8,9]
        with cols[1]:
            #Get the data
            prescriptions,stringency = get_prescriptions_and_stringency()
            country_name = country_pres
            if not reg_pres:
                stringency = stringency[(stringency.RegionName == region)]
            cdf = stringency[(stringency['PrescriptorName'] == 'V4C') & (stringency.CountryName == country_pres)]
            #Plotly
            fig = go.Figure()
            #We are going to plot different lines and scatter points, so we use the function add_trace
            #Inside the function we have the data and the type of plot, in this case scatter
            fig.add_trace(go.Scatter(x=cdf['Stringency'],y=cdf['PredictedDailyNewCases'],
                                     name="V4C", 
                                     mode='markers',
                                     marker=dict(size=10,color = 'rgb(29, 126, 235 )',
                                     line=dict(width=1,color='DarkSlateGrey'))))
            #Adittional function to get the pareto front (in the furture we will move it to another file)
            xs, ys = plot_pareto_curve_plotly(list(cdf['Stringency']),list(cdf['PredictedDailyNewCases']))
            #Same thing as before
            fig.add_trace(go.Scatter(x=xs,y=ys, 
                                    mode='lines',
                                    marker=dict(size=10,
                                    color = 'rgb(29, 126, 235 )')))
            #This plot I use it to show the point that we are going to choose
            fig.add_trace(go.Scatter(x=[np.asarray(cdf['Stringency'])[index]],y=[np.asarray(cdf['PredictedDailyNewCases'])[index]],
                                    mode='markers',
                                    marker=dict(size=14,color = 'rgb(255, 0, 0 )'),
                                    line=dict(width=3,color='DarkSlateGrey')))
            #Things to make the plot look better
            fig.update_layout(title='Pareto curve for '+country_name, 
                              xaxis_title='Stringency', 
                              yaxis_title='Predicted Daily New Cases',
                              font=dict(size=12))
            fig.update_layout(
                #margin=dict(l=20, r=20, t=20, b=20),    
                template='seaborn',
                paper_bgcolor='white',
            )
            data_fig1 = fig
            #Finally we plot the figure using st.plotly_chart
            st.plotly_chart(figure_or_data=fig)


        with cols[2]:
            #Same thing as the predictor
            prescription_index = index
            country_name = country_pres
            NPI_COLUMNS = ['C1_School closing',
                   'C2_Workplace closing',
                   'C3_Cancel public events',
                   'C4_Restrictions on gatherings',
                   'C5_Close public transport',
                   'C6_Stay at home requirements',
                   'C7_Restrictions on internal movement',
                   'C8_International travel controls',
                   'H1_Public information campaigns',
                   'H2_Testing policy',
                   'H3_Contact tracing',
                   'H6_Facial Coverings']
            region_name = None
            pdf = prescriptions
            #if not reg_pres:
            #    pdf = pdf[(pdf.RegionName == region)]
            #else:
            gdf = pdf[(pdf['PrescriptionIndex'] == prescription_index) &
                        (pdf.CountryName == country_name) &
                        (pdf['RegionName'].isna() if region_name is None else (pdf['RegionName'] == 'region_name'))]
            #Another way to plot plotly, in this case it is called plotly express (See the documentation), the easy plotly
            fig = px.bar(gdf, x ='Date', y=NPI_COLUMNS)
            data_fig2 = fig
            st.plotly_chart(figure_or_data=fig)
    else:
        col1, col2, col3 = st.columns([1,2,1])

        with col1:
            st.write("")

        with col2:
            st.image(logo)

        with col3:
           st.write("")

        st.markdown('# Who are we?')
        st.write('#### Introducction')
        cols = st.columns((2,1))
        cols[0].write('''We are a team of Spanish scientists who have been working since March 2020 in collaboration with the Valencian Government of Spain on using Data Science to help fight the SARS-CoV-2 pandemic. We have focused on 4 large areas of work: large-scale human mobility modeling via the analysis of aggregated, anonymized data derived from the mobile network infrastructure; computational epidemiological models; predictive models and citizen science by means of a large-scale citizen survey called COVID19impactsurvey which, with over 375,000 answers in Spain and around 150,000 answers from other countries is one of the largest COVID-19 citizen surveys to date. Our work has been awarded two competitive research grants. 
                      \n Since March, we have been developing two types of traditional computational epidemiological models: a metapopulation compartmental SEIR model and an agent-based model. However, for this challenge, we opted for a deep learning-based approach, inspired by the model suggested by the challenge organizers. Such an approach would enable us to build a model within the time frame of the competition with two key properties: be applicable to a large number of regions and be able to automatically learn the impact of the Non-Pharmaceutical Interventions (NPIs) on the transmission rate of the disease. The Pandemic COVID-19 XPRIZE challenge has been a great opportunity for our team to explore new modeling approaches and expand our scope beyond the Valencian region of Spain.''')
        cols[1].video('https://www.youtube.com/watch?v=RZ9wsSGH8U8')


        st.markdown('# Confirmed cases of Covid-19 and applied NPIs')

        cols = st.columns((1,1,5))
        paises = get_UN_data()
        data_ini = pd.read_csv("data/OxCGRT_latest.csv")
        data = data_ini
        with cols[0]:
            country = st.selectbox("Choose country",list(paises.index.unique()))

            data = data[data.CountryName == country]
            data['Date'] = pd.to_datetime(data['Date'], format = '%Y%m%d')

            today = min(data.Date)
            start_date = st.date_input('Start date', today)
            start_date = pd.to_datetime(start_date,format = '%Y-%m-%d')

            choose = st.radio("",("Confirmed Cases","Confirmed Deaths"))

        with cols[1]:
            reg = list(paises.index).count(country)==1
            region = " "
            regiones = list(paises[paises.index == country].RegionName.fillna(" "))
            region = cols[1].selectbox("Choose region", regiones)
            tomorrow = max(data.Date)
            end_date = st.date_input('End date', tomorrow)
            end_date = pd.to_datetime(end_date,format = '%Y-%m-%d')
            if start_date > end_date:
                st.error('Error: End date must fall after start date.')

        data = data[(data.Date >= start_date)&(data.Date <= end_date)]
        data = data.set_index("Date")
        data["ConfirmedCases7Days"] = data.groupby("CountryName")['ConfirmedCases'].rolling(7, center=False).mean().reset_index(0, drop=True)
        data["ConfirmedDeaths7Days"] = data.groupby("CountryName")['ConfirmedDeaths'].rolling(7, center=False).mean().reset_index(0, drop=True)


        with cols[2]:
            if choose == "Confirmed Cases":
                st.line_chart(data.ConfirmedCases7Days.diff().fillna(0))
            else:
                st.line_chart(data.ConfirmedDeaths7Days.diff().fillna(0))

        cols = st.columns((2,5))

        with cols[0]:
            rules = ["C1M_School closing","C2M_Workplace closing","C3M_Cancel public events",
                         "C4M_Restrictions on gatherings","C5M_Close public transport",
                         "C6M_Stay at home requirements","C7M_Restrictions on internal movement",
                         "C8EV_International travel controls","H1_Public information campaigns",
                         "H2_Testing policy","H3_Contact tracing","H6M_Facial Coverings"]
            rule = st.multiselect(
                    "Choose rule", rules,"C1M_School closing"
                )

            value_max = [3,3,2,4,2,3,2,4,2,3,2,4]

        with cols[1]:
            if len(rule)!=0:
                for j in range(len(rule)):
                    [xs,dates] = get_data_rule(data[rule[j]].fillna(0))
                    dataf = []
                    for k in range(value_max[j]):
                        dataf.append({rule[j]:str(k),"start":dates[0],"end":dates[0]})
                    for i in range(len(xs)):
                        dataf.append({rule[j]:str(int(xs[i])),"start":dates[i],"end":dates[i+1]})

                    data2 = pd.DataFrame(dataf)      

                    graf = alt.Chart(data2).mark_bar().encode(
                        x=alt.X('start',axis=alt.Axis(title='Date', labelAngle=-45, format = ("%b %Y"))),
                        x2='end',
                        y=rule[j],
                        color = alt.Color(rule[j],legend = None)
                    ).properties(width = 800)

                    st.altair_chart(graf,use_container_width=True)

        st.markdown("# Computational epidemiological models")
        cols = st.columns((5,2))

        with cols[0]:
            foto1 = Image.open("Foto1.png")
            st.image(foto1)

        with cols[1]:
            st.write("We have developed machine learning-based predictive models of the number"
                     "of hospitalizations and intensive care hospitalizations overall and for"
                     "SARS-CoV-2 patients. We have also developed a model to infer the prevalence"
                     "of the disease based on a few of the answers to our citizen survey "
                     "[https://covid19impactsurvey.org](https://covid19impactsurvey.org/)")




        st.markdown('# Predict cases of Covid-19')

        cols = st.columns((1,1,3))
        paises = get_UN_data()
        data_pred = pd.read_csv("predictions/robojudge_test.csv")
        with cols[0]:
            country_pred = st.selectbox("Choose countrys",list(paises.index.unique()))

            data_pred = data_pred[data_pred.CountryName == country_pred]
            data_pred['Date'] = pd.to_datetime(data_pred['Date'], format = '%Y-%m-%d')

            today_pred = min(data_pred.Date)
            start_date_pred = st.date_input('Start date', today_pred)
            start_date_pred = pd.to_datetime(start_date_pred,format = '%Y-%m-%d')

        with cols[1]:
            reg_pred = list(paises.index).count(country_pred)==1
            region = " "
            regiones = list(paises[paises.index == country_pred].RegionName.fillna(" "))
            region = cols[1].selectbox("Choose regions", regiones)
            tomorrow_pred = max(data_pred.Date)
            end_date_pred = st.date_input('End date', tomorrow_pred)
            end_date_pred = pd.to_datetime(end_date_pred,format = '%Y-%m-%d')
            if start_date_pred > end_date_pred:
                st.error('Error: End date must fall after start date.')

        with cols[2]:
            if not reg_pred:
                data_pred = data_pred[data_pred.RegionName == region]
            data_pred = data_pred[(data_pred.Date >= start_date_pred)&(data_pred.Date <= end_date_pred)]
            data_nopred = data_ini
            data_nopred['Date'] = pd.to_datetime(data_nopred['Date'], format = '%Y%m%d')
            data_nopred = data_nopred[(data_nopred.CountryName==country_pred)&(data_nopred.Date >= start_date_pred)&(data_nopred.Date <= end_date_pred)]
            data_nopred = data_nopred.set_index("Date")
            data_pred = data_pred.set_index("Date")

            chart_data_pred = pd.DataFrame({"pred": data_pred.PredictedDailyNewCases.diff().fillna(0),"nopred" :data_nopred.ConfirmedCases.diff().fillna(0)})
            #st.line_chart([data_pred.PredictedDailyNewCases.diff().fillna(0),data.ConfirmedCases.diff().fillna(0)],["1","2"])
            st.line_chart(chart_data_pred)

        st.markdown("# Prescriptor Models")
        cols = st.columns((2,5))

        with cols[0]:
            st.write("Our goal in the Prescription phase of the competition is to develop an"
                     "interpretable, data-driven and flexible prescription framework that would"
                     "be usable by non machine-learning experts, such as citizens and policy"
                     "makers in the Valencian Government. Our design principles are therefore"
                     "driven by developing interpretable and transparent models.")

            st.write("Given the intervention costs, it automatically generates up to 10"
                     "Pareto-optimal intervention plans. For each plan, it shows the resulting"
                     "number of cases and overall stringency, its position on the Pareto front"
                     "and the activation regime of each of the 12 types of interventions that"
                     "are part of the plan.")

        with cols[1]:
            foto2 = Image.open("Foto2.png")
            st.image(foto2)

    ############################################################################################################
    #                                         Prescriptor                                                      #
    ############################################################################################################
        st.markdown('# Prescript NPIs for Covid-19')
        cols = st.columns((1,3,3))
        paises = get_UN_data()
        data_pres = pd.read_csv("predictions/robojudge_test.csv")
        #The same as the predictor part
        with cols[0]:
            country_pres = st.selectbox("Choose the country",list(paises.index.unique()))
            data_pres = data_pres[data_pres.CountryName == country_pres]
            data_pres['Date'] = pd.to_datetime(data_pres['Date'], format = '%Y-%m-%d')
            today_pres = min(data_pres.Date)
            start_date_pres = st.date_input('Start date ', today_pres)
            start_date_pres = pd.to_datetime(start_date_pres,format = '%Y-%m-%d')

            tomorrow_pres = max(data_pres.Date)
            end_date_pres = st.date_input('End date ', tomorrow_pres)
            end_date_pres = pd.to_datetime(end_date_pres,format = '%Y-%m-%d')
            if start_date_pres > end_date_pres:
                st.error('Error: End date must fall after start date.')

        with cols[0]:
            reg_pres = list(paises.index).count(country_pres)==1
            region = " "
            regiones = list(paises[paises.index == country_pres].RegionName.fillna(" "))
            region = cols[0].selectbox("Choose the region ", regiones)
        with cols[0]:
            index = ["Index 0","Index 1","Index 2","Index 3","Index 4",
                    "Index 5","Index 6","Index 7","Index 8","Index 9"]
            value_max = [0,1,2,3,4,5,6,7,8,9]
            index = st.selectbox(
                    "Choose the level of stringency (0-9)", value_max,0
                )

            value_max = [0,1,2,3,4,5,6,7,8,9]
        with cols[1]:
            #Get the data
            prescriptions,stringency = get_prescriptions_and_stringency()
            country_name = country_pres
            if not reg_pres:
                stringency = stringency[(stringency.RegionName == region)]
            cdf = stringency[(stringency['PrescriptorName'] == 'V4C') & (stringency.CountryName == country_pres)]
            #Plotly
            fig = go.Figure()
            #We are going to plot different lines and scatter points, so we use the function add_trace
            #Inside the function we have the data and the type of plot, in this case scatter
            fig.add_trace(go.Scatter(x=cdf['Stringency'],y=cdf['PredictedDailyNewCases'],
                                     name="V4C", 
                                     mode='markers',
                                     marker=dict(size=10,color = 'rgb(29, 126, 235 )',
                                     line=dict(width=1,color='DarkSlateGrey'))))
            #Adittional function to get the pareto front (in the furture we will move it to another file)
            xs, ys = plot_pareto_curve_plotly(list(cdf['Stringency']),list(cdf['PredictedDailyNewCases']))
            #Same thing as before
            fig.add_trace(go.Scatter(x=xs,y=ys, 
                                    mode='lines',
                                    marker=dict(size=10,
                                    color = 'rgb(29, 126, 235 )')))
            #This plot I use it to show the point that we are going to choose
            fig.add_trace(go.Scatter(x=[np.asarray(cdf['Stringency'])[index]],y=[np.asarray(cdf['PredictedDailyNewCases'])[index]],
                                    mode='markers',
                                    marker=dict(size=14,color = 'rgb(255, 0, 0 )'),
                                    line=dict(width=3,color='DarkSlateGrey')))
            #Things to make the plot look better
            fig.update_layout(title='Pareto curve for '+country_name, 
                              xaxis_title='Stringency', 
                              yaxis_title='Predicted Daily New Cases',
                              font=dict(size=12))
            fig.update_layout(
                #margin=dict(l=20, r=20, t=20, b=20),    
                template='seaborn',
                paper_bgcolor='white',
            )
            data_fig1 = fig
            #Finally we plot the figure using st.plotly_chart
            st.plotly_chart(figure_or_data=fig)


        with cols[2]:
            #Same thing as the predictor
            prescription_index = index
            country_name = country_pres
            NPI_COLUMNS = ['C1_School closing',
                   'C2_Workplace closing',
                   'C3_Cancel public events',
                   'C4_Restrictions on gatherings',
                   'C5_Close public transport',
                   'C6_Stay at home requirements',
                   'C7_Restrictions on internal movement',
                   'C8_International travel controls',
                   'H1_Public information campaigns',
                   'H2_Testing policy',
                   'H3_Contact tracing',
                   'H6_Facial Coverings']
            region_name = None
            pdf = prescriptions
            #if not reg_pres:
            #    pdf = pdf[(pdf.RegionName == region)]
            #else:
            gdf = pdf[(pdf['PrescriptionIndex'] == prescription_index) &
                        (pdf.CountryName == country_name) &
                        (pdf['RegionName'].isna() if region_name is None else (pdf['RegionName'] == 'region_name'))]
            #Another way to plot plotly, in this case it is called plotly express (See the documentation), the easy plotly
            fig = px.bar(gdf, x ='Date', y=NPI_COLUMNS)
            data_fig2 = fig
            st.plotly_chart(figure_or_data=fig)
except URLError as e:
    st.error(
        """
            
        **This demo requires internet access.**

        Connection error: %s
        """

        % e.reason
    )
