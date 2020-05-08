#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# @author: Jiaji001


# standard library
 
import os
import io

# dash libs

import dash

import dash_core_components as dcc

import dash_html_components as html

from dash.dependencies import Input, Output

import plotly.figure_factory as ff

import plotly.graph_objs as go


# In[ ]:


# pydata stack

import pandas as pd

from sqlalchemy import create_engine


# In[ ]:



# set up database connnections

user = 'MKTG_PRDFOCOPLETL'
pw = 'staples123'
host = 'td-mrk-p-'
conn = create_engine('teradata://'+ user +':' + pw + '@'+ host + ':22/')


# In[ ]:


###########################
# Data Manipulation / Model
###########################

def fetch_data(q):
    result = pd.read_sql(
        sql=q,
        con=conn
    )
    return result


def get_yearmon():
    '''Returns the list of fiscal_year_month that are stored in the database'''

    yearmon_query = (
        f'''
        SELECT DISTINCT FISCAL_YEARMO
        FROM PRD_FOL_TMP.tyler_python_test2
        '''
    )
    yearmons = fetch_data(yearmon_query)
    yearmons = list(yearmons['FISCAL_YEARMO'].sort_values(ascending=True))
    return yearmons




def get_match_results(FISCAL_YEARMO):
    '''Returns match results for the selected prompts'''

    results_query = (
        f'''
        SELECT SalesVisitDate, NetSales, Qty
        FROM PRD_FOL_TMP.tyler_python_test2
        WHERE FISCAL_YEARMO='{FISCAL_YEARMO}'
        ORDER BY SalesVisitDate ASC
        '''
    )
    match_results = fetch_data(results_query)
    return match_results



def draw_date_sales_graph(results):
    dates = results['SalesVisitDate']
    sales = results['NetSales']

    figure = go.Figure(
        data=[
            go.Scatter(x=dates, y=sales, mode='lines+markers')
        ],
        layout=go.Layout(
            title='Sales Trend',
            showlegend=False
        )
   )

    return figure

'''def draw_date_sales_graph(results):
    dates = results['SalesVisitDate']
    sales = results['NetSales']

    figure = go.Figure(
        data=[
            go.Bar(x=dates, y=sales)
        ],
        layout=go.Layout(
            title='Sales Trend',
            showlegend=False
        )
    )

    return figure'''


def generate_table(dataframe, max_rows=50):
    '''Given dataframe, return template generated using Dash components
    '''
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )


def onLoad_yearmon_options():
    '''Actions to perform upon initial page load'''

    yearmon_options = (
        [{'label': FISCAL_YEARMO, 'value': FISCAL_YEARMO}
         for FISCAL_YEARMO in get_yearmon()]
    )
    return yearmon_options


# In[ ]:


#########################

# Dashboard Layout / View

#########################




# Set up Dashboard and create layout

app = dash.Dash(name = __name__)


app.layout = html.Div([



# Page Header

html.Div([

        html.H1('Dashboard Demo')

    ]),


# Dropdown Grid
        html.Div([
            # Select YearMon Dropdown
            html.Div([
                html.Div('Select YearMon', ),
                html.Div(dcc.Dropdown(id='yearmon-selector',
                                      options=onLoad_yearmon_options()),
                       )
            ]),

        ], ),

  

# Match Results Grid
html.Div([

        # Match Results Table
        html.Div(
            html.Table(id='match-results'),
            
        ),

            # Season Graph
        html.Div([
  
            # graph
            dcc.Graph(id='sales-graph')
            # style={},

        ], )
    ]),
])


# In[ ]:


#############################################
# Interaction Between Components / Controller
#############################################



# Load Match results
@app.callback(
    Output(component_id='match-results', component_property='children'),
    [
        Input(component_id='yearmon-selector', component_property='value')
    ]
)
def load_match_results(FISCAL_YEARMO):
    results = get_match_results(FISCAL_YEARMO)
    return generate_table(results, max_rows=5)




# Update Sales Point Graph
@app.callback(
    Output(component_id='sales-graph', component_property='figure'),
    [
        Input(component_id='yearmon-selector', component_property='value')
    ]
)
def load_season_points_graph(FISCAL_YEARMO):
    results = get_match_results(FISCAL_YEARMO)

    figure = []
    if len(results) > 0:
        figure = draw_date_sales_graph(results)

    return figure


# In[ ]:


# start Flask server
app.css.append_css({

    "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"

})


if __name__ == '__main__':

    app.run_server(

        debug=False

    )


# In[ ]:


# reference-- 1. coding part -- https://alysivji.github.io/reactive-dashboards-with-dash.html
#          -- 2. Dash lib documentation -- https://dash.plot.ly/installation  


# In[8]:


# About plotly.graph_objs.Figure function
go.Figure(
        data=[
            go.Scatter(x=[1,2,3], y=[1,4,9], mode='lines+markers', marker = dict(
        color = '#FFBAD2'))
        ],
        layout=go.Layout(
            title='Sales Trend',
            showlegend=False
        )
    )


# In[10]:


import numpy as np
go.Figure(
data=[go.Scatter(
    y = np.random.randn(500),
    mode='markers',
    marker=dict(
        size=16,
        color = np.random.randn(500), #set color equal to a variable
        colorscale='Viridis',
        showscale=True
    )
)],
        layout=go.Layout(
            title='Figure 1',
            showlegend=False
    
))

