# -*- coding: utf-8 -*-

import flask
import dash
import dash_table
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input, State

import os
import pathlib
import re
import json
from datetime import datetime
import string
import matplotlib.colors as mcolors
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from dateutil import relativedelta
from operator import add

#ngram_df = pd.read_csv("source/ngram_counts_data.csv", index_col=0)

DATA_PATH = pathlib.Path(__file__).parent.resolve()
EXTERNAL_STYLESHEETS = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
FILENAME = "news_final.csv"
PLOTLY_LOGO = "https://www.isb.edu/content/dam/sites/isb/about-isb/ISB_transparent_logo.png"
GLOBAL_DF = pd.read_csv(DATA_PATH.joinpath(FILENAME), header=0)
"""
We are casting the whole column to datetime to make life easier in the rest of the code.
It isn't a terribly expensive operation so for the sake of tidyness we went this way.
"""
GLOBAL_DF["Date"] = pd.to_datetime(
    GLOBAL_DF["Date"], format="%d/%m/%y"
)

"""
Precomputing senti values
"""
themel2=['aadhaar_based_subsidies',
	'aadhaar_based_schemes',
	'digital_stack',
	'enrolment_process',
	'e-governance',
	'financial_inclusion',
	'macroeconomic_policy',
	'data_security',
	'aadhaar_political_debates',
	'judiciary_right_to_privacy',
	'money_laundering',
	'crime']


"""
#  Somewhat helpful functions
"""


def sample_data(dataframe, float_percent):
    """
    Returns a subset of the provided dataframe.
    The sampling is evenly distributed and reproducible
    """
    print("making a local_df data sample with float_percent: %s" % (float_percent))
    return dataframe.sample(frac=float_percent, random_state=1)

def get_complaint_count_by_theme(dataframe):
    """ Helper function to get complaint counts for unique themes """
    theme_counts = dataframe["Theme"].value_counts()
    # we filter out all themes with less than 11 complaints for now
    theme_counts = theme_counts[theme_counts > 10]
    values = theme_counts.keys().tolist()
    counts = theme_counts.tolist()
    check=0
    for count in counts:
    	check+=count
    counts.append(check)
    values.append("all")
    return values, counts

def calculate_themes_sample_data(dataframe, sample_size, time_values):
    """ TODO """
    print(
        "making themes_sample_data with sample_size count: %s and time_values: %s"
        % (sample_size, time_values)
    )
    if time_values is not None:
        min_date = time_values[0]
        max_date = time_values[1]
        dataframe = dataframe[
            (dataframe["Date"] >= min_date)
            & (dataframe["Date"] <= max_date)
        ]

    #return values
    theme_counts = dataframe["Theme"].value_counts()
    theme_counts_sample = theme_counts[:sample_size]
    values_sample = theme_counts_sample.keys().tolist()
    counts_sample={}
    counts_sample["pos"]=[]
    counts_sample["neu"]=[]
    counts_sample["neg"]=[]

    for theme in values_sample:
#        print(theme)
        tdf=dataframe[(dataframe.Theme==theme)]
        senti_counts=tdf["Sentiment"].value_counts()
        senti_sample=senti_counts[:sample_size]
#        print(senti_sample)
        senti_values=senti_sample.keys().tolist()
#        print(senti_values)
        for val in senti_values:
            if val.startswith("pos"):
                counts_sample["pos"].append(senti_sample[val])
            elif val.startswith("neu"):
                counts_sample["neu"].append(senti_sample[val])
            elif val.startswith("neg"):
                counts_sample["neg"].append(senti_sample[val])
#        print(counts_sample)

#    print(counts_sample)

    return values_sample, counts_sample#, senti_sample

def calculate_themes_per_paper_data(dataframe, sample_size, time_values):
    """ TODO """
    print(
        "making themes_sample_data with sample_size count: %s and time_values: %s"
        % (sample_size, time_values)
    )
    if time_values is not None:
        min_date = time_values[0]
        max_date = time_values[1]
        dataframe = dataframe[
            (dataframe["Date"] >= min_date)
            & (dataframe["Date"] <= max_date)
        ]

    #return values
    paper_counts = dataframe["Newspaper"].value_counts()
    paper_counts_sample = paper_counts[:sample_size]
    values_sample = paper_counts_sample.keys().tolist()
    counts_sample={}
    for theme in themel2:
        counts_sample[theme]=[0,0,0,0,0,0,0,0,0,0]

    pc=0
    for paper in values_sample:
#        print(theme)
        pdf=dataframe[(dataframe.Newspaper==paper)]
        theme_counts=pdf["Theme"].value_counts()
        theme_sample=theme_counts[:sample_size]
        theme_values=theme_sample.keys().tolist()
        c=0
        for theme in theme_values:
            if c==0:
                counts_sample[theme][pc]=70
                c+=1
            elif c==1:
                counts_sample[theme][pc]=30
                c+=1
            else:
                counts_sample[theme][pc]=0
        pc+=1

#    print(counts_sample)

    return values_sample, counts_sample#, senti_sample

def calculate_papers_per_theme_data(dataframe, sample_size, time_values):
    """ TODO """
    print(
        "making themes_sample_data with sample_size count: %s and time_values: %s"
        % (sample_size, time_values)
    )
    if time_values is not None:
        min_date = time_values[0]
        max_date = time_values[1]
        dataframe = dataframe[
            (dataframe["Date"] >= min_date)
            & (dataframe["Date"] <= max_date)
        ]

    #return values
    paperl=["Business Line","Business Std","Hindu","Hindustan Times","Indian Express","Mint News","Times of India","Pioneer","The Economic Times","Firstpost"]

    paper_counts = dataframe["Theme"].value_counts()
    paper_counts_sample = paper_counts[:sample_size]
    values_sample = paper_counts_sample.keys().tolist()
    counts_sample={}
    for paper in paperl:
        counts_sample[paper]=[0,0,0,0,0,0,0,0,0,0,0,0]

    pc=0
    for paper in values_sample:
#        print(theme)
        pdf=dataframe[(dataframe.Theme==paper)]
        theme_counts=pdf["Newspaper"].value_counts()
        theme_sample=theme_counts[:sample_size]
        theme_values=theme_sample.keys().tolist()
        c=0
        for theme in theme_values:
            if c==0:
                counts_sample[theme][pc]=70
                c+=1
            elif c==1:
                counts_sample[theme][pc]=30
                c+=1
            else:
                counts_sample[theme][pc]=0
        pc+=1

#    print(counts_sample)

    return values_sample, counts_sample#, senti_sample


def make_local_df(selected_themes, time_values, n_selection):
    """ TODO """
    print("redrawing wordcloud...")
    n_float = float(n_selection / 100)
    print("got time window:", str(time_values))
    print("got n_selection:", str(n_selection), str(n_float))
    # sample the dataset according to the slider
    local_df = sample_data(GLOBAL_DF, n_float)
    if time_values is not None:
        time_values = time_slider_to_date(time_values)
        local_df = local_df[
            (local_df["Date"] >= time_values[0])
            & (local_df["Date"] <= time_values[1])
        ]
    if selected_themes:
        local_df = local_df[local_df["Theme"] == selected_themes]
        #add_stopwords(selected_themes)
    return local_df

def make_marks_time_slider(mini, maxi):
    """
    A helper function to generate a dictionary that should look something like:
    {1420066800: '2015', 1427839200: 'Q2', 1435701600: 'Q3', 1443650400: 'Q4',
    1451602800: '2016', 1459461600: 'Q2', 1467324000: 'Q3', 1475272800: 'Q4',
     1483225200: '2017', 1490997600: 'Q2', 1498860000: 'Q3', 1506808800: 'Q4'}
    """
    step = relativedelta.relativedelta(months=+12)
    start = datetime(year=2012, month=1, day=1)
    #thirty=[4,6,9,11]
    #if maxi.month==2:
    end = datetime(year=2020, month=2, day=28)
    #elif maxi.month in thirty:
    #	end = datetime(year=maxi.year, month=maxi.month, day=30)
    #else:
    #	end = datetime(year=maxi.year, month=maxi.month, day=31)

    ret = {}

    current = start
    while current <= end:
        current_str = int(current.timestamp())
        ret[current_str] = {
            "label": str(current.year),
            "style": {"font-weight": "bold", "font-size": 7},
        }
        current += step
    # print(ret)
    return ret

def time_slider_to_date(time_values):
    """ TODO """
    min_date = datetime.fromtimestamp(time_values[0]).strftime("%c")
    max_date = datetime.fromtimestamp(time_values[1]).strftime("%c")
    print("Converted time_values: ")
    print("\tmin_date:", time_values[0], "to: ", min_date)
    print("\tmax_date:", time_values[1], "to: ", max_date)
    return [min_date, max_date]


"""
#  Page layout and contents

In an effort to clean up the code a bit, we decided to break it apart into
sections. For instance: LEFT_COLUMN is the input controls you see in that gray
box on the top left. The body variable is the overall structure which most other
sections go into. This just makes it ever so slightly easier to find the right
spot to add to or change without having to count too many brackets.
"""

NAVBAR = dbc.Navbar(
    children=[
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(src=PLOTLY_LOGO, height="30px")),
                    dbc.Col(
                        dbc.NavbarBrand("Aadhaar in the News", className="ml-2")
                    ),
                ],
                align="center",
                no_gutters=True,
            ),
        )
    ],
    color="light",
    dark=False,
    sticky="top",
)

LEFT_COLUMN = dbc.Jumbotron(
    [
        html.H4(children="Select theme & dataset size", className="display-5"),
        html.Hr(className="my-2"),
        html.Label("Select percentage of dataset", className="lead"),
        html.P(
            "(Lower is faster. Higher is more precise)",
            style={"fontSize": 10, "font-weight": "lighter"},
        ),
        dcc.Slider(
            id="n-selection-slider",
            min=1,
            max=100,
            step=1,
            marks={
                0: "0%",
                10: "",
                20: "20%",
                30: "",
                40: "40%",
                50: "",
                60: "60%",
                70: "",
                80: "80%",
                90: "",
                100: "100%",
            },
            value=100,
        ),
        html.Label("Select a theme", style={"marginTop": 15}, className="lead"),
        html.P(
            "(You can use the dropdown or click the barchart on the right)",
            style={"fontSize": 10, "font-weight": "lighter"},
        ),
        dcc.Dropdown(
            id="themes-drop", clearable=False, style={"marginBottom": 15, "font-size": 12}
        ),
        html.Label("Select time frame", className="lead"),
        html.Div(dcc.RangeSlider(id="time-window-slider"), style={"marginBottom": 0}),
    ]
)

LEFT_COLUMN_II = dbc.Jumbotron(
    [
        html.H4(children="Time Slider", className="display-5"),
        html.Label("Select time frame", className="lead"),
        html.Div(dcc.RangeSlider(id="time-window-slider-ii"), style={"marginBottom": 0}),
    ]
)

LEFT_COLUMN_III = dbc.Jumbotron(
    [
        html.H4(children="Time Slider", className="display-5"),
        html.Label("Select time frame", className="lead"),
        html.Div(dcc.RangeSlider(id="time-window-slider-iii"), style={"marginBottom": 0}),
    ]
)

TOP_THEMES_PLOT = [
    dbc.CardHeader(html.H5("Context of Aadhaar in news media")),
    dbc.CardBody(
        [
            dcc.Loading(
                id="loading-themes-hist",
                children=[
                    dbc.Alert(
                        "Not enough data to render this plot, please adjust the filters",
                        id="no-data-alert-themes",
                        color="warning",
                        style={"display": "none"},
                    ),
                    dcc.Graph(id="themes-sample"),
                ],
                type="default",
            )
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]

THEMES_PER_PAPER_PLOT = [
    dbc.CardHeader(html.H5("Top 2 themes per newspaper")),
    dbc.CardBody(
        [
            dcc.Loading(
                id="loading-themes-hist-ii",
                children=[
                    dbc.Alert(
                        "Not enough data to render this plot, please adjust the filters",
                        id="no-data-alert-themes-ii",
                        color="warning",
                        style={"display": "none"},
                    ),
                    dcc.Graph(id="themes-per-paper"),
                ],
                type="default",
            )
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]

PAPERS_PER_THEME_PLOT = [
    dbc.CardHeader(html.H5("Top 2 newspapers per theme")),
    dbc.CardBody(
        [
            dcc.Loading(
                id="loading-themes-hist-iii",
                children=[
                    dbc.Alert(
                        "Not enough data to render this plot, please adjust the filters",
                        id="no-data-alert-themes-iii",
                        color="warning",
                        style={"display": "none"},
                    ),
                    dcc.Graph(id="papers-per-theme"),
                ],
                type="default",
            )
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]

BODY = dbc.Container(
    [
#        dbc.Row([dbc.Col(dbc.Card(TOP_NGRAM_PLOT)),], style={"marginTop": 30}),
        dbc.Row(
            [
                dbc.Col(LEFT_COLUMN, md=4, align="center"),
                dbc.Col(dbc.Card(TOP_THEMES_PLOT), md=8),
            ],
            style={"marginTop": 30},
        ),
        dbc.Row(
            [
                dbc.Col(LEFT_COLUMN_II, md=4, align="center"),
                dbc.Col(dbc.Card(THEMES_PER_PAPER_PLOT), md=8),
            ],
            style={"marginTop": 30},
        ),
        dbc.Row(
            [
                dbc.Col(LEFT_COLUMN_III, md=4, align="center"),
                dbc.Col(dbc.Card(PAPERS_PER_THEME_PLOT), md=8),
            ],
            style={"marginTop": 30},
        ),
#        dbc.Row([dbc.Col([dbc.Card(LDA_PLOTS)])], style={"marginTop": 50}),
    ],
    className="mt-12",
)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # for Heroku deployment

app.layout = html.Div(children=[NAVBAR, BODY])

"""
#  Callbacks
"""

@app.callback(
    [
        Output("time-window-slider", "marks"),
        Output("time-window-slider", "min"),
        Output("time-window-slider", "max"),
        Output("time-window-slider", "step"),
        Output("time-window-slider", "value"),
    ],
    [Input("n-selection-slider", "value")],
)
def populate_time_slider(value):
    """
    Depending on our dataset, we need to populate the time-slider
    with different ranges. This function does that and returns the
    needed data to the time-window-slider.
    """
    value += 0
    min_date = GLOBAL_DF["Date"].min()
    max_date = GLOBAL_DF["Date"].max()

    print(min_date)
    print(max_date)

    marks = make_marks_time_slider(min_date, max_date)
    min_epoch = list(marks.keys())[0]
    max_epoch = list(marks.keys())[-1]

    return (
        marks,
        min_epoch,
        max_epoch,
        (max_epoch - min_epoch) / (len(list(marks.keys())) * 3),
        [min_epoch, max_epoch],
    )

@app.callback(
    [Output("themes-sample", "figure"), Output("no-data-alert-themes", "style")],
    [Input("n-selection-slider", "value"), Input("time-window-slider", "value")],
)
def update_themes_sample_plot(n_value, time_values):
    """ TODO """
    print("redrawing sample...")
    print("\tn is:", n_value)
    print("\ttime_values is:", time_values)
    if time_values is None:
        return [{}, {"display": "block"}]
    n_float = float(n_value / 100)
    themes_sample_count = 10
    local_df = sample_data(GLOBAL_DF, n_float)
    min_date, max_date = time_slider_to_date(time_values)
    values_sample, counts_sample = calculate_themes_sample_data(
        local_df, themes_sample_count, [min_date, max_date]
    )
    data = [
        {
            "x": values_sample,
            "y": counts_sample["pos"],
            #"text": values_sample,
            #"textposition": "auto",
            "type": "bar",
            "name": "Positive",
            "marker": {"color" : '#044600'}
        },
        {
            "x": values_sample,
            "y": counts_sample["neu"],
            #"text": values_sample,
            #"textposition": "auto",
            "type": "bar",
            "name": "Neutral",
            "marker": {"color" : '#00E5F0'}
        },
        {
            "x": values_sample,
            "y": counts_sample["neg"],
            #"text": values_sample,
            #"textposition": "auto",
            "type": "bar",
            "name": "Negative",
            "marker": {"color" : '#821600'}
        },
    ]
    boldl=[]
    for value in values_sample:
    	value=re.sub(r"_"," ",value)
    	value=value.title()
    	value="<b>"+value+"</b>"
    	boldl.append(value)
    layout = {
        "autosize": True,
        #"margin": dict(t=10, b=0, l=40, r=0, pad=4),
        "barmode": 'stack',
        "xaxis": {"showticklabels": True,
                'automargin': True,
                'ticktext': boldl,
                'tickvals': values_sample,
                'title': 'Themes',
                #'nticks': 10,
                'tickangle': 30,
            },
    }
    print("redrawing themes-sample...done")
    return [{"data": data, "layout": layout}, {"display": "none"}]

@app.callback(
    [
        Output("time-window-slider-ii", "marks"),
        Output("time-window-slider-ii", "min"),
        Output("time-window-slider-ii", "max"),
        Output("time-window-slider-ii", "step"),
        Output("time-window-slider-ii", "value"),
    ],
    [Input("n-selection-slider", "value")],
)
def populate_time_slider_ii(value):
    """
    Depending on our dataset, we need to populate the time-slider
    with different ranges. This function does that and returns the
    needed data to the time-window-slider.
    """
    value += 0
    min_date = GLOBAL_DF["Date"].min()
    max_date = GLOBAL_DF["Date"].max()

    print(min_date)
    print(max_date)

    marks = make_marks_time_slider(min_date, max_date)
    min_epoch = list(marks.keys())[0]
    max_epoch = list(marks.keys())[-1]

    return (
        marks,
        min_epoch,
        max_epoch,
        (max_epoch - min_epoch) / (len(list(marks.keys())) * 3),
        [min_epoch, max_epoch],
    )

@app.callback(
    [Output("themes-per-paper", "figure"), Output("no-data-alert-themes-ii", "style")],
    [Input("n-selection-slider", "value"), Input("time-window-slider-ii", "value")],
)
def update_themes_per_paper_plot(n_value, time_values):
    """ TODO """
#    print("redrawing sample...")
#    print("\tn is:", n_value)
#    print("\ttime_values is:", time_values)
    if time_values is None:
        return [{}, {"display": "block"}]
    n_float = float(n_value / 100)
    themes_sample_count = 10
    local_df = sample_data(GLOBAL_DF, n_float)
    min_date, max_date = time_slider_to_date(time_values)
    values_sample, counts_sample = calculate_themes_per_paper_data(
        local_df, themes_sample_count, [min_date, max_date]
    )
    data = [
        {
            "x": values_sample,
            "y": counts_sample["aadhaar_based_subsidies"],
            #"text": values_sample,
            #"textposition": "auto",
            "type": "bar",
            "name": "aadhaar_based_subsidies",
             "marker": {"color" : '#B28610'}
        },
        {
            "x": values_sample,
            "y": counts_sample["aadhaar_based_schemes"],
            #"text": values_sample,
            #"textposition": "auto",
            "type": "bar",
            "name": "aadhaar_based_schemes",
            "marker": {"color" : '#8286B0'}
        },
        {
            "x": values_sample,
            "y": counts_sample["digital_stack"],
            #"text": values_sample,
            #"textposition": "auto",
            "type": "bar",
            "name": "digital_stack",
        },
        {
            "x": values_sample,
            "y": counts_sample["enrolment_process"],
            #"text": values_sample,
            #"textposition": "auto",
            "type": "bar",
            "name": "enrolment_process",
        },
        {
            "x": values_sample,
            "y": counts_sample["e-governance"],
            #"text": values_sample,
            #"textposition": "auto",
            "type": "bar",
            "name": "e-governance",
        },
        {
            "x": values_sample,
            "y": counts_sample["financial_inclusion"],
            #"text": values_sample,
            #"textposition": "auto",
            "type": "bar",
            "name": "financial_inclusion",
        },
        {
            "x": values_sample,
            "y": counts_sample["macroeconomic_policy"],
            #"text": values_sample,
            #"textposition": "auto",
            "type": "bar",
            "name": "macroeconomic_policy",
        },
        {
            "x": values_sample,
            "y": counts_sample["data_security"],
            #"text": values_sample,
            #"textposition": "auto",
            "type": "bar",
            "name": "data_security",
        },
        {
            "x": values_sample,
            "y": counts_sample["aadhaar_political_debates"],
            #"text": values_sample,
            #"textposition": "auto",
            "type": "bar",
            "name": "aadhaar_political_debates",
        },
        {
            "x": values_sample,
            "y": counts_sample["judiciary_right_to_privacy"],
            #"text": values_sample,
            #"textposition": "auto",
            "type": "bar",
            "name": "judiciary_right_to_privacy",
        },
        {
            "x": values_sample,
            "y": counts_sample["money_laundering"],
            #"text": values_sample,
            #"textposition": "auto",
            "type": "bar",
            "name": "money_laundering",
        },
        {
            "x": values_sample,
            "y": counts_sample["crime"],
            #"text": values_sample,
            #"textposition": "auto",
            "type": "bar",
            "name": "crime",
            "marker": {"color" : '#821600'}
        },
    ]
    boldl=[]
    for value in values_sample:
    	value=re.sub(r"_"," ",value)
    	value=value.title()
    	value="<b>"+value+"</b>"
    	boldl.append(value)
    layout = {
        "autosize": True,
        #"margin": dict(t=10, b=0, l=40, r=0, pad=4),
        "barmode": 'stack',
        "xaxis": {"showticklabels": True,
                'automargin': True,
                'ticktext': boldl,
                'tickvals': values_sample,
                'title': 'Newspaper',
                #'nticks': 10,
                'tickangle': 30,
            },
        "yaxis": {"showticklabels": False},
        "hovermode": False
    }
    print("redrawing themes-sample...done")
    return [{"data": data, "layout": layout}, {"display": "none"}]

@app.callback(
    [
        Output("time-window-slider-iii", "marks"),
        Output("time-window-slider-iii", "min"),
        Output("time-window-slider-iii", "max"),
        Output("time-window-slider-iii", "step"),
        Output("time-window-slider-iii", "value"),
    ],
    [Input("n-selection-slider", "value")],
)
def populate_time_slider_iii(value):
    """
    Depending on our dataset, we need to populate the time-slider
    with different ranges. This function does that and returns the
    needed data to the time-window-slider.
    """
    value += 0
    min_date = GLOBAL_DF["Date"].min()
    max_date = GLOBAL_DF["Date"].max()

    print(min_date)
    print(max_date)

    marks = make_marks_time_slider(min_date, max_date)
    min_epoch = list(marks.keys())[0]
    max_epoch = list(marks.keys())[-1]

    return (
        marks,
        min_epoch,
        max_epoch,
        (max_epoch - min_epoch) / (len(list(marks.keys())) * 3),
        [min_epoch, max_epoch],
    )

@app.callback(
    [Output("papers-per-theme", "figure"), Output("no-data-alert-themes-iii", "style")],
    [Input("n-selection-slider", "value"), Input("time-window-slider-iii", "value")],
)
def update_themes_per_paper_plot(n_value, time_values):
    """ TODO """
#    print("redrawing sample...")
#    print("\tn is:", n_value)
#    print("\ttime_values is:", time_values)
    if time_values is None:
        return [{}, {"display": "block"}]
    n_float = float(n_value / 100)
    themes_sample_count = 10
    local_df = sample_data(GLOBAL_DF, n_float)
    min_date, max_date = time_slider_to_date(time_values)
    values_sample, counts_sample = calculate_papers_per_theme_data(
        local_df, themes_sample_count, [min_date, max_date]
    )
    data = [
        {
            "x": values_sample,
            "y": counts_sample["Business Line"],
            #"text": values_sample,
            #"textposition": "auto",
            "type": "bar",
            "name": "Business Line",
        },
        {
            "x": values_sample,
            "y": counts_sample["Business Std"],
            #"text": values_sample,
            #"textposition": "auto",
            "type": "bar",
            "name": "Business Std",
        },
        {
            "x": values_sample,
            "y": counts_sample["Hindu"],
            #"text": values_sample,
            #"textposition": "auto",
            "type": "bar",
            "name": "Hindu",
        },
        {
            "x": values_sample,
            "y": counts_sample["Hindustan Times"],
            #"text": values_sample,
            #"textposition": "auto",
            "type": "bar",
            "name": "Hindustan Times",
        },
        {
            "x": values_sample,
            "y": counts_sample["Indian Express"],
            #"text": values_sample,
            #"textposition": "auto",
            "type": "bar",
            "name": "Indian Express",
        },
        {
            "x": values_sample,
            "y": counts_sample["Mint News"],
            #"text": values_sample,
            #"textposition": "auto",
            "type": "bar",
            "name": "Mint News",
        },
        {
            "x": values_sample,
            "y": counts_sample["Times of India"],
            #"text": values_sample,
            #"textposition": "auto",
            "type": "bar",
            "name": "Times of India",
        },
        {
            "x": values_sample,
            "y": counts_sample["Pioneer"],
            #"text": values_sample,
            #"textposition": "auto",
            "type": "bar",
            "name": "Pioneer",
        },
        {
            "x": values_sample,
            "y": counts_sample["The Economic Times"],
            #"text": values_sample,
            #"textposition": "auto",
            "type": "bar",
            "name": "The Economic Times",
        },
        {
            "x": values_sample,
            "y": counts_sample["Firstpost"],
            #"text": values_sample,
            #"textposition": "auto",
            "type": "bar",
            "name": "Firstpost",
        },
    ]
    boldl=[]
    for value in values_sample:
        value=re.sub(r"_"," ",value)
        value=value.title()
        value="<b>"+value+"</b>"
        boldl.append(value)
    layout = {
        "autosize": True,
        #"margin": dict(t=10, b=0, l=40, r=0, pad=4),
        "barmode": 'stack',
        "xaxis": {"showticklabels": True,
                'automargin': True,
                'ticktext': boldl,
                'tickvals': values_sample,
                'title': 'Newspaper',
                #'nticks': 10,
                'tickangle': 30,
            },
        "yaxis": {"showticklabels": False},
        "hovermode": False
    }
    print("redrawing themes-sample...done")
    return [{"data": data, "layout": layout}, {"display": "none"}]

if __name__ == "__main__":
    app.run_server(debug=True, threaded=True, host='127.0.0.3')
