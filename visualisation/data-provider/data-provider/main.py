from http.client import HTTPException
from fastapi import FastAPI, Response, HTTPException, Query, Path

import urllib.request
import os
import pandas as pd
from pandas.core.frame import DataFrame

pd.options.mode.chained_assignment = None  # default='warn'
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from .polish_data import get_gov_data, get_szczepienia_data, get_powiat_present_data, get_wojewodztwa_data
from .owid_data import get_df_full, get_df_small, get_df_with_columns
# from .schemas.polish import DataFrameSchema

import logging
LOGGER = logging.getLogger('StateHandler')

df = get_df_full()
df_small = get_df_small()
pol_gov_data = get_gov_data()
odsetek_data = get_szczepienia_data()
wojew_data = get_wojewodztwa_data()

dict = {'df': df, 'df_small': df_small, 'pol_data': pol_gov_data, 'comm_data': odsetek_data, 'prov_data': wojew_data}
# powiat_data = get_powiat_present_data()

# poland_df_clear = df_clear[(df_clear['location'] == 'Poland') | (df_clear['location'] == 'Germany')]
# poland_df_clear = poland_df_clear.append(df_clear[df_clear['location'] == 'Germany'])
# poland_df_clear = poland_df_clear[poland_df_clear['date'].dt.year == 2021]



from fastapi.responses import FileResponse
from typing import Optional
from .schemas.seaborn_schema import LinePlot
from .schemas.owid import OwidData
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU

app_provider = FastAPI(title="CovidEDA",
    # description=description,
    # version="0.0.1",
    # terms_of_service="http://example.com/terms/",
    contact={
        "name": "Tomasz Wawrykowicz",
        # "url": "http://x-force.example.com/contact/",
        "email": "246823@student.pwr.edu.pl",
    },)

@app_provider.post('/lineplot')
async def lineplot(parameters: LinePlot):
    chart = parameters.chart_name + '.png'
    if os.path.isfile(chart):
        # os.remove(chart_name)
        raise HTTPException(status_code=404, detail='Chart with that name already exist')
    if parameters.data_source == 'poland_df_clear':
            dataframe = poland_df_clear
    plt.clf()
    ax = sns.lineplot(data=dataframe, x=parameters.x, y=parameters.y, hue=parameters.hue)
    # ax = sns.lineplot(parameters)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    fig = ax.get_figure()
    plt.figure(figsize=(24, 13.5), dpi=80)
    fig.savefig(chart)
    print(chart)
    return FileResponse(chart)

import re

@app_provider.post('/boxplot')
async def boxplot(*, columns: list = Query([]), q: list = Query([]), data_source: Optional[str] = None):
    # print(**q)
    if data_source in dict.keys():
        dataframe = dict.get(data_source)
    else:
        raise HTTPException(status_code=404, detail='Please, select correctly data source')
    if len(columns) != 0:
        dataframe = dataframe[columns]
    for param in q:
        date_from = re.match(r'(?P<column>.*) > (?P<value>.*)', param)
        if date_from:
            date_from_datetype = dt.datetime.strptime(date_from.group('value'), '%Y-%m-%d')
            dataframe = dataframe[dataframe['date'] > date_from_datetype]
            if dataframe.size == 0:
                raise HTTPException(status_code=404, detail='Ups, your query is wrong (date_from)')
        date_to = re.match(r'(?P<column>.*) < (?P<value>.*)', param)
        if date_to:
            date_to_datetype = dt.datetime.strptime(date_to.group('value'), '%Y-%m-%d')
            dataframe = dataframe[dataframe['date'] < date_to_datetype]
            if dataframe.size == 0:
                raise HTTPException(status_code=404, detail='Ups, your query is wrong (date_to)')
        in_list = re.match(r'(?P<column>.*) in (?P<value>.*)', param)
        if in_list:
            dataframe = dataframe[dataframe[in_list.group('column')].isin(in_list.group('value').split(" "))]
            if dataframe.size == 0:
                raise HTTPException(status_code=404, detail='Ups, your query is wrong (in)')
        is_value = re.match(r'(?P<column>.*) = (?P<value>.*)', param)
        if is_value:
            dataframe = dataframe[dataframe[is_value.group('column')] == is_value.group('value')]
            if dataframe.size == 0:
                raise HTTPException(status_code=404, detail='Ups, your query is wrong (=)')

    print(dataframe.info())
    query_items = {"q": q}
    return query_items


@app_provider.post('/histplot')
async def histplot(parameters: LinePlot, chart_name: str, data_source: Optional[str] = None, owid_source: Optional[OwidData] = None):  # , data_frame: Optional[DataFrameSchema] = None):
    # if data_frame and data_source:
        # raise HTTPException(status_code=404, detail='Please, select one: DataFrame or DataSource')
    chart = chart_name + '.png'
    if os.path.isfile(chart):
        # os.remove(chart_name)
        raise HTTPException(status_code=404, detail='Chart with that name already exist')
    if data_source in dict.keys():
        dataframe = dict.get(data_source)
    for param in owid_source:
        if param:
            dataframe

    plt.clf()
    ax = sns.lineplot(data=dataframe, **parameters.dict())
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    fig = ax.get_figure()
    plt.figure(figsize=(24, 13.5), dpi=80)
    fig.savefig(chart)
    print(chart)
    return FileResponse(chart)

@app_provider.post('/scatterplot')
async def scatterplot():
    return ':)'

@app_provider.post('/jointplot')
async def jointplot():
    return ':)'

@app_provider.post('/heatmap')
async def heatmap():
    return ':)'


from .database import engine, Base
Base.metadata.create_all(engine)

from sqlalchemy import inspect
inspector = inspect(engine)

if not inspector.has_table(engine, 'owid'):
    LOGGER.info('no chyba nmie dziala')
    print('no chyba nie dziala')
    # df.to_sql('owid', con=engine, if_exists='replace', index=False)
from sqlalchemy.orm import Session
from .dependencies import get_db
from fastapi import Depends

@app_provider.get('/dataframe')
def get_users(db: Session = Depends(get_db)):
    db_data = pd.read_sql_query(''' select sum(w3_zaszczepieni_pelna_dawka) from poland_vacc where dane_na_dzien = '2021-11-24' ''', engine)
    # db_data = db.query('table_owid').offset(0).all()
    #     # db_data = engine.execute('select * from table_owid').fetchall()
    #     db_data = DataDbTools(db)
    #     data = db_data.get_from()
    
    print(db_data.info())
    return db_data

@app_provider.post('/sql_query')
def set_dataframe_by_sql_query(condition: str):
    
    df = pd.read_sql_query(f''' {condition} ''', engine)
    return df


@app_provider.get('/{chart_name}')
def get_chart_by_name(chart_name: str):
    chart = chart_name + '.png'
    if not os.path.isfile(chart):
        # os.remove(chart_name)
        raise HTTPException(status_code=404, detail='Chart with that name does not exist')
    return FileResponse(chart)


# df = get_gov_data()
# df_szcza = get_szczepienia_data()
# df_szcza = df_szcza.loc[:1, 'gmina_nazwa']
# print(df_szcza.to_json())
# df_szcza = df_szcza.to_html()

if not inspector.has_table(engine, 'poland_vacc'):
    LOGGER.info('no chyba dziala')
    print('no chyba dziala')
    # df_szcza.to_sql('poland_vacc', con=engine, if_exists='replace', index=False)

# print(df_szcz)

# from fastapi.responses import JSONResponse, HTMLResponse
# import logging
# LOGGER = logging.getLogger('StateHandler')
# df_szcz = get_szczepienia_data()
# df_szcz = df_szcz.iloc[:4, :3]
# html_szcz = df_szcz.to_json()


# @app_provider.get('/items', response_class=JSONResponse)
# async def get_szczepienia():
    # result = html_szcz
    # return result


