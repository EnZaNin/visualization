import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import logging

from http.client import HTTPException
import re
from fastapi import FastAPI, Response, HTTPException, Query, Path

import urllib.request
import os
import pandas as pd
from pandas.core.frame import DataFrame

from .polish_data import get_district_stats_data, get_district_vacc_data, get_powiat_present_data, \
    get_pol_df_by_condition, get_province_full
from .owid_data import get_owid_full, get_owid_small, get_owid_df_with_columns
from .methods import get_filtered_data, auto_scaler_date, pca_method

pd.options.mode.chained_assignment = None  # default='warn'

logging.basicConfig(
    level=20,
    format="%(name)-15s %(asctime)-15s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

df = get_owid_full()
df_small = get_owid_small()
district_stats_data = get_district_stats_data()
district_vacc_data = get_district_vacc_data()
province_full_data = get_province_full()

dict_owid = {'owid_full': df, 'owid_small': df_small}
dict_polish = {'district_stats': district_stats_data, 'district_vacc': district_vacc_data,
               'prov_data': province_full_data}
dict = {**dict_owid, **dict_polish}

# poland_df_clear = df_clear[(df_clear['location'] == 'Poland') | (df_clear['location'] == 'Germany')]
# poland_df_clear = poland_df_clear.append(df_clear[df_clear['location'] == 'Germany'])
# poland_df_clear = poland_df_clear[poland_df_clear['date'].dt.year == 2021]


from fastapi.responses import FileResponse
from typing import Optional
from .schemas.seaborn_schema import LinePlot, BoxPlot, HistPlot
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
                       }, )


@app_provider.get('/df/{dataframe}')
async def get_columns(dataframe: str):
    if dataframe in dict.keys():
        dataframe = dict.get(dataframe)
        return list(dataframe.columns)
    else:
        raise HTTPException(status_code=404, detail='Dataframe not found')


@app_provider.post('/lineplot_hs', summary="Create lineplot for owid data")
async def lineplot_owid(*, data_source: str, parameters: LinePlot, columns: list = Query([]),
                        conditions: list = Query([]), ):
    if data_source in dict_owid.keys():
        dataframe = dict_owid.get(data_source)
    else:
        raise HTTPException(status_code=404, detail='Please, select correctly data source')
    if len(columns) != 0:
        dataframe = dataframe[columns]
    dataframe = get_owid_df_with_columns(dataframe=dataframe, query=conditions)
    chart = parameters.chart_name + '.png'
    # if os.path.isfile(chart):
    # os.remove(chart_name)
    # raise HTTPException(status_code=404, detail='Chart with that name already exist')
    plt.clf()
    try:
        ax = sns.lineplot(data=dataframe, x=parameters.x, y=parameters.y, hue=parameters.hue)
    except ValueError:
        raise HTTPException(status_code=404, detail='Ups, something was wrong, chceck your parameters')
    # ax = sns.lineplot(parameters)
    # ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=30)
    # y_max = dataframe[parameters.y].max()
    # ax.set_ylim(top=1.1*y_max)
    fig = ax.get_figure()
    plt.figure(figsize=(24, 13.5), dpi=80)
    fig.savefig(chart)
    # print(chart)
    return FileResponse(chart)


@app_provider.post('/lineplot', summary="Create lineplot for selected data")
async def lineplot(*, parameters: LinePlot, columns: list = Query([]), conditions: list = Query([]),
                   data_source: Optional[str] = None, chart_name: Optional[str] = None):
    plt.clf()
    if not data_source:
        dataframe = province_full_data
        dataframe = dataframe[
            dataframe['wojewodztwo'].isin(['dolnośląskie', 'opolskie', 'wielkopolskie', 'mazowieckie'])]
        ax = sns.lineplot(data=dataframe, x='stan_rekordu_na', y='liczba_przypadkow', hue='wojewodztwo')
        chart = 'Wykres liniowy dla województw'
        ax.set_xlabel('Stan rekordu na', fontsize=15)
        ax.set_ylabel('Liczba przypadków', fontsize=15)
        ax.set_title(chart, fontsize=20)
        chart = chart + '.png'
        min_date = dataframe['stan_rekordu_na'].min()
        max_date = dataframe['stan_rekordu_na'].max()
        auto_scaler_date(min_date=min_date, max_date=max_date, ax=ax)
        fig = ax.get_figure()
        plt.xticks(rotation=30)
        plt.figure(figsize=(24, 13.5), dpi=80)
        fig.savefig(chart)
        return FileResponse(chart)
    elif data_source in dict.keys():
        dataframe = dict.get(data_source)
    else:
        raise HTTPException(status_code=404, detail='Please, select correctly data source')
    dataframe = get_filtered_data(df=dataframe, columns=columns, query=conditions)
    if not chart_name:
        raise HTTPException(status_code=404, detail='Please, give a name for chart')
    chart = chart_name + '.png'
    if os.path.isfile(chart):
        raise HTTPException(status_code=404, detail='Chart with that name already exist')
    try:
        ax = sns.lineplot(data=dataframe, **parameters.dict())
    except ValueError:
        raise HTTPException(status_code=404, detail='Ups, something was wrong, chceck your parameters')
    if parameters.x == 'date':
        min_date = dataframe['date'].min()
        max_date = dataframe['date'].max()
        auto_scaler_date(min_date=min_date, max_date=max_date, ax=ax)
    plt.xticks(rotation=30)
    fig = ax.get_figure()
    plt.figure(figsize=(24, 13.5), dpi=80)
    fig.savefig(chart)
    return FileResponse(chart)


@app_provider.post('/boxplot')
async def boxplot(*, params: BoxPlot, columns: list = Query([]), conditions: list = Query([]),
                  data_source: Optional[str] = None, chart_name: Optional[str] = None, xlabel: Optional[str] = None,
                  ylabel: Optional[str] = None, add_points: Optional[bool] = False):
    plt.clf()
    if not data_source:
        dataframe = province_full_data
        ax = sns.boxplot(data=dataframe, x='liczba_przypadkow', y='wojewodztwo', orient='h')
        chart = 'Wykres pudełkowy dla województw'
        ax.set_xlabel('Województwa', fontsize=15)
        ax.set_ylabel('Liczba przypadków', fontsize=15)
        ax.set_title(chart, fontsize=20)
        chart = chart + '.png'
        fig = ax.get_figure()
        plt.figure(figsize=(24, 13.5), dpi=80)
        fig.savefig(chart)
        return FileResponse(chart)
    elif data_source in dict.keys():
        dataframe = dict.get(data_source)
    else:
        raise HTTPException(status_code=404, detail='Please, select correctly data source')
    if not chart_name:
        raise HTTPException(status_code=404, detail='Please, give a name for chart')
    chart = chart_name + '.png'
    if os.path.isfile(chart):
        raise HTTPException(status_code=404, detail='Chart with that name already exist')
    dataframe = get_filtered_data(df=dataframe, columns=columns, query=conditions)

    plt.figure(figsize=(24, 13.5), dpi=80)
    try:
        ax = sns.boxplot(data=dataframe, **params.dict())
        if add_points:
            sns.stripplot(data=dataframe, **params.dict(), size=4, color=".3", linewidth=0)
    except ValueError:
        raise HTTPException(status_code=404, detail='Ups, something was wrong, chceck your parameters')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=15)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=15)
    plt.xticks(rotation=30)
    ax.set_title(chart_name, fontsize=20)
    fig = ax.get_figure()
    fig.savefig(chart)
    return FileResponse(chart)


@app_provider.post('/histplot')
async def histplot(*, params: HistPlot, columns: list = Query([]), conditions: list = Query([]),
                   data_source: Optional[str] = None, chart_name: Optional[str] = None, xlabel: Optional[str] = None,
                   ylabel: Optional[str] = None):
    # parameters: LinePlot, chart_name: str, data_source: Optional[str] = None,
    #            owid_source: Optional[OwidData] = None):  # , data_frame: Optional[DataFrameSchema] = None):
    plt.clf()
    if not data_source:
        dataframe = province_full_data
        ax = sns.histplot(
            [dataframe['liczba_testow_z_wynikiem_pozytywnym'], dataframe['liczba_testow_z_wynikiem_negatywnym']])
        chart = 'Histogram'
        ax.set_xlabel('Liczba przypadków', fontsize=15)
        ax.set_ylabel('Liczba', fontsize=15)
        ax.set_title(chart, fontsize=20)
        chart = chart + '.png'
        if os.path.isfile(chart):
            os.remove(chart)
        fig = ax.get_figure()
        plt.figure(figsize=(24, 13.5), dpi=80)
        fig.savefig(chart)
        return FileResponse(chart)
    elif data_source in dict.keys():
        dataframe = dict.get(data_source)
    else:
        raise HTTPException(status_code=404, detail='Please, select correctly data source')
    if not chart_name:
        raise HTTPException(status_code=404, detail='Please, give a name for chart')
    chart = chart_name + '.png'
    if os.path.isfile(chart):
        raise HTTPException(status_code=404, detail='Chart with that name already exist')
    dataframe = get_filtered_data(df=dataframe, columns=columns, query=conditions)
    plt.figure(figsize=(24, 13.5), dpi=80)
    try:
        ax = sns.histplot(data=dataframe, **params.dict())
    except ValueError:
        raise HTTPException(status_code=404, detail='Ups, something was wrong, chceck your parameters')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=15)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=15)
    fig = ax.get_figure()
    fig.savefig(chart)
    # print(chart)
    return FileResponse(chart)


@app_provider.get('/scatterplot')
def scatterplot():
    dataframe = province_full_data
    # print(dataframe.tail(7)['stan_rekordu_na'])
    dataframe = dataframe[dataframe['stan_rekordu_na'].isin(dataframe.tail(7)['stan_rekordu_na'])]
    # dataframe = dataframe.groupby(['wojewodztwo']).mean()
    # dataframe = dataframe.reset_index(level=['wojewodztwo'])
    ax = sns.scatterplot(data=dataframe, x='liczba_przypadkow',
                         y=dataframe['liczba_testow_z_wynikiem_negatywnym'] + dataframe[
                             'liczba_testow_z_wynikiem_pozytywnym'], hue='wojewodztwo')
    plt.figure(figsize=(24, 13.5), dpi=60)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    # try:
    #     ax = sns.histplot(data=dataframe, **params.dict())
    # except ValueError:
    #     raise HTTPException(status_code=404, detail='Ups, something was wrong, chceck your parameters')
    # if xlabel:
    #     ax.set_xlabel(xlabel, fontsize=15)
    # if ylabel:
    #     ax.set_ylabel(ylabel, fontsize=15)
    fig = ax.get_figure()
    chart = 'scatter.png'
    fig.savefig(chart)
    # print(chart)
    return FileResponse(chart)


@app_provider.post('/jointplot')
async def jointplot():
    return ':)'


@app_provider.post('/heatmap')
async def heatmap():
    return ':)'


@app_provider.post('/pca')
async def pca(*, columns: list = Query([]), conditions: list = Query([]),
              data_source: Optional[str] = None, chart_name: Optional[str] = None, filter: Optional[str] = None):
    # if not data_source:
    #     dataframe = df_small
    #     dataframe = dataframe[dataframe['location'].isin(
    #         ['Poland', 'Germany', 'Italy', 'Czechia', 'Russia', 'Slovakia', 'Ukraine', 'France'])]
    #     columns = ['total_cases', 'new_cases', 'total_deaths', 'new_deaths',
    #                'total_tests', 'new_tests', 'total_vaccinations',
    #                'people_vaccinated', 'people_fully_vaccinated', 'total_boosters', 'new_vaccinations']
    #     filter = 'location'
    #     chart_name = '2D_pca'
    #     pca_method(chart_name=chart_name, dataframe=dataframe, columns_for_pca=columns, column_filter=filter,
    #                n_components=2)
    #     return FileResponse(chart)
    return ':)'


# from .database import engine, Base
# Base.metadata.create_all(engine)

# from sqlalchemy import inspect
# inspector = inspect(engine)

# if not inspector.has_table(engine, 'owid'):
# LOGGER.info('no chyba nmie dziala')
# print('no chyba nie dziala')
# df.to_sql('owid', con=engine, if_exists='replace', index=False)
# from sqlalchemy.orm import Session
# from .dependencies import get_db
# from fastapi import Depends
from fastapi.responses import JSONResponse

# @app_provider.get('/dataframeee')
# def get_users(db: Session = Depends(get_db)):
# db_data = pd.read_sql_query(''' select sum(w3_zaszczepieni_pelna_dawka) from poland_vacc where dane_na_dzien = '2021-11-24' ''', engine)
# db_data = db.query('table_owid').offset(0).all()
#     # db_data = engine.execute('select * from table_owid').fetchall()
#     db_data = DataDbTools(db)
#     data = db_data.get_from()

# print(db_data.info())
# return db_data

# @app_provider.post('/sql_query')
# def set_dataframe_by_sql_query(condition: str):

# df = pd.read_sql_query(f''' {condition} ''', engine)
# return df
import json


@app_provider.get('/chart/{chart_name}')
def get_chart_by_name(chart_name: str):
    chart = chart_name + '.png'
    if not os.path.isfile(chart):
        raise HTTPException(status_code=404, detail='Chart with that name does not exist')
    return FileResponse(chart)


@app_provider.get('/dataframe')
def get_dataframe(dataframe: str, orient: Optional[str] = None):
    if dataframe in dict.keys():
        df = dict.get(dataframe)
        df = df.sample(10)
        result = df.to_json(orient=orient, date_format='iso')
        return json.loads(result)
    else:
        raise HTTPException(status_code=404, detail='Wrong name')

# df = get_gov_data()
# df_szcza = get_szczepienia_data()
# df_szcza = df_szcza.loc[:1, 'gmina_nazwa']
# print(df_szcza.to_json())
# df_szcza = df_szcza.to_html()

# if not inspector.has_table(engine, 'poland_vacc'):
# LOGGER.info('no chyba dziala')
# print('no chyba dziala')
# df_szcza.to_sql('poland_vacc', con=engine, if_exists='replace', index=False)


# from fastapi.responses import JSONResponse, HTMLResponse
# import logging
# LOGGER = logging.getLogger('StateHandler')
# df_szcz = get_szczepienia_data()
# df_szcz = df_szcz.iloc[:4, :3]
# html_szcz = df_szcz.to_json()
