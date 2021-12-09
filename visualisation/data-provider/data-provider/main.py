import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging

from http.client import HTTPException
import re
from fastapi import FastAPI, HTTPException, Query, Depends

import urllib.request
import os
import pandas as pd
import json
from matplotlib import ticker
from fastapi.responses import FileResponse
from typing import Optional, Dict, List
from .schemas.seaborn_schema import LinePlot, BoxPlot, HistPlot, ScatterPlot, JointPlot, RelPlot

from .polish_data import get_district_stats_data, get_district_vacc_data, get_province_full
from .owid_data import get_owid_full, get_owid_small
from .methods import get_filtered_data, auto_scaler_date, pca_method, mds_method

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

app_provider = FastAPI(title="CovidEDA",
                       # description=description,
                       # version="0.0.1",
                       # terms_of_service="http://example.com/terms/",
                       contact={
                           "name": "Tomasz Wawrykowicz",
                           # "url": "http://x-force.example.com/contact/",
                           "email": "246823@student.pwr.edu.pl",
                       }, )


@app_provider.get('/dataframe')
async def get_sample_from_dataframe(dataframe: str, orient: Optional[str] = None):
    if dataframe in dict.keys():
        dfr = dict.get(dataframe)
        dfr = dfr.sample(10).T
        result = dfr.to_json(orient=orient, date_format='iso')
        return json.loads(result)
    else:
        raise HTTPException(status_code=404, detail='Wrong name')


@app_provider.get('/df/{dataframe}')
async def get_columns(dataframe: str):
    if dataframe in dict.keys():
        dataframe = dict.get(dataframe)
        return list(dataframe.columns)
    else:
        raise HTTPException(status_code=404, detail='Dataframe not found')


@app_provider.get('/stats')
async def get_basics_stats(*, data_source: str, columns: list = Query([]), conditions: list = Query([])):
    if not data_source:
        dfr = dict.get(data_source)
        dfr_stats = dfr.describe().T
        dfr_stats['missing'] = dfr.isnull().sum()
        result = dfr_stats.T.to_json(date_format='iso')
        return json.loads(result)
    elif data_source in dict.keys():
        dataframe = dict.get(data_source)
    else:
        raise HTTPException(status_code=404, detail='Dataframe not found')
    dfr_stats_filtered = get_filtered_data(df=dataframe, columns=columns, query=conditions)
    dfr_stats = dfr_stats_filtered.describe().T
    dfr_stats['missing'] = dfr_stats_filtered.isnull().sum()
    # dfr_stats = dfr_stats.apply(np.ceil)
    dfr_stats = dfr_stats.apply(lambda x: round(x, 2))
    result = dfr_stats.T.to_json(date_format='iso')
    return json.loads(result)


def agg_dict(aggfunc: List[str] = Query([])):
    return list(map(json.loads, aggfunc))


@app_provider.get('/pivot')
async def get_pivot(*, columns: list = Query([]), conditions: list = Query([]), data_source: Optional[str] = None,
                    values: list = Query([]), index: list = Query([]), pivot_columns: list = Query([]),
                    agg_func: list = Depends(agg_dict)):
    if not data_source:
        for k, v in agg_func[0].items():
            print(k, v)
        dataframe = df_small
        dataframe = dataframe[dataframe['date'].isin(dataframe['date'].sample(10))]
        dataframe = dataframe[dataframe['location'].isin(dataframe['location'].sample(10))]
        pivo = pd.pivot_table(dataframe, values=['total_cases'], index=['location'], columns=['date'])
        result = pivo.to_json(date_format='iso')
        return json.loads(result)
    elif data_source in dict.keys():
        dataframe = dict.get(data_source)
    else:
        raise HTTPException(status_code=404, detail='Please, select correctly data source')
    dataframe = get_filtered_data(df=dataframe, columns=columns, query=conditions)
    agg = {}
    for idx, part in enumerate(agg_func):
        for k, v in agg_func[idx].items():
            if v == 'sum':
                agg[k] = np.sum
            elif v == 'mean':
                agg[k] = np.mean
    pivo = pd.pivot_table(dataframe, values=values, index=index, columns=pivot_columns, aggfunc=agg)
    result = pivo.to_json(date_format='iso')

    return json.loads(result)


@app_provider.get('/charts')
async def get_name_all_created_charts():
    files = os.listdir()
    chart_list = []
    for file in files:
        if file.endswith('.png'):
            chart_list.append(file)
    return chart_list


@app_provider.get('/chart/{chart_name}')
async def get_chart_by_name(chart_name: str):
    chart = chart_name + '.png'
    if not os.path.isfile(chart):
        raise HTTPException(status_code=404, detail='Chart with that name does not exist')
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
        raise HTTPException(status_code=404, detail='Ups, something was wrong, check your parameters')
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
    return FileResponse(chart)


@app_provider.post('/scatterplot')
def scatterplot(*, params: ScatterPlot, columns: list = Query([]), conditions: list = Query([]),
                data_source: Optional[str] = None, chart_name: Optional[str] = None, xlabel: Optional[str] = None,
                ylabel: Optional[str] = None):
    if not data_source:
        dataframe = province_full_data
        dataframe = dataframe[dataframe['stan_rekordu_na'].isin(dataframe.tail(7)['stan_rekordu_na'])]
        dataframe = dataframe.groupby(['wojewodztwo']).mean()
        dataframe = dataframe.reset_index(level=['wojewodztwo'])
        plt.figure(figsize=(12, 8))
        try:
            ax = sns.scatterplot(data=dataframe, x='liczba_przypadkow',
                                 y=dataframe['liczba_testow_z_wynikiem_negatywnym'] + dataframe[
                                     'liczba_testow_z_wynikiem_pozytywnym'], hue='wojewodztwo')
        except ValueError:
            raise HTTPException(status_code=404, detail='Ups, something was wrong, chceck your parameters')
        sns.move_legend(ax, "upper left")
        chart = 'Scatter'
        ax.set_xlabel('Liczba przypadków', fontsize=15)
        ax.set_ylabel('Liczba wszystkich testów', fontsize=15)
        ax.set_title(chart, fontsize=20)
        chart = chart + '.png'
        if os.path.isfile(chart):
            os.remove(chart)
        fig = ax.get_figure()
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
    plt.figure(figsize=(12, 8))
    try:
        ax = sns.scatterplot(data=dataframe, **params.dict())
    except ValueError:
        raise HTTPException(status_code=404, detail='Ups, something was wrong, chceck your parameters')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=15)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=15)
    fig = ax.get_figure()
    fig.savefig(chart)
    return FileResponse(chart)


@app_provider.post('/jointplot')
async def jointplot(*, params: JointPlot, columns: list = Query([]), conditions: list = Query([]),
                    data_source: Optional[str] = None, chart_name: Optional[str] = None):
    if not data_source:
        dataframe = province_full_data
        dataframe['log_liczba_przypakdow'] = np.log(dataframe['liczba_przypadkow'])
        dataframe['log_zgony'] = np.log(dataframe['zgony_w_wyniku_covid_i_chorob_wspolistniejacych'] + dataframe[
            'zgony_w_wyniku_covid_bez_chorob_wspolistniejacych'])
        try:
            sns.jointplot(data=dataframe, x='log_liczba_przypakdow', y='log_zgony', kind='hist')
        except ValueError:
            raise HTTPException(status_code=404, detail='Ups, something was wrong, chceck your parameters')
        chart = 'JointPlot'
        chart = chart + '.png'
        if os.path.isfile(chart):
            os.remove(chart)
        plt.savefig(chart)
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
    try:
        sns.jointplot(data=dataframe, **params.dict())
    except ValueError:
        print(params)
        raise HTTPException(status_code=404, detail='Ups, something was wrong, chceck your parameters')
    plt.savefig(chart)
    return FileResponse(chart)


@app_provider.post('/catplot')
async def catplot():
    dataframe = province_full_data
    g = sns.catplot(data=dataframe, x='stan_rekordu_na', y='liczba_przypadkow', col='wojewodztwo',
                    col_wrap=4)
    for ax in g.axes.flat:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(24))
        ax.set_xticklabels(rotation=30)
    chart = 'CatPlot'
    chart = chart + '.png'
    if os.path.isfile(chart):
        os.remove(chart)
    plt.savefig(chart)
    return FileResponse(chart)


@app_provider.post('/relplot')
async def relplot(*, params: RelPlot, columns: list = Query([]), conditions: list = Query([]),
                  data_source: Optional[str] = None, chart_name: Optional[str] = None):
    if data_source in dict.keys():
        dataframe = dict.get(data_source)
    else:
        raise HTTPException(status_code=404, detail='Please, select correctly data source')
    if not chart_name:
        raise HTTPException(status_code=404, detail='Please, give a name for chart')
    chart = chart_name + '.png'
    if os.path.isfile(chart):
        raise HTTPException(status_code=404, detail='Chart with that name already exist')
    dataframe = get_filtered_data(df=dataframe, columns=columns, query=conditions)
    try:
        sns.relplot(data=dataframe, **params.dict())
    except ValueError:
        raise HTTPException(status_code=404, detail='Ups, something was wrong, check your parameters')
    plt.savefig(chart)
    return FileResponse(chart)


@app_provider.post('/scattermatrix')
async def scatter_matrix(*, params: ScatterPlot, columns: list = Query([]), conditions: list = Query([]),
                         data_source: Optional[str] = None, chart_name: Optional[str] = None):
    if not data_source:
        dataframe = province_full_data
        dataframe = dataframe[['liczba_przypadkow', 'zgony_w_wyniku_covid_bez_chorob_wspolistniejacych',
                               'zgony_w_wyniku_covid_i_chorob_wspolistniejacych', 'liczba_ozdrowiencow',
                               'liczba_osob_objetych_kwarantanna', 'liczba_wykonanych_testow', 'wojewodztwo']]
        sns.pairplot(dataframe, hue='wojewodztwo')
        chart = 'scattermatrix.png'
        plt.savefig(chart)
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
    plt.figure(figsize=(12, 8))
    try:
        sns.pairplot(data=dataframe, **params.dict())
    except ValueError:
        raise HTTPException(status_code=404, detail='Ups, something was wrong, chceck your parameters')
    plt.savefig(chart)
    return FileResponse(chart)


@app_provider.post('/pca')
async def pca(*, columns: list = Query([]), conditions: list = Query([]),
              data_source: Optional[str] = None, chart_name: Optional[str] = None, filter: Optional[str] = None):
    if not data_source:
        dataframe = province_full_data
        #     dataframe = dataframe[dataframe['location'].isin(
        #         ['Poland', 'Germany', 'Italy', 'Czechia', 'Russia', 'Slovakia', 'Ukraine', 'France'])]
        #     columns = ['total_cases', 'new_cases', 'total_deaths', 'new_deaths',
        #                'total_tests', 'new_tests', 'total_vaccinations',
        #                'people_vaccinated', 'people_fully_vaccinated', 'total_boosters', 'new_vaccinations']
        columns = ['liczba_przypadkow', 'zgony_w_wyniku_covid_bez_chorob_wspolistniejacych',
                   'zgony_w_wyniku_covid_i_chorob_wspolistniejacych', 'liczba_zlecen_poz', 'liczba_ozdrowiencow',
                   'liczba_osob_objetych_kwarantanna', 'liczba_wykonanych_testow',
                   'liczba_testow_z_wynikiem_pozytywnym', 'liczba_testow_z_wynikiem_negatywnym']
        filter = 'wojewodztwo'
        chart_name = f'From {len(columns)}'
        chart = pca_method(chart_name=chart_name, dataframe=dataframe, columns_for_pca=columns, column_filter=filter,
                           n_components=2)
        return FileResponse(chart)
    #
    # return ':)'


@app_provider.post('/mds')
async def mds(*, columns: list = Query([]), conditions: list = Query([]),
              data_source: Optional[str] = None, chart_name: Optional[str] = None, filter: Optional[str] = None):
    if not data_source:
        dataframe = province_full_data
        #     dataframe = dataframe[dataframe['location'].isin(
        #         ['Poland', 'Germany', 'Italy', 'Czechia', 'Russia', 'Slovakia', 'Ukraine', 'France'])]
        #     columns = ['total_cases', 'new_cases', 'total_deaths', 'new_deaths',
        #                'total_tests', 'new_tests', 'total_vaccinations',
        #                'people_vaccinated', 'people_fully_vaccinated', 'total_boosters', 'new_vaccinations']
        columns = ['liczba_przypadkow', 'zgony_w_wyniku_covid_bez_chorob_wspolistniejacych',
                   'zgony_w_wyniku_covid_i_chorob_wspolistniejacych', 'liczba_zlecen_poz', 'liczba_ozdrowiencow',
                   'liczba_osob_objetych_kwarantanna', 'liczba_wykonanych_testow',
                   'liczba_testow_z_wynikiem_pozytywnym', 'liczba_testow_z_wynikiem_negatywnym']
        filter = 'wojewodztwo'
        chart = 'mds.png'
        mds_method(chart_name=chart, dataframe=dataframe, column_filter=filter, columns_for_pca=columns)
        return FileResponse(chart)


# from .database import engine, Base
# Base.metadata.create_all(engine)

# from sqlalchemy import inspect
# inspector = inspect(engine)

# if not inspector.has_table(engine, 'owid'):
# df.to_sql('owid', con=engine, if_exists='replace', index=False)
# from sqlalchemy.orm import Session
# from .dependencies import get_db
# from fastapi import Depends

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
