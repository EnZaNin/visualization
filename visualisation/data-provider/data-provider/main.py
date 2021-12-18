import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging

from http.client import HTTPException
from fastapi import FastAPI, HTTPException, Query, Depends

import os
import pandas as pd
import json
from fastapi.responses import FileResponse
from typing import Optional, List
from .schemas.seaborn_schema import LinePlot, BoxPlot, HistPlot, ScatterPlot, JointPlot, RelPlot, CatPlot, PairPlot

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


def reload_data():
    df = get_owid_full()
    df_small = get_owid_small()
    district_stats_data = get_district_stats_data()
    district_vacc_data = get_district_vacc_data()
    province_full_data = get_province_full()
    df['date'] = df['date'].astype('datetime64[ns]')
    df_small['date'] = df_small['date'].astype('datetime64[ns]')
    district_stats_data['stan_rekordu_na'] = district_stats_data['stan_rekordu_na'].astype('datetime64[ns]')
    district_vacc_data['stan_rekordu_na'] = district_vacc_data['stan_rekordu_na'].astype('datetime64[ns]')
    province_full_data['stan_rekordu_na'] = province_full_data['stan_rekordu_na'].astype('datetime64[ns]')


df = get_owid_full()
df_small = get_owid_small()
district_stats_data = get_district_stats_data()
district_vacc_data = get_district_vacc_data()
province_full_data = get_province_full()
df['date'] = df['date'].astype('datetime64[ns]')
df_small['date'] = df_small['date'].astype('datetime64[ns]')
district_stats_data['stan_rekordu_na'] = district_stats_data['stan_rekordu_na'].astype('datetime64[ns]')
district_vacc_data['stan_rekordu_na'] = district_vacc_data['stan_rekordu_na'].astype('datetime64[ns]')
province_full_data['stan_rekordu_na'] = province_full_data['stan_rekordu_na'].astype('datetime64[ns]')

dict_owid = {'owid_full': df, 'owid_small': df_small}
dict_polish = {'district_stats': district_stats_data, 'district_vacc': district_vacc_data,
               'prov_data': province_full_data}
dict_full = {**dict_owid, **dict_polish}

description = """

CovidEDA is a tool for exploratory data analysis.

Is build on top of Pandas and Seaborn.

Examples of necessary quotes:

    * For parameter "columns" - columns is list of selected from DataFrame columns
        e.g. type here column name without ' ' -> date     
    * For parameter "conditions" - type here simple query to filter DataFrame
        e.g. select rows where location is Poland -> dataframe['location'] == 'Poland'
        e.g. select rows where location is Poland or Germany -> dataframe['location'].isin(['Poland', 'Germany'])
        e.g. select rows where date is more than 2020-05-23 -> dataframe['date_column'] > pd.to_datetime('2020-05-23')
        supported type: >=, >, ==, <, <=
    And that is what you need to know how to work with data!
"""

covid_app = FastAPI(title="CovidEDA",
                    description=description,
                    version="0.0.1",
                    # terms_of_service="http://example.com/terms/",
                    contact={
                           "name": "Tomasz Wawrykowicz",
                           # "url": "http://x-force.example.com/contact/",
                           "email": "246823@student.pwr.edu.pl",
                       }, )


@covid_app.get('/dataframe/part')
async def get_part_of_dataframe(*, data_source: str, columns: list = Query([]), conditions: list = Query([])):
    if data_source in dict_full.keys():
        dataframe = dict_full.get(data_source)
    else:
        raise HTTPException(status_code=404, detail='Dataframe not found')
    smaller_df = get_filtered_data(df=dataframe, columns=columns, query=conditions)
    if smaller_df.shape[0] > 100:
        smaller_df = smaller_df.tail(100)

    result = smaller_df.T.to_json(date_format='iso')
    logger.info('Loaded dataframe into json')
    return json.loads(result)


@covid_app.get('/dataframe/sample')
async def get_sample_from_dataframe(dataframe: str, orient: Optional[str] = None):
    """
    Get a random sample from DataFrame

    This will let the API user get a small table of data
    * Select data source
    * Returning data to you with json format, so you can select orient
    """
    if dataframe in dict_full.keys():
        dfr = dict_full.get(dataframe)
        dfr = dfr.sample(10).T
        result = dfr.to_json(orient=orient, date_format='iso')
        return json.loads(result)
    else:
        raise HTTPException(status_code=404, detail='Wrong name')


@covid_app.get('/df/{dataframe}')
async def get_columns(dataframe: str):
    """
    Get columns from DataFrame
    * Select data source
    * Return a list of columns
    """
    if dataframe in dict_full.keys():
        dataframe = dict_full.get(dataframe)
        return list(dataframe.columns)
    else:
        raise HTTPException(status_code=404, detail='Dataframe not found')


@covid_app.get('/stats')
async def get_summary_statistics(*, data_source: str, columns: list = Query([]), conditions: list = Query([])):
    """
    Get summary statistics from DataFrame
    * Select data source
    * If you want, you can select smaller DataFrame:
        - Columns - list of columns which be selected in returned DataFrame
        - Conditions - special query for selected DataFrame
    """
    if not data_source:
        dfr = dict_full.get(data_source)
        dfr_stats = dfr.describe().T
        dfr_stats['missing'] = dfr.isnull().sum()
        result = dfr_stats.T.to_json(date_format='iso')
        return json.loads(result)
    elif data_source in dict_full.keys():
        dataframe = dict_full.get(data_source)
    else:
        raise HTTPException(status_code=404, detail='Dataframe not found')
    dfr_stats_filtered = get_filtered_data(df=dataframe, columns=columns, query=conditions)
    dfr_stats = dfr_stats_filtered.describe().T
    dfr_stats['missing'] = dfr_stats_filtered.isnull().sum()
    dfr_stats = dfr_stats.apply(lambda x: round(x, 2))
    result = dfr_stats.T.to_json(date_format='iso')
    return json.loads(result)


@covid_app.get('/lastday')
async def get_stats_for_last_day(data_source: str):
    if data_source in dict_full.keys():
        dataframe = dict_full.get(data_source)
    else:
        raise HTTPException(status_code=404, detail='Dataframe not found')
    if 'date' in list(dataframe.columns):
        date = pd.unique(dataframe['date'])
        date.sort()
        date_key = date[-1]
        dataframe = dataframe[dataframe['date'] == date_key]
        dfr_stats = dataframe.describe().T
        dfr_stats['missing'] = dataframe.isnull().sum()
        dfr_stats = dfr_stats.apply(lambda x: round(x, 2))
        result = json.loads(dfr_stats.T.to_json(date_format='iso'))
        return {str(date_key): result}
    elif 'stan_rekordu_na' in list(dataframe.columns):
        date = pd.unique(dataframe['stan_rekordu_na'])
        date.sort()
        date_key = date[-1]
        dataframe = dataframe[dataframe['stan_rekordu_na'] == date_key]
        dfr_stats = dataframe.describe().T
        dfr_stats['missing'] = dataframe.isnull().sum()
        dfr_stats = dfr_stats.apply(lambda x: round(x, 2))
        result = dfr_stats.T.to_json(date_format='iso')
        return {str(date_key): json.loads(result)}


def agg_dict(aggfunc: List[str] = Query([])):
    return list(map(json.loads, aggfunc))


@covid_app.get('/pivot')
async def get_pivot(*, columns: list = Query([]), conditions: list = Query([]), data_source: Optional[str] = None,
                    values: list = Query([]), pivot_rows: list = Query([]), pivot_columns: list = Query([]),
                    agg_func: list = Depends(agg_dict)):
    if not data_source:
        dataframe = df_small
        dataframe = dataframe[dataframe['date'].isin(dataframe['date'].sample(10))]
        dataframe = dataframe[dataframe['location'].isin(dataframe['location'].sample(10))]
        pivo = pd.pivot_table(dataframe, values=['total_cases'], index=['location'], columns=['date'])
        result = pivo.to_json(date_format='iso')
        return json.loads(result)
    elif data_source in dict_full.keys():
        dataframe = dict_full.get(data_source)
    else:
        raise HTTPException(status_code=404, detail='Please, select correctly data source')
    dataframe = get_filtered_data(df=dataframe, columns=columns, query=conditions)
    agg = {}
    try:
        for idx, part in enumerate(agg_func):
            for k, v in agg_func[idx].items():
                if v == 'sum':
                    agg[k] = np.sum
                elif v == 'mean':
                    agg[k] = np.mean
                    # {"liczba_przypadkow": "sum"}
        pivo = pd.pivot_table(dataframe, values=values, index=pivot_rows, columns=pivot_columns, aggfunc=agg)
    except ValueError:
        raise HTTPException(status_code=400, detail='Please, select correctly parameters')
    result = pivo.to_json(date_format='iso')

    return json.loads(result)


@covid_app.get('/charts')
async def get_name_all_created_charts():
    files = os.listdir()
    chart_list = []
    for file in files:
        if file.endswith('.png'):
            chart_list.append(file)
    if len(chart_list) == 0:
        raise HTTPException(status_code=404, detail='Empty list - no charts')
    return chart_list


@covid_app.get('/chart/{chart_name}')
async def get_chart_by_name(chart_name: str):
    chart = chart_name + '.png'
    if not os.path.isfile(chart):
        raise HTTPException(status_code=404, detail='Chart with that name does not exist')
    return FileResponse(chart)


@covid_app.put('/update')
async def update_data_manually():
    os.remove('owid_data.csv')
    os.remove('szczepienia.csv')
    os.remove('gov_data.csv')
    os.remove('province_full.csv')
    os.remove('metadata.csv')
    print('deleted files')
    reload_data()


@covid_app.post('/lineplot', summary="Create lineplot for selected data")
async def lineplot(*, parameters: LinePlot, columns: list = Query([]), conditions: list = Query([]),
                   data_source: Optional[str] = None, chart_name: Optional[str] = None, xlabel: Optional[str] = None,
                   ylabel: Optional[str] = None):
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
        plt.figure(figsize=(14, 8))
        fig.savefig(chart)
        return FileResponse(chart)
    elif data_source in dict_full.keys():
        dataframe = dict_full.get(data_source)
    else:
        raise HTTPException(status_code=404, detail='Please, select correctly data source')
    dataframe = get_filtered_data(df=dataframe, columns=columns, query=conditions)
    if not chart_name:
        raise HTTPException(status_code=400, detail='Please, give a name for chart')
    chart = chart_name + '.png'
    if os.path.isfile(chart):
        raise HTTPException(status_code=400, detail='Chart with that name already exist')
    try:
        ax = sns.lineplot(data=dataframe, **parameters.dict())
    except ValueError or AttributeError:
        raise HTTPException(status_code=400, detail='Ups, something was wrong, chceck your parameters')
    if parameters.x == 'date':
        min_date = dataframe['date'].min()
        max_date = dataframe['date'].max()
        auto_scaler_date(min_date=min_date, max_date=max_date, ax=ax)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=15)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=15)
    plt.xticks(rotation=30)
    ax.set_title(chart_name, fontsize=20)
    plt.xticks(rotation=30)
    fig = ax.get_figure()
    plt.figure(figsize=(24, 13.5), dpi=80)
    fig.savefig(chart)
    return FileResponse(chart)


@covid_app.post('/boxplot')
async def boxplot(*, params: BoxPlot, columns: list = Query([]), conditions: list = Query([]),
                  data_source: Optional[str] = None, chart_name: Optional[str] = None, xlabel: Optional[str] = None,
                  ylabel: Optional[str] = None, add_points: Optional[bool] = False):
    plt.clf()
    if not data_source:
        dataframe = province_full_data
        ax = sns.boxplot(data=dataframe, x='liczba_przypadkow', y='wojewodztwo', orient='h')
        chart = 'Wykres pudełkowy dla województw'
        ax.set_xlabel('Liczba przypadków', fontsize=15)
        ax.set_ylabel('Województwa', fontsize=15)
        ax.set_title(chart, fontsize=20)
        chart = chart + '.png'
        fig = ax.get_figure()
        plt.figure(figsize=(12, 8))
        fig.savefig(chart)
        return FileResponse(chart)
    elif data_source in dict_full.keys():
        dataframe = dict_full.get(data_source)
    else:
        raise HTTPException(status_code=404, detail='Please, select correctly data source')
    if not chart_name:
        raise HTTPException(status_code=400, detail='Please, give a name for chart')
    chart = chart_name + '.png'
    if os.path.isfile(chart):
        raise HTTPException(status_code=400, detail='Chart with that name already exist')
    dataframe = get_filtered_data(df=dataframe, columns=columns, query=conditions)

    plt.figure(figsize=(12, 8))
    try:
        ax = sns.boxplot(data=dataframe, **params.dict())
        if add_points:
            sns.stripplot(data=dataframe, **params.dict(), size=4, color=".3", linewidth=0)
    except ValueError or AttributeError:
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


@covid_app.post('/histplot')
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
    elif data_source in dict_full.keys():
        dataframe = dict_full.get(data_source)
    else:
        raise HTTPException(status_code=404, detail='Please, select correctly data source')
    if not chart_name:
        raise HTTPException(status_code=400, detail='Please, give a name for chart')
    chart = chart_name + '.png'
    if os.path.isfile(chart):
        raise HTTPException(status_code=400, detail='Chart with that name already exist')
    dataframe = get_filtered_data(df=dataframe, columns=columns, query=conditions)
    plt.figure(figsize=(24, 13.5), dpi=80)
    try:
        ax = sns.histplot(data=dataframe, **params.dict())
    except ValueError or AttributeError:
        raise HTTPException(status_code=400, detail='Ups, something was wrong, chceck your parameters')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=15)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=15)
    fig = ax.get_figure()
    fig.savefig(chart)
    return FileResponse(chart)


@covid_app.post('/scatterplot')
async def scatterplot(*, params: ScatterPlot, columns: list = Query([]), conditions: list = Query([]),
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
    elif data_source in dict_full.keys():
        dataframe = dict_full.get(data_source)
    else:
        raise HTTPException(status_code=404, detail='Please, select correctly data source')
    if not chart_name:
        raise HTTPException(status_code=400, detail='Please, give a name for chart')
    chart = chart_name + '.png'
    if os.path.isfile(chart):
        raise HTTPException(status_code=400, detail='Chart with that name already exist')
    dataframe = get_filtered_data(df=dataframe, columns=columns, query=conditions)
    plt.figure(figsize=(12, 8))
    try:
        ax = sns.scatterplot(data=dataframe, **params.dict())
    except ValueError or AttributeError:
        raise HTTPException(status_code=404, detail='Ups, something was wrong, chceck your parameters')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=15)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=15)
    fig = ax.get_figure()
    fig.savefig(chart)
    return FileResponse(chart)


@covid_app.post('/heatmap')
async def heatmap(*, columns: list = Query([]), conditions: list = Query([]), data_source: str,
                  chart_name: str, value: str, x: str, y: str):
    if data_source in dict_full.keys():
        dataframe = dict_full.get(data_source)
    else:
        raise HTTPException(status_code=404, detail='Please, select correctly data source')
    if not chart_name:
        raise HTTPException(status_code=400, detail='Please, give a name for chart')
    chart = chart_name + '.png'
    if os.path.isfile(chart):
        raise HTTPException(status_code=400, detail='Chart with that name already exist')
    dataframe = get_filtered_data(df=dataframe, columns=columns, query=conditions)
    dataframe = dataframe.pivot(value, x, y)
    try:
        ax = sns.heatmap(dataframe)
    except ValueError or AttributeError:
        raise HTTPException(status_code=404, detail='Sorry, this method is not allowed')
    plt.savefig(chart)
    return FileResponse(chart)


@covid_app.post('/jointplot')
async def jointplot(*, params: JointPlot, columns: list = Query([]), conditions: list = Query([]),
                    data_source: Optional[str] = None, chart_name: Optional[str] = None):
    plt.clf()
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
    elif data_source in dict_full.keys():
        dataframe = dict_full.get(data_source)
    else:
        raise HTTPException(status_code=404, detail='Please, select correctly data source')
    if not chart_name:
        raise HTTPException(status_code=400, detail='Please, give a name for chart')
    chart = chart_name + '.png'
    if os.path.isfile(chart):
        raise HTTPException(status_code=400, detail='Chart with that name already exist')
    dataframe = get_filtered_data(df=dataframe, columns=columns, query=conditions)
    try:
        g = sns.jointplot(data=dataframe, **params.dict())
        for tick in g.ax_joint.get_xticklabels():
            tick.set_rotation(30)
    except ValueError or AttributeError:
        raise HTTPException(status_code=404, detail='Ups, something was wrong, chceck your parameters')
    plt.savefig(chart)
    return FileResponse(chart)


@covid_app.post('/catplot')
async def catplot(*, params: CatPlot, columns: list = Query([]), conditions: list = Query([]),
                  data_source: Optional[str] = None, chart_name: Optional[str] = None):
    if not data_source:
        dataframe = province_full_data
        g = sns.catplot(data=dataframe, x='stan_rekordu_na', y='liczba_przypadkow', col='wojewodztwo',
                        col_wrap=4)
        min_date = dataframe['stan_rekordu_na'].min()
        max_date = dataframe['stan_rekordu_na'].max()
        ax2 = g.axes[0].twiny()
        auto_scaler_date(min_date=min_date, max_date=max_date, ax=ax2)
        chart = 'CatPlot'
        chart = chart + '.png'
        if os.path.isfile(chart):
            os.remove(chart)
        plt.savefig(chart)
        return FileResponse(chart)
    elif data_source in dict_full.keys():
        dataframe = dict_full.get(data_source)
    else:
        raise HTTPException(status_code=404, detail='Please, select correctly data source')
    if not chart_name:
        raise HTTPException(status_code=400, detail='Please, give a name for chart')
    chart = chart_name + '.png'
    if os.path.isfile(chart):
        raise HTTPException(status_code=400, detail='Chart with that name already exist')
    dataframe = get_filtered_data(df=dataframe, columns=columns, query=conditions)
    try:
        g = sns.catplot(data=dataframe, **params.dict())
        if params.x == 'stan_rekordu_na':
            min_date = dataframe['stan_rekordu_na'].min()
            max_date = dataframe['stan_rekordu_na'].max()
            ax2 = g.axes[0].twiny()
            auto_scaler_date(min_date=min_date, max_date=max_date, ax=ax2)
        elif params.x == 'date':
            min_date = dataframe['date'].min()
            max_date = dataframe['date'].max()
            ax2 = g.axes[0].twiny()
            auto_scaler_date(min_date=min_date, max_date=max_date, ax=ax2)
    except ValueError or AttributeError:
        raise HTTPException(status_code=400, detail='Ups, something was wrong, check your parameters')
    plt.savefig(chart)
    return FileResponse(chart)


@covid_app.post('/relplot')
async def relplot(*, params: RelPlot, columns: list = Query([]), conditions: list = Query([]),
                  data_source: Optional[str] = None, chart_name: Optional[str] = None):
    if data_source in dict_full.keys():
        dataframe = dict_full.get(data_source)
    else:
        raise HTTPException(status_code=404, detail='Please, select correctly data source')
    if not chart_name:
        raise HTTPException(status_code=400, detail='Please, give a name for chart')
    chart = chart_name + '.png'
    if os.path.isfile(chart):
        raise HTTPException(status_code=400, detail='Chart with that name already exist')
    dataframe = get_filtered_data(df=dataframe, columns=columns, query=conditions)
    try:
        sns.relplot(data=dataframe, **params.dict())
    except ValueError or AttributeError:
        raise HTTPException(status_code=404, detail='Ups, something was wrong, check your parameters')
    plt.savefig(chart)
    return FileResponse(chart)


@covid_app.post('/pairplot')
async def pairplot(*, params: PairPlot, columns: list = Query([]), conditions: list = Query([]),
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
    elif data_source in dict_full.keys():
        dataframe = dict_full.get(data_source)
    else:
        raise HTTPException(status_code=404, detail='Please, select correctly data source')
    if not chart_name:
        raise HTTPException(status_code=400, detail='Please, give a name for chart')
    chart = chart_name + '.png'
    if os.path.isfile(chart):
        raise HTTPException(status_code=400, detail='Chart with that name already exist')
    dataframe = get_filtered_data(df=dataframe, columns=columns, query=conditions)
    plt.figure(figsize=(12, 8))
    try:
        sns.pairplot(data=dataframe, **params.dict())
    except ValueError or AttributeError:
        raise HTTPException(status_code=404, detail='Ups, something was wrong, chceck your parameters')
    plt.savefig(chart)
    return FileResponse(chart)


@covid_app.post('/pca')
async def pca(*, columns: list = Query([]), conditions: list = Query([]), columns_for_pca: list = Query([]),
              data_source: Optional[str] = None, chart_name: Optional[str] = None, column_filter: Optional[str] = None):
    if not data_source:
        dataframe = province_full_data
        columns = ['liczba_przypadkow', 'zgony_w_wyniku_covid_bez_chorob_wspolistniejacych',
                   'zgony_w_wyniku_covid_i_chorob_wspolistniejacych', 'liczba_ozdrowiencow',
                   'liczba_osob_objetych_kwarantanna', 'liczba_wykonanych_testow',
                   'liczba_testow_z_wynikiem_pozytywnym', 'liczba_testow_z_wynikiem_negatywnym',
                   'w1_zaszczepieni_pacjenci', 'w3_zaszczepieni_pelna_dawka']
        column_filter = 'wojewodztwo'
        chart_name = 'PCA'
        chart = pca_method(chart_name=chart_name, dataframe=dataframe, columns_for_pca=columns,
                           column_filter=column_filter, n_components=2)
        return FileResponse(chart)
    elif data_source in dict_full.keys():
        dataframe = dict_full.get(data_source)
    else:
        raise HTTPException(status_code=404, detail='Dataframe not found')
    if not chart_name:
        raise HTTPException(status_code=400, detail='Please, give a name for chart')
    chart = chart_name + '.png'
    if os.path.isfile(chart):
        raise HTTPException(status_code=404, detail='Chart with that name already exist')
    dataframe = get_filtered_data(df=dataframe, columns=columns, query=conditions)
    try:
        chart = pca_method(chart_name=chart_name, dataframe=dataframe, columns_for_pca=columns_for_pca,
                           column_filter=column_filter, n_components=2)
    except ValueError:
        raise HTTPException(status_code=400, detail='Error with PCA algorithm')
    return FileResponse(chart)


@covid_app.post('/mds')
async def mds(*, columns: list = Query([]), conditions: list = Query([]), columns_for_mds: list = Query([]),
              data_source: Optional[str] = None, chart_name: Optional[str] = None, column_filter: Optional[str] = None):
    if not data_source:
        dataframe = province_full_data
        columns = ['liczba_przypadkow', 'zgony_w_wyniku_covid_bez_chorob_wspolistniejacych',
                   'zgony_w_wyniku_covid_i_chorob_wspolistniejacych', 'liczba_zlecen_poz', 'liczba_ozdrowiencow',
                   'liczba_osob_objetych_kwarantanna', 'liczba_wykonanych_testow',
                   'liczba_testow_z_wynikiem_pozytywnym', 'liczba_testow_z_wynikiem_negatywnym']
        column_filter = 'wojewodztwo'
        chart_name = 'MDS'
        chart = mds_method(chart_name=chart_name, dataframe=dataframe, columns_for_mds=columns,
                           column_filter=column_filter)
        return FileResponse(chart)
    elif data_source in dict_full.keys():
        dataframe = dict_full.get(data_source)
    else:
        raise HTTPException(status_code=404, detail='Dataframe not found')
    if not chart_name:
        raise HTTPException(status_code=400, detail='Please, give a name for chart')
    chart = chart_name + '.png'
    if os.path.isfile(chart):
        raise HTTPException(status_code=400, detail='Chart with that name already exist')
    dataframe = get_filtered_data(df=dataframe, columns=columns, query=conditions)
    chart = mds_method(chart_name=chart_name, dataframe=dataframe, columns_for_mds=columns_for_mds,
                       column_filter=column_filter)
    return FileResponse(chart)
