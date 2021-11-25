from http.client import HTTPException
from fastapi import FastAPI, Response, HTTPException, Query, Path

import urllib.request
import os
# import io
import pandas as pd
from pandas.core.frame import DataFrame

pd.options.mode.chained_assignment = None  # default='warn'
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from .polish_data import get_gov_data, get_szczepienia_data

file = 'owid-covid-data.csv'
URL = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/' + file
if not os.path.isfile(file):
    print('Download file from ', URL)
    urllib.request.urlretrieve(URL, file)
    print('Download success')
else:
    print(f'File {file} is already downloaded')

df = pd.read_csv('owid-covid-data.csv')


df_clear = df[['iso_code', 'continent', 'location', 'date', 'population', 'total_cases', 'new_cases', 'total_deaths', 'new_deaths',
               'icu_patients', 'hosp_patients', 'total_tests', 'new_tests', 'tests_units', 'total_vaccinations',
               'people_vaccinated', 'people_fully_vaccinated', 'total_boosters', 'new_vaccinations']]
df_clear['date'] = df_clear['date'].astype('datetime64[ns]')

poland_df_clear = df_clear[df_clear['location'] == 'Poland']
poland_df_clear = poland_df_clear.append(df_clear[df_clear['location'] == 'Germany'])
# poland_df_clear = poland_df_clear[poland_df_clear['date'].dt.year == 2021]


# bytes_image = io.BytesIO()
# plt.savefig(ax, format='png')
# bytes_image.seek(0)

from fastapi.responses import FileResponse
from typing import Optional
from .schemas.seaborn_schema import LinePlot
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU

app_provider = FastAPI()

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

@app_provider.post('/boxplot')
async def boxplot(*, q: list = Query([])):
    # return ':)'
    # print(**q)
    query_items = {"q": q}
    return query_items
    # ax = sns.lineplot(q)


@app_provider.post('/histplot')
async def histplot(chart_name: str, data_source: str, parameters: LinePlot):
    # return ':)'
    chart = chart_name + '.png'
    if os.path.isfile(chart):
        # os.remove(chart_name)
        raise HTTPException(status_code=404, detail='Chart with that name already exist')
    if data_source == 'poland_df_clear':
            dataframe = poland_df_clear
    plt.clf()
    ax = sns.lineplot(data=dataframe, **parameters.dict())
    # ax = sns.lineplot(parameters)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    fig = ax.get_figure()
    # plt.figure(figsize=(24, 13.5), dpi=80)
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
# df.to_sql('owid', con=engine, if_exists='replace', index=False)
# # engine.execute("SELECT * FROM users").fetchall()

from sqlalchemy.orm import Session
from .dependencies import get_db
from fastapi import Depends
# from .cruds.cruds import DataDbTools

@app_provider.get('/dataframe')
def get_users(db: Session = Depends(get_db)):
    db_data = pd.read_sql_query('''select date, new_cases, total_cases from owid where location = 'Poland' ''', engine)
    # db_data = db.query('table_owid').offset(0).all()
    #     # db_data = engine.execute('select * from table_owid').fetchall()
    #     db_data = DataDbTools(db)
    #     data = db_data.get_from()
    
    print(db_data.info())
    return db_data

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


