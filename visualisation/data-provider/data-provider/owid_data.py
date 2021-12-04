import urllib.request
import os
import pandas as pd
import re
import datetime as dt
from fastapi import HTTPException
from .methods import download_file, query_match

import logging

LOGGER = logging.getLogger(__name__)

# file = 'owid-covid-data.csv'
# URL = 'https://covid.ourworldindata.org/data/'
# download_file(file, URL)


# df = pd.read_csv('owid-covid-data.csv')
# df['date'] = df['date'].astype('datetime64[ns]')

# population = 'population_latest.csv'
# POP_URL = 'https://raw.githubusercontent.com/owid/covid-19-data/master/scripts/input/un/' + population
# # download_file(population, POP_URL)

# if not os.path.isfile(population):
#         LOGGER.info('Download file from', POP_URL)
#         urllib.request.urlretrieve(POP_URL, population)
#         LOGGER.info('Download success')
# else:
#         LOGGER.info(f'File {population} is already downloaded')
# pop_file = pd.read_csv(population)
# print(pop_file)
# for index, row in df.iterrows():
#     if row['location'] == 'International':
#         value = pop_file[pop_file['entity'] == 'World']
#         value = value['population']
#         df.loc[index, 'population'] = value.values[0]
#     elif row['location'] == 'Northern Cyprus':
#         df.loc[index, 'population'] = 326000
#     if (' ' in row['location']):
#         df.loc[index, 'location'] = df.loc[index, 'location'].replace(' ', '_')

def make_owid():
    if os.path.isfile('owid_data.csv'):
        return pd.read_csv('owid_data.csv', low_memory=False)
    
    file = 'owid-covid-data.csv'
    url = 'https://covid.ourworldindata.org/data/'
    download_file(file, url)
    df = pd.read_csv('owid-covid-data.csv')
    df['date'] = df['date'].astype('datetime64[ns]')

    population = 'population_latest.csv'
    POP_URL = 'https://raw.githubusercontent.com/owid/covid-19-data/master/scripts/input/un/' + population
    if not os.path.isfile(population):
        LOGGER.info('Download file from', POP_URL)
        urllib.request.urlretrieve(POP_URL, population)
        LOGGER.info('Download success')
    else:
        LOGGER.info(f'File {population} is already downloaded')
    pop_file = pd.read_csv(population)

    for index, row in df.iterrows():
        if row['location'] == 'International':
            value = pop_file[pop_file['entity'] == 'World']
            value = value['population']
            df.loc[index, 'population'] = value.values[0]
        elif row['location'] == 'Northern Cyprus':
            df.loc[index, 'population'] = 326000
        if (' ' in row['location']):
            df.loc[index, 'location'] = df.loc[index, 'location'].replace(' ', '_')
    
    df.to_csv('owid_data.csv')
    return df

df = make_owid()

def get_owid_full():
    return df


df_small = df[['iso_code', 'continent', 'location', 'date', 'population', 'total_cases', 'new_cases', 'total_deaths', 'new_deaths',
               'icu_patients', 'hosp_patients', 'total_tests', 'new_tests', 'total_vaccinations',
               'people_vaccinated', 'people_fully_vaccinated', 'total_boosters', 'new_vaccinations', 'median_age', 'aged_70_older']]

def get_owid_small():
    return df_small


def get_owid_df_with_columns(dataframe, query):
    dataframe = dataframe

    # for param in query:
    #     date_from = re.match(r'(?P<column>.*) > (?P<value>.*)', param)
    #     if date_from:
    #         date_from_datetype = dt.datetime.strptime(date_from.group('value'), '%Y-%m-%d')
    #         dataframe = dataframe[dataframe['date'] > date_from_datetype]
    #         if dataframe.size == 0:
    #             raise HTTPException(status_code=404, detail='Ups, your query is wrong (date_from)')

    #     date_to = re.match(r'(?P<column>.*) < (?P<value>.*)', param)
    #     if date_to:
    #         date_to_datetype = dt.datetime.strptime(date_to.group('value'), '%Y-%m-%d')
    #         dataframe = dataframe[dataframe['date'] < date_to_datetype]
    #         if dataframe.size == 0:
    #             raise HTTPException(status_code=404, detail='Ups, your query is wrong (date_to)')

    #     in_list = re.match(r'(?P<column>.*) in (?P<value>.*)', param)
    #     if in_list:
    #         list = in_list.group('value').split(" ")
    #         dataframe = dataframe[dataframe[in_list.group('column')].isin(list)]
    #         if dataframe.size == 0:
    #             raise HTTPException(status_code=404, detail='Ups, your query is wrong (in)')

    #     is_value = re.match(r'(?P<column>.*) = (?P<value>.*)', param)
    #     if is_value:
    #         dataframe = dataframe[dataframe[is_value.group('column')] == is_value.group('value')]
    #         if dataframe.size == 0:
    #             raise HTTPException(status_code=404, detail='Ups, your query is wrong (=)')
    dataframe = query_match(df=dataframe, query=query, data_column='date')

    return dataframe