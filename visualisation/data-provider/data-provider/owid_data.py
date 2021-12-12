import urllib.request
import os
import pandas as pd
from .methods import download_file

import logging

LOGGER = logging.getLogger(__name__)


def make_owid():
    if os.path.isfile('owid_data.csv'):
        return pd.read_csv('owid_data.csv', low_memory=False)

    file = 'owid-covid-data.csv'
    url = 'https://covid.ourworldindata.org/data/'
    download_file(file, url)
    dataframe = pd.read_csv('owid-covid-data.csv')
    dataframe['date'] = dataframe['date'].astype('datetime64[ns]')

    population = 'population_latest.csv'
    pop_url = 'https://raw.githubusercontent.com/owid/covid-19-data/master/scripts/input/un/' + population
    if not os.path.isfile(population):
        LOGGER.info('Download file from', pop_url)
        urllib.request.urlretrieve(pop_url, population)
        LOGGER.info('Download success')
    else:
        LOGGER.info(f'File {population} is already downloaded')
    pop_file = pd.read_csv(population)

    for index, row in dataframe.iterrows():
        if row['location'] == 'International':
            value = pop_file[pop_file['entity'] == 'World']
            value = value['population']
            dataframe.loc[index, 'population'] = value.values[0]
        elif row['location'] == 'Northern Cyprus':
            dataframe.loc[index, 'population'] = 326000
        if ' ' in row['location']:
            dataframe.loc[index, 'location'] = dataframe.loc[index, 'location'].replace(' ', '_')

    dataframe.to_csv('owid_data.csv', index=False)
    return dataframe


df = make_owid()


def get_owid_full():
    return df


df_small = df[['iso_code', 'continent', 'location', 'date', 'population', 'total_cases', 'new_cases', 'total_deaths',
               'new_deaths', 'total_tests', 'new_tests', 'total_vaccinations',
               'people_vaccinated', 'people_fully_vaccinated', 'total_boosters', 'new_vaccinations', 'median_age',
               'aged_70_older']]


def get_owid_small():
    return df_small
