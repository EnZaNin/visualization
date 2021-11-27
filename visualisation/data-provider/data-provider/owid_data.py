import urllib.request
import os
import pandas as pd

import logging

LOGGER = logging.getLogger()

file = 'owid-covid-data.csv'
URL = 'https://covid.ourworldindata.org/data/' + file
if not os.path.isfile(file):
    LOGGER.info('Download file from', URL)
    urllib.request.urlretrieve(URL, file)
    LOGGER.info('Download success')
else:
    LOGGER.info(f'File {file} is already downloaded')

# dodać tworzenie dfów na podstawie wybranych kolumn przeze mnie
# pobrać oryginalny df i go oczyścic
# dodać tworzenie dfów na podstawie kolumn wybieranych przez użytkownika
# możliwe że trzeba przerzucić się na sql z powodu grupowania itp.

df = pd.read_csv('owid-covid-data.csv')

pop_file = pd.read_csv(
    'https://raw.githubusercontent.com/owid/covid-19-data/master/scripts/input/un/population_latest.csv')
for index, row in df.iterrows():
    if row['location'] == 'International':
        value = pop_file[pop_file['entity'] == 'World']
        value = value['population']
        df.loc[index, 'population'] = value.values[0]
    elif row['location'] == 'Northern Cyprus':
        df.loc[index, 'population'] = 326000

def get_df_full():
    return df


df_small = df[['iso_code', 'continent', 'location', 'date', 'population', 'total_cases', 'new_cases', 'total_deaths', 'new_deaths',
               'icu_patients', 'hosp_patients', 'total_tests', 'new_tests', 'tests_units', 'total_vaccinations',
               'people_vaccinated', 'people_fully_vaccinated', 'total_boosters', 'new_vaccinations', 'median_age', 'aged_70_older']]

def get_df_small():
    return df_small

def get_df_with_columns(columns_list):
    own_df = df[columns_list]
    return own_df
