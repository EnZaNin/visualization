import numpy as np
import pandas as pd
import requests
import datetime
import os
import zipfile

from pathlib import Path
from pandas.core.frame import DataFrame

pd.options.mode.chained_assignment = None  # default='warn'


def get_gov_data() -> DataFrame:
    if os.path.isfile('gov_data.csv'):
        return pd.read_csv('gov_data.csv', low_memory=False)
    url = 'https://arcgis.com/sharing/rest/content/items/a8c562ead9c54e13a135b02e0d875ffb/data'

    r = requests.get(url, stream=True)

    with open('data_poland.zip', 'wb') as fd:
        for chunk in r.iter_content(chunk_size=128):
            fd.write(chunk)
    if os.path.isdir('data_poland'):
        Path('data_poland').mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile('data_poland.zip', 'r') as zip_ref:
        list_of_files = zip_ref.namelist()
        for file in list_of_files:
            path = Path('data_poland/' + file)
            if path.is_file() is False:
                zip_ref.extract(member=file, path='data_poland')

    cwd = os.path.abspath('data_poland')
    files = os.listdir(cwd)
    dataframe = pd.DataFrame()

    for count, file in enumerate(files):
        if file.endswith('.csv'):
            current_file = pd.read_csv("data_poland/" + file, delimiter=';', encoding='utf-8',
                                       encoding_errors='ignore')
            if current_file.iloc[0]['wojewodztwo'] != 'Cały kraj':
                current_file = pd.read_csv("data_poland/" + file, delimiter=';',
                                           encoding='windows-1250')
            if 'stan_rekordu_na' not in current_file.columns:
                current_file['stan_rekordu_na'] = dataframe.iloc[-1]['stan_rekordu_na'] + datetime.timedelta(days=1)
            current_file['stan_rekordu_na'] = current_file['stan_rekordu_na'].astype('datetime64[ns]')
            dataframe = dataframe.append(current_file, ignore_index=True)

    dataframe['liczba_ozdrowiencow'].fillna(0, inplace=True)
    dataframe['zgony'].fillna(0, inplace=True)
    dataframe['zgony_w_wyniku_covid_i_chorob_wspolistniejacych'].fillna(0, inplace=True)
    dataframe['zgony_w_wyniku_covid_bez_chorob_wspolistniejacych'].fillna(0, inplace=True)
    dataframe['liczba_zlecen_poz'].fillna(0, inplace=True)
    dataframe.to_csv('gov_data.csv', index=False)

    return dataframe


def get_szczepienia_data() -> DataFrame:
    if os.path.isfile('szczepienia.csv'):
        return pd.read_csv('szczepienia.csv', low_memory=False)
    url = 'https://api.dane.gov.pl/1.4/datasets/2476/resources/metadata.csv'
    r = requests.get(url)
    local_file = 'metadata.csv'
    with open(local_file, 'wb') as file:
        file.write(r.content)

    df = pd.read_csv('metadata.csv', delimiter=';')
    df = df[['Dane na dzień', 'URL pliku (do pobrania)']]
    df['Dane na dzień'] = pd.to_datetime(df['Dane na dzień'])
    df.sort_values(by='Dane na dzień', key=pd.to_datetime, inplace=True)
    mz_df = pd.DataFrame()
    for index, row in df.iterrows():
        data = pd.read_csv(row['URL pliku (do pobrania)'], delimiter=';', encoding_errors='ignore')
        data['dane_na_dzien'] = row['Dane na dzień']
        mz_df = mz_df.append(data, ignore_index=True)
    mz_df.to_csv('szczepienia.csv', index=False)

    return mz_df


def get_wojewodztwa_data() -> DataFrame:
    szczep_df = get_szczepienia_data()
    other_df = get_gov_data()
    szczep_df = szczep_df.groupby(['wojewodztwo_nazwa', 'dane_na_dzien']). \
        agg({'liczba_ludnosci': np.sum,
             'w1_zaszczepieni_pacjenci': np.sum,
             'w3_zaszczepieni_pelna_dawka': np.sum,
             '%_zaszczepionych': np.sum,
             '%_zaszczepionych_pen_dawk': np.sum,
             'w1_zaszczepieni_w_wieku_0_11': np.sum,
             'w1_zaszczepieni_w_wieku_12_19': np.sum,
             'w1_zaszczepieni_w_wieku_20_39': np.sum,
             'w1_zaszczepieni_w_wieku_40_59': np.sum,
             'w1_zaszczepieni_w_wieku_60_69': np.sum,
             'w1_zaszczepieni_w_wieku_70plus': np.sum,
             'w3_zaszczepieni_pen_dawk_w_wieku_0_11': np.sum,
             'w3_zaszczepieni_pen_dawk_w_wieku_12_19': np.sum,
             'w3_zaszczepieni_pen_dawk_w_wieku_20_39': np.sum,
             'w3_zaszczepieni_pen_dawk_w_wieku_40_59': np.sum,
             'w3_zaszczepieni_pen_dawk_w_wieku_60_69': np.sum,
             'w3_zaszczepieni_pen_dawk_w_wieku_70plus': np.sum
             })
    szczep_df = szczep_df.reset_index(level=['wojewodztwo_nazwa', 'dane_na_dzien'])

    woj_df = other_df.merge(szczep_df, left_on=['wojewodztwo', 'stan_rekordu_na'],
                            right_on=['wojewodztwo_nazwa', 'dane_na_dzien'], how='right')
    del woj_df['wojewodztwo_nazwa']
    del woj_df['dane_na_dzien']

    return woj_df


def get_powiat_szczep_data():
    if os.path.isfile('szczep_rap.csv'):
        return pd.read_csv('szczep_rap.csv')
    url = 'https://arcgis.com/sharing/rest/content/items/3f47db945aff47e582db8aa383ccf3a1/data'
    r = requests.get(url)

    with open('data_szczepienia.zip', 'wb') as fd:
        for chunk in r.iter_content(chunk_size=128):
            fd.write(chunk)
    if os.path.isdir('data_szczepienia'):
        Path('data_szczepienia').mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile('data_szczepienia.zip', 'r') as zip_ref:
        list_of_files = zip_ref.namelist()
        for file in list_of_files:
            path = Path('data_szczepienia/' + file)
            if path.is_file() is False:
                zip_ref.extract(member=file, path='data_szczepienia')

    cwd = os.path.abspath('data_szczepienia')
    files = os.listdir(cwd)

    for count, file in enumerate(files):
        if file.endswith('rap_rcb_pow_szczepienia.csv'):
            current_file = pd.read_csv("data_szczepienia/" + file, delimiter=';', encoding='windows-1250')
            current_file.to_csv('szczep_rap.csv')

            return current_file
