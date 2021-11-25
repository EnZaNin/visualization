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
