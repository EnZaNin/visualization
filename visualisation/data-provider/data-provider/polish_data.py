import numpy as np
import pandas as pd
import requests
import datetime
import os
import zipfile
import re
import datetime as dt

from pathlib import Path
from pandas.core.frame import DataFrame
from fastapi import HTTPException
from .methods import query_match

pd.options.mode.chained_assignment = None  # default='warn'


def get_district_stats_data() -> DataFrame:
    if os.path.isfile('gov_data.csv'):
        return pd.read_csv('gov_data.csv', low_memory=False)
    # url = 'https://arcgis.com/sharing/rest/content/items/a8c562ead9c54e13a135b02e0d875ffb/data'
    url = 'https://arcgis.com/sharing/rest/content/items/e16df1fa98c2452783ec10b0aea4b341/data'

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
                                       encoding_errors='ignore', decimal=',')
            if current_file.iloc[0]['wojewodztwo'] != 'Cały kraj':
                current_file = pd.read_csv("data_poland/" + file, delimiter=';',
                                           encoding='windows-1250', decimal=',')
            if 'stan_rekordu_na' not in current_file.columns:
                current_file['stan_rekordu_na'] = dataframe.iloc[-1]['stan_rekordu_na'] + datetime.timedelta(days=1)
            current_file['stan_rekordu_na'] = current_file['stan_rekordu_na'].astype('datetime64[ns]')
            dataframe = dataframe.append(current_file, ignore_index=True)

    dataframe['liczba_ozdrowiencow'].fillna(0, inplace=True)
    dataframe['liczba_przypadkow'].fillna(0, inplace=True)
    dataframe['liczba_na_10_tys_mieszkancow'].fillna(0, inplace=True)
    dataframe['liczba_osob_objetych_kwarantanna'].fillna(0, inplace=True)
    dataframe['liczba_wykonanych_testow'].fillna(0, inplace=True)
    dataframe['liczba_testow_z_wynikiem_pozytywnym'].fillna(0, inplace=True)
    dataframe['liczba_testow_z_wynikiem_negatywnym'].fillna(0, inplace=True)
    dataframe['liczba_pozostalych_testow'].fillna(0, inplace=True)
    dataframe['zgony'].fillna(0, inplace=True)
    dataframe['zgony_w_wyniku_covid_i_chorob_wspolistniejacych'].fillna(0, inplace=True)
    dataframe['zgony_w_wyniku_covid_bez_chorob_wspolistniejacych'].fillna(0, inplace=True)
    dataframe['liczba_zlecen_poz'].fillna(0, inplace=True)
    dataframe = dataframe[dataframe['powiat_miasto'].notna()]

    dataframe['wojewodztwo'] = dataframe['wojewodztwo'].apply(str)
    dataframe['powiat_miasto'] = dataframe['powiat_miasto'].apply(str)
    dataframe['liczba_na_10_tys_mieszkancow'] = dataframe['liczba_na_10_tys_mieszkancow'].apply(float)

    dataframe['stan_rekordu_na'] = dataframe['stan_rekordu_na'].astype('datetime64[ns]')
    del dataframe['teryt']
    del dataframe['liczba_na_10_tys_mieszkancow']
    dataframe = dataframe[~dataframe['wojewodztwo'].isin(['Cały kraj'])]

    dataframe.to_csv('gov_data.csv', index=False)

    return dataframe


def get_district_vacc_data() -> DataFrame:
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
        url = row['URL pliku (do pobrania)']
        data = pd.read_csv(url, delimiter=';', encoding_errors='ignore', decimal=',')
        data['stan_rekordu_na'] = row['Dane na dzień']
        if url.endswith('2021-09-07/file'):
            data['stan_rekordu_na'] = pd.to_datetime('2021-09-07')
        mz_df = mz_df.append(data, ignore_index=True)
    dfs = mz_df

    del dfs['wojewodztwo_teryt']
    del dfs['powiat_teryt']
    del dfs['gmina_teryt']

    dfs['%_zaszczepionych'].fillna(dfs['%_zsszczepionych'], inplace=True)
    dfs['%_zaszczepionych'].fillna(dfs['%_zaczepionych'], inplace=True)
    dfs['%_zaszczepionych'].fillna(dfs['&_zaszczepionych'], inplace=True)
    dfs['%_zaszczepionych'].fillna(dfs['%_ zaszczepionych'], inplace=True)
    dfs['%_zaszczepionych'].fillna(dfs['% zaszczepionych'], inplace=True)
    del dfs['%_zsszczepionych']
    del dfs['%_zaczepionych']
    del dfs['&_zaszczepionych']
    del dfs['%_ zaszczepionych']
    del dfs['% zaszczepionych']

    dfs['%_zaszczepionych_pen_dawk'].fillna(dfs['% zaszczepionych pen dawk'], inplace=True)
    dfs['%_zaszczepionych_pen_dawk'].fillna(dfs['%_zaszczepionych_pena_dawk'], inplace=True)
    dfs['%_zaszczepionych_pen_dawk'].fillna(dfs['%_zsszczepionych_pen_dawk'], inplace=True)
    dfs['%_zaszczepionych_pen_dawk'].fillna(dfs['%_zszczepionych_pen_dawk'], inplace=True)
    dfs['%_zaszczepionych_pen_dawk'].fillna(dfs['&_zaszczepionych_pen_dawk'], inplace=True)
    dfs['%_zaszczepionych_pen_dawk'].fillna(dfs['%_zaszczepionych_pn_dawk'], inplace=True)
    del dfs['% zaszczepionych pen dawk']
    del dfs['%_zaszczepionych_pena_dawk']
    del dfs['%_zsszczepionych_pen_dawk']
    del dfs['%_zszczepionych_pen_dawk']
    del dfs['&_zaszczepionych_pen_dawk']
    del dfs['%_zaszczepionych_pn_dawk']

    dfs['w3_zaszczepieni_pen_dawk_w_wieku_0_11'].fillna(dfs['w3zaszczepieni_pen_dawk_w_wieku__0_11'], inplace=True)
    dfs['w1_zaszczepieni_w_wieku_12_19'].fillna(dfs['w1_zaszczepiemi_w_wieku_12_19'], inplace=True)
    del dfs['w3zaszczepieni_pen_dawk_w_wieku__0_11']
    del dfs['w1_zaszczepiemi_w_wieku_12_19']

    dfs = dfs.groupby(['wojewodztwo_nazwa', 'powiat_nazwa', 'stan_rekordu_na']). \
        agg({'liczba_ludnosci': np.sum,
             'w1_zaszczepieni_pacjenci': np.sum,
             'w3_zaszczepieni_pelna_dawka': np.sum,
            #  '%_zaszczepionych': np.mean,
            #  '%_zaszczepionych_pen_dawk': np.mean,
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
    dfs['%_zaszczepionych'] = dfs['w1_zaszczepieni_pacjenci']/dfs['liczba_ludnosci']
    dfs['%_zaszczepionych_pen_dawk'] = dfs['w3_zaszczepieni_pelna_dawka']/dfs['liczba_ludnosci']
    dfs = dfs.reset_index(level=['wojewodztwo_nazwa', 'powiat_nazwa', 'stan_rekordu_na'])
    dfs['%_zaszczepionych'] = dfs['%_zaszczepionych'].apply(lambda x: round(x, 2))
    dfs['%_zaszczepionych_pen_dawk'] = dfs['%_zaszczepionych_pen_dawk'].apply(lambda x: round(x, 2))
    dfs['stan_rekordu_na'] = pd.to_datetime(dfs['stan_rekordu_na'])

    dfs.to_csv('szczepienia.csv', index=False)

    return dfs


def get_province_full() -> DataFrame:
    if os.path.isfile('province_full.csv'):
        return pd.read_csv('province_full.csv', low_memory=False)
    stat_df = get_district_stats_data()
    vacc_df = get_district_vacc_data()
    # vacc_df.rename({'stan_rekordu_na': 'dane_na_dzien'})
    vacc_df['dane_na_dzien'] = vacc_df['stan_rekordu_na']
    del vacc_df['stan_rekordu_na']

    stat_df = stat_df.groupby(['wojewodztwo', 'stan_rekordu_na']).agg(
        {
            'liczba_przypadkow': np.sum,
            # 'liczba_na_10_tys_mieszkancow': np.sum,
            'zgony_w_wyniku_covid_bez_chorob_wspolistniejacych': np.sum,
            'zgony_w_wyniku_covid_i_chorob_wspolistniejacych': np.sum,
            'liczba_zlecen_poz': np.sum,
            'liczba_ozdrowiencow': np.sum,
            'liczba_osob_objetych_kwarantanna': np.sum,
            'liczba_wykonanych_testow': np.sum,
            'liczba_testow_z_wynikiem_pozytywnym': np.sum,
            'liczba_testow_z_wynikiem_negatywnym': np.sum,
            'liczba_pozostalych_testow': np.sum
        }
    )
    stat_df = stat_df.reset_index(level=['wojewodztwo', 'stan_rekordu_na'])
    vacc_df = vacc_df.groupby(['wojewodztwo_nazwa', 'dane_na_dzien']).agg(
        {
            'liczba_ludnosci': np.sum,
             'w1_zaszczepieni_pacjenci': np.sum,
             'w3_zaszczepieni_pelna_dawka': np.sum,
            #  '%_zaszczepionych': np.mean,
            #  '%_zaszczepionych_pen_dawk': np.mean,
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
        }
    )
    vacc_df = vacc_df.reset_index(level=['wojewodztwo_nazwa', 'dane_na_dzien'])
    vacc_df['%_zaszczepionych'] = vacc_df['w1_zaszczepieni_pacjenci']/vacc_df['liczba_ludnosci']
    vacc_df['%_zaszczepionych_pen_dawk'] = vacc_df['w3_zaszczepieni_pelna_dawka']/vacc_df['liczba_ludnosci']
    vacc_df['%_zaszczepionych'] = vacc_df['%_zaszczepionych'].apply(lambda x: round(x, 2))
    vacc_df['%_zaszczepionych_pen_dawk'] = vacc_df['%_zaszczepionych_pen_dawk'].apply(lambda x: round(x, 2))

    vacc_df['dane_na_dzien'] = pd.to_datetime(vacc_df['dane_na_dzien'])
    stat_df['stan_rekordu_na'] = stat_df['stan_rekordu_na'].astype('datetime64[ns]')
    
    province_df = stat_df.merge(vacc_df, left_on=['wojewodztwo', 'stan_rekordu_na'],
                            right_on=['wojewodztwo_nazwa', 'dane_na_dzien'], how='right')
    del province_df['wojewodztwo_nazwa']
    del province_df['dane_na_dzien']
    province_df['liczba_na_10_tys_mieszkancow'] = 10000 * province_df['liczba_przypadkow'] / province_df['liczba_ludnosci']

    province_df.to_csv('province_full.csv', index=False)

    return province_df

def get_pol_df_by_condition(df, query):
    dataframe = df

    # for param in query:
    #     date_from = re.match(r'(?P<column>.*) > (?P<value>.*)', param)
    #     if date_from:
    #         if date_from.group('column') == 'date_from':
    #             date_from_datetype = dt.datetime.strptime(date_from.group('value'), '%Y-%m-%d')
    #             dataframe = dataframe[dataframe['stan_rekordu_na'] > date_from_datetype]
    #         else:
    #             dataframe = dataframe[dataframe[date_from.group('column')] > int(date_from.group('value'))]
    #         if dataframe.size == 0:
    #             raise HTTPException(status_code=404, detail='Ups, your query is wrong (date_from)')

    #     date_to = re.match(r'(?P<column>.*) < (?P<value>.*)', param)
    #     if date_to:
    #         if date_to.group('column') == 'date_to':
    #             date_to_datetype = dt.datetime.strptime(date_to.group('value'), '%Y-%m-%d')
    #             dataframe = dataframe[dataframe['stan_rekordu_na'] < date_to_datetype]
    #         else:
    #             dataframe = dataframe[dataframe[date_to.group('column')] < int(date_to.group('value'))]
    #         if dataframe.size == 0:
    #             raise HTTPException(status_code=404, detail='Ups, your query is wrong (date_to)')

    #     in_list = re.match(r'(?P<column>.*) in (?P<value>.*)', param)
    #     if in_list:
    #         list = in_list.group('value').split(" ")
    #         # print(list)
    #         dataframe = dataframe[dataframe[in_list.group('column')].isin(list)]
    #         if dataframe.size == 0:
    #             raise HTTPException(status_code=404, detail='Ups, your query is wrong (in)')

    #     is_value = re.match(r'(?P<column>.*) = (?P<value>.*)', param)
    #     if is_value:
    #         dataframe = dataframe[dataframe[is_value.group('column')] == is_value.group('value')]
    #         if dataframe.size == 0:
    #             raise HTTPException(status_code=404, detail='Ups, your query is wrong (=)')
    dataframe = query_match(df=dataframe, query=query, data_column='stan_rekordu_na')

    return dataframe

def get_powiat_present_data():
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

# def get_polish_data(df, columns, query):
#     dataframe = df
#     if len(columns) !=0:
#         dataframe = dataframe[columns]
#         if dataframe.size == 0:
#                 raise HTTPException(status_code=404, detail='Ups, your query is wrong (columns)')
#     for param in query:
#         # print(eval(param))
#         dataframe = dataframe[eval(param)]
#     if dataframe.size == 0:
#                 raise HTTPException(status_code=404, detail=f'Ups, your query is wrong: {param}')
    
#     return dataframe


# def validate_polish(data_source, dictionary):
    # if data_source in dictionary.keys():
        # return da
