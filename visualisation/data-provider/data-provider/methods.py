import urllib.request
import os
import re

import numpy as np
import pandas as pd
import datetime as dt
from fastapi import HTTPException
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS

import logging

logger = logging.getLogger(__name__)


def download_file(file, url):
    url = url + file
    if not os.path.isfile(file):
        logger.info('Download file from', url)
        urllib.request.urlretrieve(url, file)
        logger.info('Download success')
    else:
        logger.info(f'File {file} is already downloaded')


def query_match(df, query, data_column):
    dataframe = df

    for param in query:
        date_from = re.match(r'(?P<column>.*) > (?P<value>.*)', param)
        if date_from:
            if date_from.group('column') == 'date_from':
                date_from_datetype = dt.datetime.strptime(date_from.group('value'), '%Y-%m-%d')
                dataframe = dataframe[dataframe[data_column] > date_from_datetype]
            else:
                dataframe = dataframe[dataframe[date_from.group('column')] > int(date_from.group('value'))]
            if dataframe.size == 0:
                raise HTTPException(status_code=404, detail='Ups, your query is wrong (date_from)')

        date_to = re.match(r'(?P<column>.*) < (?P<value>.*)', param)
        if date_to:
            if date_to.group('column') == 'date_to':
                date_to_datetype = dt.datetime.strptime(date_to.group('value'), '%Y-%m-%d')
                dataframe = dataframe[dataframe[data_column] < date_to_datetype]
            else:
                dataframe = dataframe[dataframe[date_to.group('column')] < int(date_to.group('value'))]
            if dataframe.size == 0:
                raise HTTPException(status_code=404, detail='Ups, your query is wrong (date_to)')

        in_list = re.match(r'(?P<column>.*) in (?P<value>.*)', param)
        if in_list:
            list = in_list.group('value').split(" ")
            dataframe = dataframe[dataframe[in_list.group('column')].isin(list)]
            if dataframe.size == 0:
                raise HTTPException(status_code=404, detail='Ups, your query is wrong (in)')

        is_value = re.match(r'(?P<column>.*) = (?P<value>.*)', param)
        if is_value:
            dataframe = dataframe[dataframe[is_value.group('column')] == is_value.group('value')]
            if dataframe.size == 0:
                raise HTTPException(status_code=404, detail='Ups, your query is wrong (=)')

    return dataframe


def get_filtered_data(df, columns, query):
    dataframe = df
    if len(columns) != 0:
        dataframe = dataframe[columns]
        if dataframe.size == 0:
            raise HTTPException(status_code=404, detail='Ups, your query is wrong (columns)')
    for param in query:
        dataframe = dataframe[eval(param)]
    if dataframe.size == 0:
        raise HTTPException(status_code=404, detail=f'Ups, your query is wrong: {param}')

    return dataframe


def auto_scaler_date(min_date: dt, max_date: dt, ax):
    '''
    This function is ... from:
    https://www.programcreek.com/python/?code=TimRivoli%2FStock-Price-Trade-Analyzer%2FStock-Price-Trade-Analyzer-master%2F_classes%2FPriceTradeAnalyzer.py
    '''

    if type(min_date) == str:
        min_date = dt.datetime.strptime(min_date, '%Y-%m-%d')
        max_date = dt.datetime.strptime(max_date, '%Y-%m-%d')
        days_in_chart = (max_date - min_date).days
    else:
        days_in_chart = (max_date - min_date).days
    if days_in_chart >= 365 * 2:
        major_locator = mdates.YearLocator()
        minor_locator = mdates.MonthLocator()
        # majorFormatter = mdates.DateFormatter('%m/%d/%Y')

    elif days_in_chart >= 365:
        major_locator = mdates.MonthLocator()
        minor_locator = mdates.WeekdayLocator()
        # majorFormatter = mdates.DateFormatter('%m/%d/%Y')
    elif days_in_chart < 30:
        major_locator = mdates.DayLocator()
        minor_locator = mdates.DayLocator()
        # majorFormatter = mdates.DateFormatter('%m/%d/%Y')
    else:
        major_locator = mdates.WeekdayLocator()
        minor_locator = mdates.DayLocator()
        # majorFormatter = mdates.DateFormatter('%m/%d/%Y')
    ax.xaxis.set_major_locator(major_locator)
    # ax.xaxis.set_major_formatter(majorFormatter)
    ax.xaxis.set_minor_locator(minor_locator)
    # ax.xaxis.set_minor_formatter(daysFmt)
    # ax.set_xlim(min_date, max_date)
    # return ax


def pca_method(chart_name, dataframe, columns_for_pca, column_filter, n_components):
    dataframe = dataframe.groupby([column_filter]).mean()
    dataframe = dataframe.reset_index(level=[column_filter])
    indexs = []
    for idx, row in dataframe.iterrows():
        if row.isna().sum() >= 0.3 * len(columns_for_pca):
            indexs.append(idx)
    for idx in indexs:
        dataframe = dataframe.drop(index=idx)
    df_x = dataframe.loc[:, columns_for_pca].values
    df_y = dataframe.loc[:, column_filter].values
    pca = PCA(n_components=n_components)
    x = StandardScaler().fit_transform(df_x)
    principal_components = pca.fit_transform(x)
    if n_components == 2:
        principal_df = pd.DataFrame(data=principal_components,
                                    columns=['principal component 1', 'principal component 2'])
        principal_df[column_filter] = df_y
        plt.clf()
        plt.figure(figsize=(16, 8))

        ax = sns.scatterplot(data=principal_df, x='principal component 1', hue=column_filter, y='principal component 2')
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
        ax.set_xlabel(f'Principal Component 1: {pca.explained_variance_ratio_[0] * 100}', fontsize=15)
        ax.set_ylabel(f'Principal Component 2: {pca.explained_variance_ratio_[1] * 100}', fontsize=15)
        ax.set_title(f'{chart_name} to 2 Component PCA: {np.cumsum(pca.explained_variance_ratio_ * 100)[1]}',
                     fontsize=20)
        chart_name = chart_name + '.png'
        plt.savefig(chart_name)
    return chart_name


def mds_method(chart_name, dataframe, column_filter, columns_for_pca):
    dataframe = dataframe.groupby([column_filter]).mean()
    dataframe = dataframe.reset_index(level=[column_filter])
    # indexs = []
    # for idx, row in dataframe.iterrows():
    #     if row.isna().sum() >= 0.5 * len(columns_for_pca):
    #         indexs.append(idx)
    # for idx in indexs:
    #     dataframe = dataframe.drop(index=idx)
    df_x = dataframe.loc[:, columns_for_pca].values
    df_y = dataframe.loc[:, column_filter].values
    x = StandardScaler().fit_transform(df_x)
    model2d = MDS(n_components=2)
    # x = df_x
    X_trans = model2d.fit_transform(x)
    print('The new shape of X: ', X_trans.shape)
    print('No. of Iterations: ', model2d.n_iter_)
    print('Stress: ', model2d.stress_)
    principal_df = pd.DataFrame(data=X_trans,
                                columns=['principal component 1', 'principal component 2'])

    principal_df[column_filter] = df_y
    plt.clf()
    plt.figure(figsize=(16, 8))
    print(X_trans)
    ax = sns.scatterplot(data=principal_df, x='principal component 1', hue=column_filter, y='principal component 2')
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    # ax.set_xlabel(f'Principal Component 1: {pca.explained_variance_ratio_[0] * 100}', fontsize=15)
    # ax.set_ylabel(f'Principal Component 2: {pca.explained_variance_ratio_[1] * 100}', fontsize=15)
    # ax.set_title(f'{chart_name} to 2 Component PCA: {np.cumsum(pca.explained_variance_ratio_ * 100)[1]}',
    #              fontsize=20)
    print(principal_df)
    chart_name = chart_name
    plt.savefig(chart_name)
