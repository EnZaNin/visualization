import urllib.request
import os
import re

import numpy as np
import pandas as pd
import datetime as dt
from fastapi import HTTPException
import matplotlib as mpl
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
    This function comes from:
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

    elif days_in_chart >= 365:
        major_locator = mdates.MonthLocator()
        minor_locator = mdates.WeekdayLocator()
    elif days_in_chart < 30:
        major_locator = mdates.DayLocator()
        minor_locator = mdates.DayLocator()
    else:
        major_locator = mdates.WeekdayLocator()
        minor_locator = mdates.DayLocator()
    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_minor_locator(minor_locator)


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
        plt.figure(figsize=(10, 8))

        ax = sns.scatterplot(data=principal_df, x='principal component 1', hue=column_filter, y='principal component 2',
                             s=60)
        for i, txt in enumerate(df_y):
            ax.annotate(txt, (
                principal_df['principal component 1'].values[i], principal_df['principal component 2'].values[i]))
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
        ax.set_xlabel(
            f'Principal Component 1: '
            f'{round(pca.explained_variance_ratio_[0] * 100, 2)}% \n of explained variance ratio in data',
            fontsize=15)
        ax.set_ylabel(
            f'Principal Component 2: '
            f'{round(pca.explained_variance_ratio_[1] * 100, 2)}% \n of explained variance ratio in data',
            fontsize=15)
        ax.set_title(
            f'From {len(columns_for_pca)} to 2 Component PCA: '
            f'{round(np.cumsum(pca.explained_variance_ratio_ * 100)[1], 2)}% of explained variance ratio'
            f' in data - {chart_name}',
            fontsize=20)
        chart_name = chart_name + '.png'
        ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        plt.savefig(chart_name, bbox_inches='tight')
        return chart_name
    else:
        raise HTTPException(status_code=404,
                            detail=f'Sorry, for n_component = {n_components} method is yet not allowed')


def mds_method(chart_name, dataframe, column_filter, columns_for_mds):
    dataframe = dataframe.groupby([column_filter]).mean()
    dataframe = dataframe.reset_index(level=[column_filter])
    df_x = dataframe.loc[:, columns_for_mds].values
    df_y = dataframe.loc[:, column_filter].values
    x = StandardScaler().fit_transform(df_x)
    # x = df_x
    model2d = MDS(n_components=2)
    x_trans = model2d.fit_transform(x)
    # print('The new shape of X: ', X_trans.shape)
    # print('No. of Iterations: ', model2d.n_iter_)
    # print('Stress: ', model2d.stress_)
    mds_df = pd.DataFrame(data=x_trans, columns=['Standard X', 'Standard Y'])

    mds_df[column_filter] = df_y
    plt.clf()
    plt.figure(figsize=(10, 8))
    ax = sns.scatterplot(data=mds_df, x='Standard X', hue=column_filter, y='Standard Y', s=50)
    ax.set_title(
        f'The new shape of X: {x_trans.shape}, No. of Iterations: {model2d.n_iter_}, Stress: {model2d.stress_}',
        fontsize=20)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    chart_name = chart_name + '.png'
    plt.savefig(chart_name, bbox_inches='tight')
    return chart_name
