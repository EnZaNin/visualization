from fastapi import FastAPI

import urllib.request
import os
# import io
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt

file = 'owid-covid-data.csv'
URL = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/' + file
if not os.path.isfile(file):
    print('Download file from ', URL)
    urllib.request.urlretrieve(URL, file)
    print('Download success')
else:
    print(f'File {file} is already downloaded')

df = pd.read_csv('owid-covid-data.csv')


df_clear = df[['iso_code', 'location', 'date', 'population', 'total_cases', 'new_cases', 'total_deaths', 'new_deaths',
               'icu_patients', 'hosp_patients', 'total_tests', 'new_tests', 'tests_units', 'total_vaccinations',
               'people_vaccinated', 'people_fully_vaccinated', 'total_boosters', 'new_vaccinations']]
df_clear['date'] = df_clear['date'].astype('datetime64[ns]')

poland_df_clear = df_clear[df_clear['location'] == 'Poland']
poland_df_clear = poland_df_clear.append(df_clear[df_clear['location'] == 'Germany'])
poland_df_clear = poland_df_clear[poland_df_clear['date'].dt.year == 2021]
sns.set(rc={'figure.figsize': (11.7, 8.27)})
ax = sns.lineplot(data=poland_df_clear, x='date', y='new_cases', hue='location')
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45)
# plt.show()
fig = ax.get_figure()
fig.savefig('chart.png')

# bytes_image = io.BytesIO()
# plt.savefig(ax, format='png')
# bytes_image.seek(0)

from fastapi.responses import FileResponse

app_provider = FastAPI()

@app_provider.get("/")
async def main():
    return FileResponse("chart.png")
