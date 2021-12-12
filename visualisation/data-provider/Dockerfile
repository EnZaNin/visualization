FROM python:3.8

ENV APP_MODULE=covid.data-provider.main:covid_app

COPY ./requirements.txt ./

RUN pip install -r requirements.txt

COPY ./data-provider /covid
ENV PYTHONPATH=/

CMD ["uvicorn", "covid.data-provider.main:covid_app", "--reload", "--host", "0.0.0.0", "--port", "5000"]