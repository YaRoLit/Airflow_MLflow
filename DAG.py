from datetime import datetime, timedelta, date
from fileinput import filename

import warnings

import pickle

import pandas as pd
import numpy as np

from airflow import task, DAG
from airflow.operators.python import PythonOperator

import mlflow
import mlflow.sklearn

from xmlrpc.client import DateTime
from telethon.sync import TelegramClient
from telethon.tl.functions.messages import GetDialogsRequest
from telethon.tl.types import InputPeerEmpty
from telethon.tl.functions.messages import GetHistoryRequest
from telethon.tl.types import PeerChannel

from sklearn.neighbors import KNeighborsRegressor
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.base import ForecastingHorizon
from sktime.performance_metrics.forecasting import  MeanAbsolutePercentageError, MeanSquaredError
smape = MeanAbsolutePercentageError(symmetric = True) 

warnings.filterwarnings("ignore")


# !!!! ВНИМАНИЕ !!!! Для запуска парсера нужно указать данные своего API-токена в Телеграм, без этого не заработает !!!!
api_id = ХХХХХХХ        # ============= здесь ИД
api_hash = 'хххххх'     # ============= здесь хэш
phone = '7913ХХХХХХХХ'  # ============= здесь номер телефона
#===============================================

link = 'https://t.me/+KxlX36pb-3hjMjRi'     # Ссылка на чат в Телеграм
msg_limit = 500                             # Количество сообщений для считывания

tmp_file = 'tmp_mess.csv'


mlflow.set_tracking_uri('http://0.0.0.0:5000') # Указываем местоположение сервера Airflow
experiment = mlflow.set_experiment("Airflow conv control")

print("mlflow tracking uri:", mlflow.tracking.get_tracking_uri())
print("experiment:", experiment)


def get_messages():
    '''Получаем N последних сообщений из чата (N=msg_limit и задается выше)'''

    client = TelegramClient(phone, api_id, api_hash)
    client.start()

    df = pd.DataFrame(columns=['datetime', 'user_id', 'text'])

    for message in client.get_messages(link, msg_limit):
        df.loc[len(df)] = (message.date, message.sender_id, message.text)

    df.to_csv(tmp_file, sep='\t', index=False, encoding='utf-8')


def concat_mess():
    '''Объединяем полученный фрэйм сообщений с файлом истории сообщений за весь период'''

    df = pd.read_csv(tmp_file, sep='\t', encoding='utf-8', parse_dates=['datetime'])

    try:
        fd = pd.read_csv('all_messages.csv', sep='\t', encoding='utf-8', parse_dates=['datetime'])
        
        df = pd.concat((fd, df), axis=0).drop_duplicates()
    
    except:
        df.to_csv('all_messages.csv', sep='\t', index=False, encoding='utf-8')


def data_preprocessing():
    '''Обработка сообщений - делаем почасовую разбивку и считаем количество сообщений за каждый час'''

    df = pd.read_csv(tmp_file, sep='\t', encoding='utf-8', parse_dates=['datetime'])
    fd = pd.read_csv('all_messages.csv', sep='\t', encoding='utf-8', parse_dates=['datetime'])

    df_group = df.groupby([pd.Grouper(key='datetime', freq='H')]).agg(txt_cnt=('text', 'count')).reset_index()
    fd_group = fd.groupby([pd.Grouper(key='datetime', freq='H')]).agg(txt_cnt=('text', 'count')).reset_index()

    df_group = df_group[df_group.datetime.dt.date == (date.today() - timedelta(days = 1))]
    df_group.to_csv(tmp_file, sep='\t', index=False, encoding='utf-8')

    fd_group = fd_group[fd_group.datetime.dt.date != date.today()]
    fd_group.to_csv('all_messages_cnt.csv', sep='\t', index=False, encoding='utf-8')


def check_metrics(ti):
    '''Проверяем метрику sMAPE по предсказанию модели и реальному количеству сообщений'''

    df = pd.read_csv(tmp_file, sep='\t', encoding='utf-8', parse_dates=['datetime'])

    y_val = np.array(df[df.datetime.dt.date == date.today() - timedelta(days = 1)].txt_cnt)
    
    y_pred = pickle.load(open('pred.mtr', 'rb'))

    with mlflow.start_run(experiment_id=experiment.experiment_id):
        try:
            score = smape(y_pred, y_val)
    
        except:
            score = 1

        print('============', y_val, '===========', y_pred, '===========', score)

        mlflow.log_param("y_pred", y_pred)
        mlflow.log_param("y_val", y_val)        
        mlflow.log_param("Score", score)

        ti.xcom_push('Score', score)


def remake_model(ti):
    '''Переобучаем модель если метрика недостаточно маленькая'''

    score = ti.xcom_pull(task_ids='check_metrics', key='Score')

    if score > 0.05:
        print('Model remaking start')

        df = pd.read_csv('all_messages_cnt.csv', sep='\t', encoding='utf-8', parse_dates=['datetime'])
        y = df.txt_cnt

        REGRESSION_WINDOW = 24*7

        regressor  = KNeighborsRegressor(n_neighbors=1)
        forecaster = make_reduction(regressor, window_length=REGRESSION_WINDOW, strategy="recursive")

        forecaster.fit(y)
        pickle.dump(forecaster, open('model.pkl', 'wb'))

        print('Model remaking end')
    
    else:
        print(f'Score {score} is good, remake model skiped.')


def new_predict():
    '''Делаем предсказание с помощью модели на следующие сутки'''

    fh = ForecastingHorizon(np.arange(1, 25))

    model = pickle.load(open('model.pkl', 'rb'))

    y_pred = np.array(model.predict(fh))

    pickle.dump(y_pred, open('pred.mtr', 'wb'))


args = {
    'owner': 'YARO',
    'start_date': datetime(2018, 11, 1),
    'provide_context': True
}


with DAG('Parsr_data',
         description='Parsr_TG_chat_hrly',
         schedule='1 0 * * *',
         catchup=False,
         default_args=args) as dag:

    task_1 = PythonOperator(task_id="get_messages", python_callable=get_messages)
    task_2 = PythonOperator(task_id="concatenate_messages", python_callable=concat_mess)
    task_3 = PythonOperator(task_id="data_preprocessing", python_callable=data_preprocessing)
    task_4 = PythonOperator(task_id="check_metrics", python_callable=check_metrics)
    task_5 = PythonOperator(task_id="remake_model", python_callable=remake_model)
    task_6 = PythonOperator(task_id="new_predict", python_callable=new_predict)


    task_1 >> task_2 >> task_3 >> task_4 >> task_5 >> task_6