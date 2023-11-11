# 3-е учебное задание по МЛОПС

Выполнил магистрант Литаврин Я.И.

**!!! Внимание !!! Данный скрипт для запуска требует указания идентификационных данных API-токена мессенджера Telegram, так как исходную информацию для работы модели получает из него (парсит наш учебный групповой чат). Свои идентификационные данные я не могу размещать в открытом репозитории, так как это очень чувствительная информация. Я могу продемонстрировать работоспособность скрипта в онлайн-режиме любым удобным для Вас способом. Также для проверки Вы можете указать собственные персональные данные API-токена Telegram.**

Общий принцип работы конвейера:

Конвейер настроен на ежедневный запуск в 00:01:00 и разделен на 6 последовательных операций.

**task_1:** Производится считывание 500 последних сообщений из отслеживаемого Телеграм-чата. Этого вполне достаточно, так как более чем в три раза превышает максимальную суточную активность за всю историю наблюдений. Можно увеличить число отслеживаемых сообщений или поменять ссылку на чат - это задается константами до объявления функций.

**task_2:** Объединяются два файла, в одном из которых содержатся все сообщения из чата за всю историю наблюдения, а в другом - полученные в 1-й задаче последние 500 сообщений. Файл с историей необходим для более качественного обучения модели, так как её скользящее окно составляет 24 * 7 временных лагов (в данном случае - часов) и при необходимости может быть увеличено. При объединении дубликаты убираются.

**task_3:** Осуществляется препроцессинг сообщений. Так как модель работает с временными рядами, а полученные сообщения ещё таковыми не являются, переводим их в формат временного ряда. Для этого сутки разбиваются на равномерные отрезки длительностью по часу и для каждого из этих отрезков подсчитывается количество сообщений, которые отправлялись в чат в этот период. Делается два временных ряда - первый для общей истории сообщений и второй для последних суток, сохраняются в отдельные файлы.

**task_4:** Проверяется метрика модели. ***Внимание! сама модель для этого не загружается!*** Мы просто берем предсказание модели на текущие сутки, которое было получено при прошлом её запуске, и сравниваем это предсказание с теми данными о количестве сообщений (с разбивкой по часам), которые мы получили в шагах 1-3. Считаем метрику sMAPE, которую отсылаем в MLFlow вместе с двумя кортежами - реального количества сообщений и предсказанного.

**task_5:** Если метрика получилась полная хрень (а для моей модели, к сожалению, это обычная ситуация), то переучиваем модель на файле с историей сообщений (переведенном в формат временного ряда) с обновлением за последние сутки. 

**task_6:** С помощью модели делаем предсказание о количестве сообщений в чате для следующих (то есть уже 1 минуту как наступивших) суток. 


**В репозитории также имеется файл юпитерского ноутбука, в котором была попытка провести EDA и выбрать оптимальный алгоритм и гиперпараметры модели. На самом деле это сильно обрезанный вариант, я перебирал многие при выполнении итогового проекта по МОМО, в данном файле указан лучший. К сожалению, даже лучшее из того, что у меня получилось, имеет крайне низкую метрику, но времени на дальнейшие эксперименты, увы, не осталось.**

Ниже привожу некоторые скрины работы моего конвейера в Airflow и его выгрузку в MLflow:
![Image alt](https://github.com/YaRoLit/Airflow_MLflow/raw/main/Screenshots/airflow_4.png)

![Image alt](https://github.com/YaRoLit/Airflow_MLflow/raw/main/Screenshots/airflow.png)

![Image alt](https://github.com/YaRoLit/Airflow_MLflow/raw/main/Screenshots/airflow2.png)

![Image alt](https://github.com/YaRoLit/Airflow_MLflow/raw/main/Screenshots/airflow3.png)

![Image alt](https://github.com/YaRoLit/Airflow_MLflow/raw/main/Screenshots/mlflow_1.png)

![Image alt](https://github.com/YaRoLit/Airflow_MLflow/raw/main/Screenshots/mlflow_2.png)
