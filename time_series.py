import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
import statsmodels.api as sm
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings("ignore")

rcParams['figure.figsize'] = 15, 10

#Читаем данные из файла Данные.xlsx
data = pd.read_excel('Данные.xlsx', index_col='Date',parse_dates=['Date'], 
                      engine="openpyxl")
data.sort_values('Date', ascending = True)
data = data.dropna(axis = 1)
print(data)
print (data.isna().sum())

#Рисуем график ряда
plt.figure(figsize=(10,5))
plt.plot(data.Value,'red', label='Временной ряд')
plt.plot(data.Value.rolling(window=7).mean(), 'blue', 
                            label='Скользящее среднее')
plt.plot(data.Value.rolling(window=7).std(), 'green', 
                            label='Скользящее стандартное отклонение')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

#Тест Дики-Фуллера
def aug_dickey_fuller(y):
    test = sm.tsa.adfuller(y)
    print ('adf: ', test[0]) 
    print ('p-value: ', test[1])
    print('Critical values: ', test[4])
    if (test[0]> test[4]['5%']): 
        print ('есть единичные корни, ряд не стационарен')
        return False
    else:
        print ('единичных корней нет, ряд стационарен')
        return True

#Проверяем ряд на стационарность
aug_dickey_fuller(data.Value)

#Аддитивная модель
add = sm.tsa.seasonal_decompose(data.Value, 'additive')
add.plot()
plt.show()
print("ТРЕНД")
aug_dickey_fuller(add.trend.dropna())
print("СЕЗОНАЛЬНОСТЬ")
aug_dickey_fuller(add.seasonal.dropna())
print("ОСТАТОК")
aug_dickey_fuller(add.resid.dropna())

#Мультипликативная модель
mult = sm.tsa.seasonal_decompose(data.Value, 'multiplicate')
mult.plot()
plt.show()
print("ТРЕНД")
aug_dickey_fuller(mult.trend.dropna())
print("СЕЗОНАЛЬНОСТЬ")
aug_dickey_fuller(mult.seasonal.dropna())
print("ОСТАТОК")
aug_dickey_fuller(mult.resid.dropna())

#Функция определения порядка интегрированности временного ряда
def order(y):
    time_series = y
    order = 0
    while (not aug_dickey_fuller(time_series)):
        order += 1
        time_series = time_series.diff().dropna()
    return order

#Проверяем порядок интегрированности
time_series = data.Value
order = order(time_series)
print("Порядок интегрированности - ", order)

#Строим графики автокорреляции и частичной автокорреляции
time_series = data.Value.diff(periods=1).dropna()
fig = plt.figure(figsize=(15,10))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(time_series, lags=60, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(time_series, lags=60, ax=ax2)

#Читаем данные из файла Ответы.xlsx
data_test = pd.read_excel('Ответы.xlsx', index_col='Date',
                          parse_dates=['Date'], engine="openpyxl")
data_test.sort_values('Date', ascending=True)
data_test = data_test.dropna(axis=1)
print (data_test.isna().sum())

#Вывод данных о ARIMA модели
def arima(data, order, test):
    model = sm.tsa.ARIMA(data, order=order, freq='MS').fit()
    pred = model.predict(start=data.shape[0], 
                         end=data.shape[0]+test.shape[0]-1, typ='levels')
    plt.clf()
    plt.plot(data_test.Value, 'blue', label='anwer')
    plt.plot(pred, 'red', label='pred' + str(order))
    plt.legend(loc='upper left')
    plt.show()
    print("R2-metrics", r2_score(pred, test))
    print("AIC :", model.aic)
    print(model.summary())
    return pred

#Тесты ARIMA модели
pred1 = arima(data, (2,1,1), data_test)
pred2 = arima(data, (12, 1, 3), data_test)
pred3 = arima(data, (12, 1, 4), data_test)
pred4 = arima(data, (12, 1, 2), data_test)
pred5 = arima(data, (8, 1, 2), data_test)
pred6 = arima(data, (8, 1, 6), data_test)
pred7 = arima(data, (6, 1, 4), data_test)
pred8 = arima(data, (7, 1, 2), data_test)
pred9 = arima(data, (25, 1, 4), data_test)

#Сравнительный график результатов
plt.plot(data_test.Value, 'black', label='anwer')
plt.plot(pred1, 'purple', label='prediction 1 - (2,1,1)')
plt.plot(pred2, 'green', label='prediction 2 - (12, 1, 3)')
plt.plot(pred3, 'blue', label='prediction 3 - (12, 1, 4)')
plt.plot(pred4, 'coral', label='prediction 4 - (12, 1, 2)')
plt.plot(pred5, 'brown', label='prediction 5 - (8, 1, 2)')
plt.plot(pred6, 'red', label='prediction 6 - (8, 1, 6)')
plt.plot(pred7, 'yellow', label='prediction 7 - (6, 1, 4)')
plt.plot(pred8, 'magenta', label='prediction 8 - (7, 1, 2)')
plt.legend(loc='upper left')
plt.show()