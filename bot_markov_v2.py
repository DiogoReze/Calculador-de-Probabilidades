import MetaTrader5 as mt5
from datetime import datetime
import time
import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# display data on the MetaTrader 5 package
print("MetaTrader5 package author: ",mt5.__author__)
print("MetaTrader5 package version: ",mt5.__version__)

# establish connection to MetaTrader 5 terminal
if not mt5.initialize():
    print("initialize() failed, error code =",mt5.last_error())
    quit()

#===============================================================================================
par = "EURUSD"
login = '51503177'
password = 'G2wyGNXJ'
server = 'ICMarketsSC-Demo'
timeFrame = '1M'
n_candles = 500
#===============================================================================================

status = mt5.login(login, server,password)
if not status:
    print("Login MT5 OK")
    print(mt5.account_info())
else:
    print("Falha no login")

time.sleep(1)

print("Informações")
symbolInfo = mt5.symbol_info_tick(par)
for i in symbolInfo:
    print(i)
print("oi----",symbolInfo[0])

def obtemDatas(timestamp):
    dt = datetime.fromtimestamp(timestamp)
    # Extrair os componentes de data e hora
    ano = dt.year; mes = dt.month; dia = dt.day; hora = dt.hour; minuto = dt.minute; segundo = dt.second
    timeServer = datetime(ano, mes, dia, hora, minuto, segundo)
    return hora, minuto, segundo, timeServer

def obterTaxas(par, timeFrame, timeServer, n_candles):
    if timeFrame == '1M':
        rates = mt5.copy_rates_from(par, mt5.TIMEFRAME_M1, timeServer, n_candles)    
    if timeFrame == '5M':
        rates = mt5.copy_rates_from(par, mt5.TIMEFRAME_M5, timeServer, n_candles)
    if timeFrame == '15M':
        rates = mt5.copy_rates_from(par, mt5.TIMEFRAME_M15, timeServer, n_candles)
    if timeFrame == '30M':
        rates = mt5.copy_rates_from(par, mt5.TIMEFRAME_M30, timeServer, n_candles)
    if timeFrame == 'H1':
        rates = mt5.copy_rates_from(par, mt5.TIMEFRAME_H1, timeServer, n_candles)
    if timeFrame == 'H4':
        rates = mt5.copy_rates_from(par, mt5.TIMEFRAME_H4, timeServer, n_candles)
    if timeFrame == 'H8':
        rates = mt5.copy_rates_from(par, mt5.TIMEFRAME_H8, timeServer, n_candles)
    if timeFrame == 'H12':
        rates = mt5.copy_rates_from(par, mt5.TIMEFRAME_H12, timeServer, n_candles)
    if timeFrame == 'D1':
        rates = mt5.copy_rates_from(par, mt5.TIMEFRAME_D1, timeServer, n_candles)
    rates = pd.DataFrame(rates)
    # convert time in seconds into the datetime format
    rates['time']=pd.to_datetime(rates['time'], unit='s')

    # Adicionando colunas separadas para ano, mês, dia, hora, minuto e segundo
    rates['ano'] = rates['time'].dt.year
    rates['mes'] = rates['time'].dt.month
    rates['dia'] = rates['time'].dt.day
    rates['hora'] = rates['time'].dt.hour
    rates['minuto'] = rates['time'].dt.minute
    rates['segundo'] = rates['time'].dt.second
    
    rates['number_candles'] = 0  
    rates.loc[rates['close'] > rates['open'], 'number_candles'] = 1
    rates.loc[rates['close'] < rates['open'], 'number_candles'] = -1
    rates['size'] = rates['close'] - rates['open']
    
    rates['rsi_02'] = ta.rsi(rates.close, 2)
    rates['rsi_03'] = ta.rsi(rates.close, 3)
    rates['rsi_05'] = ta.rsi(rates.close, 5)
    
    rsi_02 = []
    for i in range(len(rates)):
        if rates.rsi_02[i] > 90: rsi_02.append(1)
        elif rates.rsi_02[i] < 10: rsi_02.append(-1)
        else: rsi_02.append(0)
    rates['rsi_02'] = rsi_02
    
    rsi_03 = []
    for i in range(len(rates)):
        if rates.rsi_03[i] > 90: rsi_03.append(1)
        elif rates.rsi_03[i] < 10: rsi_03.append(-1)
        else: rsi_03.append(0)
    rates['rsi_03'] = rsi_03
    
    rsi_05 = []
    for i in range(len(rates)):
        if rates.rsi_05[i] > 80: rsi_05.append(1)
        elif rates.rsi_05[i] < 20: rsi_05.append(-1)
        else: rsi_05.append(0)
    rates['rsi_05'] = rsi_05
    
    stoch_df  = ta.stoch(rates.high, rates.low, rates.close, k=5, d=3, smooth_k=3)
    # rates['stoch_533_k'] = stoch_df['STOCHk_5_3_3']
    rates['stoch_533_s'] = stoch_df['STOCHd_5_3_3']
    
    stoch_df  = ta.stoch(rates.high, rates.low, rates.close, k=7, d=3, smooth_k=3)
    # rates['stoch_733_k'] = stoch_df['STOCHk_7_3_3']
    rates['stoch_733_s'] = stoch_df['STOCHd_7_3_3']
    
    stoch_df  = ta.stoch(rates.high, rates.low, rates.close, k=9, d=3, smooth_k=3)
    # rates['stoch_933_k'] = stoch_df['STOCHk_9_3_3']
    rates['stoch_933_s'] = stoch_df['STOCHd_9_3_3']
    
    # rates = rates.loc[:, 'ano':'stoch_933_s']
    rates = rates.dropna()    
    rates.reset_index(inplace=True)
    rates = rates.drop('index',axis=1)
    
    stoch_533 = []
    for i in range(len(rates)):
        if rates.stoch_533_s[i] > 80: stoch_533.append(1)
        elif rates.stoch_533_s[i] < 20: stoch_533.append(-1)
        else: stoch_533.append(0)
    rates['stoch_533'] = stoch_533

    stoch_733 = []
    for i in range(len(rates)):
        if rates.stoch_733_s[i] > 80: stoch_733.append(1)
        elif rates.stoch_733_s[i] < 20: stoch_733.append(-1)
        else: stoch_733.append(0)
    rates['stoch_733'] = stoch_733
    
    stoch_933 = []
    for i in range(len(rates)):
        if rates.stoch_933_s[i] > 80: stoch_933.append(1)
        elif rates.stoch_933_s[i] < 20: stoch_933.append(-1)
        else: stoch_933.append(0)
    rates['stoch_933'] = stoch_933
            
    rates = rates.loc[:,['ano', 'mes', 'dia', 'hora', 'minuto', 'segundo', 'number_candles', 'size', 'rsi_02', 'rsi_03', 'rsi_05', 'stoch_533', 'stoch_733', 'stoch_933']]
    # rates = rates.loc[:,['ano', 'mes', 'dia', 'hora', 'minuto', 'segundo', 'close', 'open', 'number_candles']]

    return rates

def checa1M(segundo):
    if segundo == 0:
        return True
    else:
        return False

def checa5M(minuto, segundo):
    #if (minuto ==  4 or minuto ==  9 or minuto == 14 or minuto == 19 or minuto == 24 or minuto == 29 or 
    #    minuto == 34 or minuto == 39 or minuto == 44 or minuto == 49 or minuto == 54 or minuto == 59) and segundo == 58:
    if (minuto == 0 or minuto == 5 or minuto == 10 or minuto == 15 or minuto == 20 or minuto == 25 or minuto == 30 or 
        minuto == 35 or minuto == 40 or minuto == 45 or minuto == 50 or minuto == 55) and segundo == 0:
        return True
    else:
        return False

def checa15M(minuto, segundo):
    #if (minuto == 14 or minuto == 29 or minuto == 44 or minuto == 59) and segundo == 58: # 15 minutos
    if (minuto == 15 or minuto == 30 or minuto == 45 or minuto == 0) and segundo == 0: # 15 minutos
        return True
    else:
        return False

def checa30M(minuto, segundo):
    #if (minuto == 29 or minuto == 59) and segundo == 58: # 15 minutos
    if (minuto == 30 or minuto == 0) and segundo == 0:
        return True
    else:
        return False

def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def horizontDados(X, y):
    dados, alvo = [], []
    for i in range(len(X)):
        dados.append(X[i].flatten())
        alvo.append(y[i].flatten())
    return np.array(dados), np.array(alvo)

#===============================================================================================
symbolInfo = mt5.symbol_info_tick(par)
timestamp = symbolInfo[0]
hora, minuto, segundo, timeServer = obtemDatas(timestamp)
temp = obterTaxas(par, timeFrame, timeServer, n_candles)
print(temp.shape)
temp = temp.loc[:,['size', 'rsi_02', 'rsi_03', 'rsi_05', 'stoch_533', 'stoch_733', 'stoch_933', 'number_candles']].copy()
temp = temp.iloc[:-1, :].copy()

print(temp)

temp_array = temp.to_numpy()
n_steps = 30
X, y = split_sequences(temp_array, n_steps)
n_features = X.shape[2]
print(X.shape,y.shape)
print(X[-1],y[-1])
print('================================================')

X, y = horizontDados(X, y)
print(X.shape,y.shape)
print(X[-1],y[-1])
print('================================================')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svm_classifier = SVC(probability=True)
svm_classifier.fit(X, y)
ultima_amostra = X[-1].reshape(1, -1)  # Reshape para garantir que seja uma matriz 2D
probabilidades_ultima_amostra = svm_classifier.predict_proba(ultima_amostra)
print("Probabilidades para a última amostra:")
print(f"Probabilidade de -1: {probabilidades_ultima_amostra[0, 0]:.3f}")
print(f"Probabilidade de 0: {probabilidades_ultima_amostra[0, 1]:.3f}")
print(f"Probabilidade de 1: {probabilidades_ultima_amostra[0, 2]:.3f}")

probabilidades_teste = svm_classifier.predict_proba(X_test)
previsoes_teste = np.argmax(probabilidades_teste, axis=1)
acuracia = accuracy_score(y_test, previsoes_teste)
acuracia_percentual = accuracy_score(y_test, previsoes_teste) * 100
print(f'Acurácia do modelo: {acuracia_percentual:.2f}%')
print(f'Acurácia do modelo: {acuracia:.3f}')
#===============================================================================================

'''dados = []
alvo = []
for i in range(len(X)):
  dados.append(X[i].flatten())
  alvo.append(y[i].flatten())
X = np.array(dados)
y = np.array(alvo)
print(X.shape,y.shape)
print(X[-1],y[-1])
'''

"""
while True:
    symbolInfo = mt5.symbol_info_tick(par)
    timestamp = symbolInfo[0]
    hora, minuto, segundo, timeServer = obtemDatas(timestamp)
    if timeFrame == '1M':
        if checa1M(segundo):
            symbolInfo = mt5.symbol_info_tick(par)
            timestamp = symbolInfo[0]
            hora, minuto, segundo, timeServer = obtemDatas(timestamp)
            print(obterTaxas(par, timeFrame, timeServer, n_candles).tail(10))
    if timeFrame == '5M':
        if checa5M(minuto, segundo):
            symbolInfo = mt5.symbol_info_tick(par)
            timestamp = symbolInfo[0]
            hora, minuto, segundo, timeServer = obtemDatas(timestamp)
            print(obterTaxas(par, timeFrame, timeServer, n_candles))
    if timeFrame == '15M':
        if checa15M(minuto, segundo):
            symbolInfo = mt5.symbol_info_tick(par)
            timestamp = symbolInfo[0]
            hora, minuto, segundo, timeServer = obtemDatas(timestamp)
            print(obterTaxas(par, timeFrame, timeServer, n_candles).tail(1))
    if timeFrame == '30M':
        if checa30M(minuto, segundo):
            symbolInfo = mt5.symbol_info_tick(par)
            timestamp = symbolInfo[0]
            hora, minuto, segundo, timeServer = obtemDatas(timestamp)
            print(obterTaxas(par, timeFrame, timeServer, n_candles))
    time.sleep(1)
"""