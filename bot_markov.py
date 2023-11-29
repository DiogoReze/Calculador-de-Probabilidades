import MetaTrader5 as mt5
from datetime import datetime
import time
import os
import pandas as pd
import numpy as np
import pandas_ta as ta

# display data on the MetaTrader 5 package
print("MetaTrader5 package author: ",mt5.__author__)
print("MetaTrader5 package version: ",mt5.__version__)

# establish connection to MetaTrader 5 terminal
if not mt5.initialize():
    print("initialize() failed, error code =",mt5.last_error())
    quit()

par = "EURCAD"
login = '51503177'
password = 'G2wyGNXJ'
server = 'ICMarketsSC-Demo'
timeFrame = '15M'
n_candles = 30

status = mt5.login(login, server,password)
if not status:
    print("Login MT5 OK")
    print(mt5.account_info())
else:
    print("Falha no login")

time.sleep(1)

print("Informações")
symbolInfo = mt5.symbol_info_tick('USDJPY')
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
    
    number_candles = []
    for i in range(len(rates)):
        if rates.close[i] - rates.open[i] > 0:
            number_candles.append(1)
        elif rates.close[i] - rates.open[i] < 0:
            number_candles.append(-1)
        else:
            number_candles.append(0)
    rates['number_candles'] = number_candles
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
    
    rates = rates.loc[:, 'ano':'stoch_933_s']
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
            
    rates = rates.loc[:,['ano', 'mes', 'dia', 'hora', 'minuto', 'segundo', 'number_candles', 'rsi_02', 'rsi_03', 'rsi_05', 'stoch_533', 'stoch_733', 'stoch_933']]
    return rates
    
symbolInfo = mt5.symbol_info_tick(par)
timestamp = symbolInfo[0]
hora, minuto, segundo, timeServer = obtemDatas(timestamp)

print(obterTaxas(par, timeFrame, timeServer, n_candles))