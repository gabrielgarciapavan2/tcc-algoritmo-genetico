# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:54:01 2021

@author: Gabriel
"""

import pandas
import random

ativos2 = [
    "AAPL.csv",
    "AMZN.csv",
    "FB.csv",
    "GOOG.csv",
    "MSFT.csv",
    "NFLX.csv",
    "TSLA.csv",
    "BTC-USD.csv",
    "ETH-USD.csv",
    "LTC-USD.csv"
]

files_teste = ["AAPL.csv","AMZN.csv","BABA.csv","FB.csv","GOOG.csv","MSFT.csv","NFLX.csv","TSLA.csv","TWTR.csv","BTC-USD.csv"]

def gerarArquivos(nome_arquivo, files):
    f = pandas.DataFrame()
    asset_returns = pandas.DataFrame()
    
    u_data = pandas.DataFrame()
    a_data = pandas.DataFrame()
    
    for index, file in enumerate(files):
        csv = pandas.read_csv(f'AtivosTech/{file}', sep=',')
        csv = csv[["Date","Close"]]
        if index == 0:
            f["Date"] = csv["Date"]
        f[file] = csv["Close"]    
    
    for index, column in enumerate(f):
        if column == "Date":
            continue
        prices = f[column]
        prices_prev = f.iloc[0,index]
        returns = (prices - prices_prev) / prices_prev
        f[column] = returns
        
    
    for index, column in enumerate(f):
        if column == "Date":
            continue
        asset_returns[column] = f[column]
        
    asset_returns = asset_returns.iloc[1:,:]
        
    stocks_len = len(files)
    period_len = asset_returns.shape[0]
    
    u_matrix = [0] * stocks_len
    a_matrix = [[0 for i in range(stocks_len)] for j in range(period_len)]
    
    for index, column in enumerate(asset_returns):
        u_matrix[index] = asset_returns[column].mean()
        
    asset_returns_matrix = asset_returns.to_numpy()
    
    file = open("correcao.txt", "w")
    
    for index_stock in range(stocks_len):
        for index_period in range(period_len):
            a_matrix[index_period][index_stock] = u_matrix[index_stock] - asset_returns_matrix[index_period][index_stock]
    
    u_frame = pandas.DataFrame(u_matrix).T
    a_frame = pandas.DataFrame(a_matrix)
    c_frame = u_frame.append(a_frame)
    
    # u_frame.to_csv(f"Ativos3/U {nome_arquivo}.csv", index=True, sep=";")
    # a_frame.to_csv(f"Ativos3/A {nome_arquivo}.csv", index=True, sep=";")
    c_frame.to_csv(f"Ativos5/C {nome_arquivo}.csv", index=True, sep=";")
    
    return u_frame, a_frame, c_frame

gerarArquivos("TECH10", ativos2)
# gerarArquivos("200", random.sample(ativos, 200))
# gerarArquivos("300", random.sample(ativos, 300))
# gerarArquivos("400", random.sample(ativos, 400))
# gerarArquivos("500", random.sample(ativos, 500))
# gerarArquivos("600", random.sample(ativos, 600))
# gerarArquivos("700", random.sample(ativos, 700))
# gerarArquivos("800", random.sample(ativos, 800))
# gerarArquivos("900", random.sample(ativos, 900))
# gerarArquivos("1000", random.sample(ativos, 1000))
# gerarArquivos("1100", random.sample(ativos, 1100))
# gerarArquivos("1200", random.sample(ativos, 1200))
# gerarArquivos("1300", random.sample(ativos, 1300))
