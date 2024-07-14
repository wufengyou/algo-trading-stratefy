import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from sklearn.ensemble import RandomForestClassifier
from statsmodels.tsa.seasonal import STL
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")
from src.analyze_stock import analyze_stock
from data_visualization.plotlyplot import create_dynamic_chart
from data_visualization.matplotlib_ani import create_animated_chart

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 使用函數
# stock_data, portfolio, trades=analyze_stock('AAPL')  # 分析蘋果公司股票
# stock_data, portfolio, trades= analyze_stock('MSFT')  # 分析微軟公司股票
# stock_data, portfolio, trades= analyze_stock('GOOGL')  # 分析谷歌公司股票
if __name__=='__main__':
    
    ticker = 'GOOGL'
    start_date = '2020-01-01'    
    end_date = '2023-07-15'
    initial_capital=10000 
    transaction_cost=0.001
    logger.info("Fetching stock data...")
    stock_data, portfolio, trades= analyze_stock(ticker,start_date=start_date,end_date=end_date,initial_capital=initial_capital,transaction_cost=transaction_cost)  # 分析谷歌公司股票
    #analyze_stock(ticker, start_date='2020-01-01', end_date='2024-07-12', initial_capital=10000, transaction_cost=0.001):
    # create_dynamic_chart(stock_data, portfolio,trades)
    create_animated_chart(stock_data, portfolio, trades,output_file=None)



