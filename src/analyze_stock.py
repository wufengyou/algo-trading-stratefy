import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from sklearn.ensemble import RandomForestClassifier
from statsmodels.tsa.seasonal import STL
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

def analyze_stock(ticker, start_date='2020-01-01', end_date='2024-07-12', initial_capital=10000, transaction_cost=0.001):
    # 獲取數據
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    # 計算 SMA
    def sma(data, n):
        return data.rolling(window=n).mean()

    stock_data['sma_20'] = sma(stock_data['Close'], 20)
    stock_data['sma_50'] = sma(stock_data['Close'], 50)

    # 時間序列分解
    def decompose_time_series(data, column='Close', period=30):
        stl = STL(data[column], period=period)
        result = stl.fit()
        return result.trend

    stock_data['trend'] = decompose_time_series(stock_data)

    # 創建特徵
    def create_features(df, window=20):
        df['trend_change'] = df['trend'].pct_change(periods=window)
        df['volatility'] = df['Close'].rolling(window=window).std()
        df['momentum'] = df['Close'].pct_change(periods=window)
        return df

    stock_data = create_features(stock_data)

    # 準備機器學習數據
    stock_data['target'] = np.sign(stock_data['trend'].shift(-1) - stock_data['trend'])
    features = ['trend_change', 'volatility', 'momentum']

    # 刪除包含 NaN 的行
    stock_data_clean = stock_data.dropna()

    # 行走前向測試
    train_size = 252  # 使用一年的數據來初始訓練
    test_size = 20    # 每次預測未來20天

    stock_data['predicted_trend'] = np.nan

    for i in range(train_size, len(stock_data_clean), test_size):
        # 訓練數據
        X_train = stock_data_clean[features].iloc[max(0, i-train_size):i]
        y_train = stock_data_clean['target'].iloc[max(0, i-train_size):i]
        
        # 測試數據
        X_test = stock_data_clean[features].iloc[i:i+test_size]
        
        # 訓練模型
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # 預測
        predictions = rf.predict(X_test)
        stock_data.loc[X_test.index, 'predicted_trend'] = predictions

    # 實現交易策略
    def implement_realistic_strategy(close, sma1, sma2, predicted_trend, initial_capital=initial_capital, transaction_cost=transaction_cost):
        portfolio = pd.DataFrame(index=close.index)
        portfolio['close'] = close
        portfolio['sma1'] = sma1
        portfolio['sma2'] = sma2
        portfolio['predicted_trend'] = predicted_trend
        
        portfolio['position'] = 0
        portfolio['cash'] = initial_capital
        portfolio['holdings'] = 0
        portfolio['total'] = initial_capital
        
        trades = []
        
        for i in range(1, len(portfolio)):
            if pd.isna(portfolio['predicted_trend'].iloc[i-1]) or pd.isna(portfolio['sma1'].iloc[i-1]) or pd.isna(portfolio['sma2'].iloc[i-1]):
                portfolio['position'].iloc[i] = portfolio['position'].iloc[i-1]
                portfolio['cash'].iloc[i] = portfolio['cash'].iloc[i-1]
                portfolio['holdings'].iloc[i] = portfolio['position'].iloc[i] * portfolio['close'].iloc[i]
                portfolio['total'].iloc[i] = portfolio['cash'].iloc[i] + portfolio['holdings'].iloc[i]
                continue
            
            if (portfolio['sma1'].iloc[i-1] > portfolio['sma2'].iloc[i-1]) and (portfolio['predicted_trend'].iloc[i-1] > 0):
                signal = 1
            else:
                signal = -1
            
            prev_position = portfolio['position'].iloc[i-1]
            prev_cash = portfolio['cash'].iloc[i-1]
            prev_holdings = portfolio['holdings'].iloc[i-1]
            
            if signal == 1 and prev_position == 0:  # 買入
                max_shares = int(prev_cash / (portfolio['close'].iloc[i] * (1 + transaction_cost)))
                cost = max_shares * portfolio['close'].iloc[i] * (1 + transaction_cost)
                portfolio['position'].iloc[i] = max_shares
                portfolio['cash'].iloc[i] = prev_cash - cost
                portfolio['holdings'].iloc[i] = max_shares * portfolio['close'].iloc[i]
                trades.append((portfolio.index[i], portfolio['close'].iloc[i], 'buy'))
            elif signal == -1 and prev_position > 0:  # 賣出
                revenue = prev_position * portfolio['close'].iloc[i] * (1 - transaction_cost)
                portfolio['position'].iloc[i] = 0
                portfolio['cash'].iloc[i] = prev_cash + revenue
                portfolio['holdings'].iloc[i] = 0
                trades.append((portfolio.index[i], portfolio['close'].iloc[i], 'sell'))
            else:  # 保持不變
                portfolio['position'].iloc[i] = prev_position
                portfolio['cash'].iloc[i] = prev_cash
                portfolio['holdings'].iloc[i] = prev_position * portfolio['close'].iloc[i]
            
            portfolio['total'].iloc[i] = portfolio['cash'].iloc[i] + portfolio['holdings'].iloc[i]
        
        return portfolio, trades

    # 運行策略
    portfolio, trades = implement_realistic_strategy(
        stock_data['Close'], 
        stock_data['sma_20'], 
        stock_data['sma_50'], 
        stock_data['predicted_trend'],
        initial_capital=initial_capital,
        transaction_cost=transaction_cost
    )

    # 計算策略收益
    portfolio['returns'] = portfolio['total'].pct_change()
    portfolio['cumulative_returns'] = (portfolio['total'] / portfolio['total'].iloc[0]) - 1

    # 創建買賣訊號序列
    portfolio['signal'] = 0
    for date, _, action in trades:
        portfolio.loc[date, 'signal'] = 1 if action == 'buy' else -1 if action == 'sell' else 0

    # 繪製結果
    fig, axs = plt.subplots(4, 1, figsize=(15, 25))

    # 股票價格和交易點
    axs[0].plot(stock_data.index, stock_data['Close'], label=f'{ticker} Close', alpha=0.7)
    axs[0].plot(stock_data.index, stock_data['sma_20'], label='SMA 20', alpha=0.7)
    axs[0].plot(stock_data.index, stock_data['sma_50'], label='SMA 50', alpha=0.7)

    buy_dates = [trade[0] for trade in trades if trade[2] == 'buy']
    buy_prices = [trade[1] for trade in trades if trade[2] == 'buy']
    sell_dates = [trade[0] for trade in trades if trade[2] == 'sell']
    sell_prices = [trade[1] for trade in trades if trade[2] == 'sell']

    axs[0].scatter(buy_dates, buy_prices, color='green', marker='^', s=100, label='Buy')
    axs[0].scatter(sell_dates, sell_prices, color='red', marker='v', s=100, label='Sell')

    axs[0].set_title(f'{ticker} Price, SMAs, and Trade Points')
    axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Price')
    axs[0].legend()

    # 機器學習預測的趨勢
    axs[1].plot(stock_data.index, stock_data['predicted_trend'], label='Predicted Trend')
    axs[1].set_title('Machine Learning Predicted Trend')
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('Predicted Trend')
    axs[1].legend()

    # 買賣訊號
    axs[2].plot(portfolio.index, portfolio['signal'], label='Trade Signal')
    axs[2].set_title('Buy/Sell Signals Over Time')
    axs[2].set_xlabel('Date')
    axs[2].set_ylabel('Signal')
    axs[2].set_yticks([-1, 0, 1])
    axs[2].set_yticklabels(['Sell', 'Hold', 'Buy'])
    axs[2].legend()

    # 累積收益
    axs[3].plot(portfolio.index, portfolio['cumulative_returns'], label='Cumulative Returns')
    axs[3].set_title('Cumulative Returns Over Time')
    axs[3].set_xlabel('Date')
    axs[3].set_ylabel('Cumulative Returns')
    axs[3].legend()

    # 設置日期格式
    for ax in axs:
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    # 計算和打印性能指標
    def calculate_performance_metrics(cumulative_returns):
        total_return = cumulative_returns.iloc[-1]
        n_years = len(cumulative_returns) / 252  # 假設一年有252個交易日
        annualized_return = (1 + total_return) ** (1 / n_years) - 1
        daily_returns = cumulative_returns.pct_change().dropna()
        sharpe_ratio = daily_returns.mean() / (daily_returns.std() + 1e-6) * np.sqrt(252)  # 添加小常數避免除以零
        max_drawdown = (cumulative_returns.cummax() - cumulative_returns).max()
        
        return {
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown
        }

    performance = calculate_performance_metrics(portfolio['cumulative_returns'])
    print(f"Strategy Performance for {ticker}:")
    for metric, value in performance.items():
        print(f"{metric}: {value:.4f}")

    print(f"\nTotal number of trades: {len(trades)}")
    print(f"Number of buy trades: {len([t for t in trades if t[2] == 'buy'])}")
    print(f"Number of sell trades: {len([t for t in trades if t[2] == 'sell'])}")
    return stock_data, portfolio, trades