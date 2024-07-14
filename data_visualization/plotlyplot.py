import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_dynamic_chart(stock_data, portfolio,trades):
    # 創建一個有兩個子圖的圖表
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, subplot_titles=('Stock Price', 'Profit/Loss'))

    # 添加股票價格線
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Stock Price'),
                  row=1, col=1)

    # 添加買入點
    fig.add_trace(go.Scatter(x=[], y=[], mode='markers', name='Buy',
                             marker=dict(symbol='triangle-up', size=10, color='green')),
                  row=1, col=1)

    # 添加賣出點
    fig.add_trace(go.Scatter(x=[], y=[], mode='markers', name='Sell',
                             marker=dict(symbol='triangle-down', size=10, color='red')),
                  row=1, col=1)

    # 添加獲利/虧損線
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Profit/Loss'),
                  row=2, col=1)

    # 更新佈局
    fig.update_layout(height=800, title_text="Dynamic Stock Trading Visualization")
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Profit/Loss", row=2, col=1)

    # 創建動態更新函數
    def update_chart(frame):
        date = stock_data.index[frame]
        price = stock_data['Close'].iloc[frame]
        profit = portfolio['total'].iloc[frame] - portfolio['total'].iloc[0]

        # 更新股票價格
        fig.data[0].x = stock_data.index[:frame+1]
        fig.data[0].y = stock_data['Close'][:frame+1]

        # 更新買入點
        buy_dates = [trade[0] for trade in trades if trade[2] == 'buy' and trade[0] <= date]
        buy_prices = [trade[1] for trade in trades if trade[2] == 'buy' and trade[0] <= date]
        fig.data[1].x = buy_dates
        fig.data[1].y = buy_prices

        # 更新賣出點
        sell_dates = [trade[0] for trade in trades if trade[2] == 'sell' and trade[0] <= date]
        sell_prices = [trade[1] for trade in trades if trade[2] == 'sell' and trade[0] <= date]
        fig.data[2].x = sell_dates
        fig.data[2].y = sell_prices

        # 更新獲利/虧損
        fig.data[3].x = portfolio.index[:frame+1]
        fig.data[3].y = portfolio['total'][:frame+1] - portfolio['total'].iloc[0]

        return fig

    # 創建動畫
    animation = go.Figure(
        data=fig.data,
        layout=fig.layout,
        frames=[go.Frame(data=update_chart(i).data) for i in range(len(stock_data))]
    )

    # 顯示圖表
    animation.show()



# 在 analyze_stock 函數的末尾調用這個函數
# create_dynamic_chart(stock_data, portfolio,trades)


