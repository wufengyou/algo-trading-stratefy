import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.dates import DateFormatter
import numpy as np

def create_animated_chart(stock_data, portfolio, trades,output_file):
    # 設置圖表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Stock Price and Profit/Loss Over Time', fontsize=16)

    # 格式化 x 軸日期
    date_formatter = DateFormatter("%Y-%m-%d")
    ax1.xaxis.set_major_formatter(date_formatter)
    ax2.xaxis.set_major_formatter(date_formatter)

    # 設置 y 軸標籤
    ax1.set_ylabel('Stock Price')
    if output_file:
        ax1.set_ylabel(f'{output_file} Stock Price')
    ax2.set_ylabel('Profit/Loss')

    # 初始化線條
    line1, = ax1.plot([], [], lw=2, label='Stock Price')
    buy_scatter = ax1.scatter([], [], color='green', marker='^', s=100, label='Buy')
    sell_scatter = ax1.scatter([], [], color='red', marker='v', s=100, label='Sell')
    line2, = ax2.plot([], [], lw=2, label='Profit/Loss')

    # 設置圖例
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')

    # 設置 x 軸範圍
    ax1.set_xlim(stock_data.index[0], stock_data.index[-1])
    ax2.set_xlim(stock_data.index[0], stock_data.index[-1])

    # 設置 y 軸範圍
    ax1.set_ylim(stock_data['Close'].min() * 0.9, stock_data['Close'].max() * 1.1)
    max_profit_loss = max(abs(portfolio['total'].max() - portfolio['total'].iloc[0]),
                          abs(portfolio['total'].min() - portfolio['total'].iloc[0]))
    ax2.set_ylim(-max_profit_loss * 1.1, max_profit_loss * 1.1)

    # 初始化函數
    def init():
        line1.set_data([], [])
        buy_scatter.set_offsets(np.empty((0, 2)))
        sell_scatter.set_offsets(np.empty((0, 2)))
        line2.set_data([], [])
        return line1, buy_scatter, sell_scatter, line2

    # 更新函數
    def update(frame):
        # 更新股票價格
        line1.set_data(stock_data.index[:frame], stock_data['Close'][:frame])

        # 更新買入點
        buy_trades = [(trade[0], trade[1]) for trade in trades if trade[2] == 'buy' and trade[0] <= stock_data.index[frame]]
        if buy_trades:
            buy_dates, buy_prices = zip(*buy_trades)
            buy_scatter.set_offsets(np.column_stack((buy_dates, buy_prices)))
        else:
            buy_scatter.set_offsets(np.empty((0, 2)))

        # 更新賣出點
        sell_trades = [(trade[0], trade[1]) for trade in trades if trade[2] == 'sell' and trade[0] <= stock_data.index[frame]]
        if sell_trades:
            sell_dates, sell_prices = zip(*sell_trades)
            sell_scatter.set_offsets(np.column_stack((sell_dates, sell_prices)))
        else:
            sell_scatter.set_offsets(np.empty((0, 2)))

        # 更新獲利/虧損
        profit_loss = portfolio['total'][:frame] - portfolio['total'].iloc[0]
        line2.set_data(portfolio.index[:frame], profit_loss)

        return line1, buy_scatter, sell_scatter, line2

    # 創建動畫
    ani = animation.FuncAnimation(fig, update, frames=len(stock_data),
                                  init_func=init, blit=True, interval=10)

    # 顯示動畫
    plt.tight_layout()
    plt.show()

    # 如果要保存為視頻文件，可以取消下面的註釋
    if output_file:
        ani.save(f'{output_file}.mp4', writer='ffmpeg', fps=30)

# 在 analyze_stock 函數的末尾調用這個函數
# stock_data, portfolio, trades = analyze_stock('AAPL')
# create_animated_chart(stock_data, portfolio, trades,"stock_animation")