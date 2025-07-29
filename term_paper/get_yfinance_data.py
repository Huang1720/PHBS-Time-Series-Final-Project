import yfinance as yf
import pandas as pd

def get_yfinance_data(ticker, start_date, end_date):
    """获取 Yahoo Finance 数据"""
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        raise ValueError("Invalid ticker symbol or no data available")
    data = data.reset_index()
    data['Date'] = pd.to_datetime(data['Date'])
    data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    return data


data = get_yfinance_data('^GSPC', '1959-01-01', '2002-01-01')[['date','close']]
data['date'] = pd.to_datetime(data['date'])
data["year"] = data["date"].dt.year
data['month'] = data['date'].dt.month


x_data = pd.read_csv('favar_reproduce/xdata.csv')
y_data = pd.read_csv('favar_reproduce/ydata.csv')
x_data["Unnamed: 0"] = pd.to_datetime(x_data["Unnamed: 0"], format="%Y:%m")
x_data = x_data.rename(columns={"Unnamed: 0": "date"})
x_data['date'] = pd.to_datetime(x_data['date'], format="%Y:%m")
y_data["Unnamed: 0"] = pd.to_datetime(y_data["Unnamed: 0"], format="%Y:%m").rename({"Unnamed: 0": "date"})
y_data = y_data.rename(columns={"Unnamed: 0": "date"})
x_data["month"] = x_data["date"].dt.month
y_data["month"] = y_data["date"].dt.month
x_data["year"] = x_data["date"].dt.year
y_data["year"] = y_data["date"].dt.year


import akshare as ak
import pandas as pd


month_col = x_data.columns[1:]
union_data = pd.merge(x_data,data, on=["month","year"],how = 'right')
union_data[month_col] = union_data[month_col].ffill()
union_data = union_data.drop(columns=['date_x'])
union_data = union_data.rename(columns={'date_y': 'date'})

macro_tickers = {
    'dxy': 'DX-Y.NYB',               # 美元指数
    'fed_funds_future': 'ZQ=F',     # 联邦基金利率期货
    'treasury_10y': '^TNX',         # 10年期国债收益率（×10）
    'treasury_3m': '^IRX',          # 3月期国债收益率（×100）
    'volatility': '^VIX'          
}

# 下载所有宏观数据
macro_dfs = []
for name, ticker in macro_tickers.items():
    df = get_yfinance_data(ticker, start_date='1959-01-01', end_date='2002-01-01').set_index('date')[['close']]
    df = df.rename(columns={'close': name})
    macro_dfs.append(df)
    
    
macro_data = pd.concat(macro_dfs, axis=1,join='outer').reset_index()
union_data = pd.merge(union_data, macro_data, on='date', how='left').set_index('date')
full_index = pd.date_range(start=union_data.index.min(), end=union_data.index.max(), freq='D')  # 自然日
df_full = union_data.reindex(full_index).ffill()
df_full.to_csv('favar_reproduce/x_union_data.csv')
pass