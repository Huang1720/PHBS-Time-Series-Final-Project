import yfinance as yf
import pandas as pd

def get_yfinance_data(ticker, start_date, end_date):
    """获取 Yahoo Finance 数据"""
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        return pd.DataFrame()
    data = data.reset_index()
    data['Date'] = pd.to_datetime(data['Date'])
    data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    return data


data = get_yfinance_data('^GSPC', '1959-01-01', '2002-01-01')
data['date'] = pd.to_datetime(data['date'])
data["year"] = data["date"].dt.year
data['month'] = data['date'].dt.month

x_data = pd.read_csv('favar_reproduce/xdata.csv')
y_data = pd.read_csv('favar_reproduce/ydata.csv')
x_data["Unnamed: 0"] = pd.to_datetime(x_data["Unnamed: 0"], format="%Y:%m")
x_data = x_data.rename(columns={"Unnamed: 0": "date"})
y_data["Unnamed: 0"] = pd.to_datetime(y_data["Unnamed: 0"], format="%Y:%m").rename({"Unnamed: 0": "date"})
y_data = y_data.rename(columns={"Unnamed: 0": "date"})
x_data["month"] = x_data["date"].dt.month
y_data["month"] = y_data["date"].dt.month
x_data["year"] = x_data["date"].dt.year
y_data["year"] = y_data["date"].dt.year

month_col = x_data.columns[1:]
union_data = pd.merge(x_data,data, on=["month","year"],how = 'right')
union_data[month_col] = union_data[month_col].ffill()
union_data = union_data.drop(columns=['date_x'])

union_data.to_csv('favar_reproduce/x_union_data.csv', index=False)
pass