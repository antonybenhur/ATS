import pandas as pd 
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
import pandas_datareader.data as web
from plotly import graph_objs as go
from keras.models import load_model
from datetime import date
import streamlit as st
import yfinance as yf
from nsetools import Nse
from autots import AutoTS
from tqdm.notebook import tqdm
from time import sleep
import finplot as fplt
import mplfinance as mpf



# creating a Nse object
nse = Nse()
 


st.title('AI Machine Learning - Predictive Analytic Engine')

user_input = st.text_input('Enter Stock Ticker','INFY')
user_input = user_input.upper()
stdt = st.date_input('Start', value = pd.to_datetime('2019-01-01'))
endt = st.date_input('End', value = pd.to_datetime('today'))

if user_input.endswith('.NS'):
    nse_user_input = user_input.replace(".NS","")
    #print (user_input)
else:
    nse_user_input = user_input

if nse.is_valid_code(nse_user_input):
    st.success("Valid Ticker")
else:
    st.error("Invalid Ticker")
    st.stop()

# getting quote of the ticker
quote = nse.get_quote(nse_user_input)


quote_lp = quote.get("lastPrice")

st.header(quote.get("companyName"))
st.subheader('Last Price: '+str(quote.get("lastPrice")))
st.caption('Day High: '+str(quote.get("dayHigh")))
st.caption('Day Low: '+str(quote.get("dayLow")))
st.caption('Previous Close: '+str(quote.get("previousClose")))

top_gainers = nse.get_top_gainers()
top_gainers = pd.DataFrame(nse.get_top_gainers())
top_losers = pd.DataFrame(nse.get_top_losers())


col1,col2 = st.columns(2)
with col1:
    st.subheader("NSE Top Gainers")
    col1.write(top_gainers)
with col2:
    st.subheader("NSE Top Losers")
    col2.write(top_losers)


#START = "2019-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
#ticker = "^NSEI"
df = yf.download(user_input, stdt, endt)
df.reset_index(inplace=True)
#st.subheader( "Past 5 Days Trading - "+"("+ quote.get("companyName")+")")
st.write(df.tail(10))


def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=df['Date'], y=df['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="stock_close"))
	fig.layout.update(title_text='Actual Stock Trend for '+ user_input, xaxis_rangeslider_visible=True)
	st.plotly_chart(fig,use_container_width=False)

plot_raw_data()

# Candlestick Chart
#style.use('ggplot')
#df_ohlc = df['Adj Close'].resample('10D').ohlc()
#df_volume = df['Volume'].resample('10D').sum()
#df_ohlc.reset_index(inplace=True)
#df_ohlc['Date']=df_ohlc['Date'].map(mdates.date2num)
#ax1=plt.subplot2grid((6,1),(0,0),rowspan=5,colspan=1)
#ax2=plt.subplot2grid((6,1),(5,0),rowspan=5, colspan=1,sharex=ax1)
#ax1.xaxis_date()
#candlestick_ohlc(ax1,df_ohlc.values,width=5,colorup='g')
#ax2.fill_between(df_volume.index.map(mdates.date2num),df_volume.values,0)
#plt.show()
#---------------

#dff = yf.download('AAPL',start='2018-01-01', end = '2020-04-29')
#fplt.candlestick_ochl(dff[['Open','Close','High','Low']])
#fplt.plot(dff.Close.rolling(50).mean())
#fplt.plot(dff.Close.rolling(200).mean())
#fplt.show()

#plotdf = df.drop(['Adj Close'],axis=1)
#mpf.plot(plotdf)

#Stochaic RSI----------------------------------------- 
bt_RSI_start = st.button('Start Stochaic RSI Analysis')
if bt_RSI_start:
#Set date as index
    df = df.set_index(pd.DatetimeIndex(df['Date'].values))

#Create a Exponential Moving Average indicator function
    def EMA(data, period=20, column='Close'):
        return data[column].ewm(span=period,adjust=False).mean()

#Create a function to calculate the Stochastic Relative Strength Index
    def StochRSI(data, period=14, column='Close'):
        delta = data[column].diff(1)
        delta = delta.dropna()
        up = delta.copy()
        down = delta.copy()
        up[up<0]=0 #change all values less than 0 to 0
        down[down>0]=0 #change all values greater than 0 to 0
        data['up'] = up
        data['down'] = down
        AVG_Gain = EMA(data, period, column='up')
        AVG_Loss = abs(EMA(data, period, column='down'))
        RS = AVG_Gain/AVG_Loss
        RSI = 100.0 - (100.0/(1.0 + RS))

        stockrsi = (RSI-RSI.rolling(period).min()) / (RSI.rolling(period).max() - RSI.rolling(period).min())
    
        return stockrsi

#Store the stochastic RSI Data
    df['StochRSI'] = StochRSI(df)
    #Plot the data
    #Create a fig and a set of subplots
    fig,(ax1,ax2) = plt.subplots(nrows=2,sharex=True)
    #Remove vertical gaps
    plt.subplots_adjust(hspace=.0)
    ax1.grid()
    ax2.grid()
    #Plot the close price
    ax1.plot(df.index,df['Close'],color = 'r')
    #pot the stochRSI
    ax2.plot(df.index,df['StochRSI'],color = 'y', linestyle = '-')
    #Plot oversold(.2) and overbought .8 lines
    ax2.axhline(.2, linewidth=4, color='r')
    ax2.axhline(.8, linewidth=4, color='r')
    #Rotate Xtick by 45 degrees
    plt.xticks(rotation=45)
    st.plotly_chart(fig,use_container_width=False)

#Stochaic RSI----------------------------------------- 





# ATOS Modelling ---------------------------------------
forecast_len = st.slider('Forecast Length in Days', min_value=10, max_value=120, value=10, step=10)
max_gen = st.slider('Maximum Generations', min_value=1, max_value=15, value=5, step=1)

bt_model_start = st.button('Start ATOS Predictive Analysis')

if bt_model_start:
    st.write("Prediction Started...Please Wait...")
    model = AutoTS(
            forecast_length=forecast_len, 
            frequency='infer', 
            ensemble='auto',
            holiday_country='IN',
            validation_method = 'backwards',
            max_generations= max_gen
            )
    model = model.fit(df, date_col='Date', value_col='Close', id_col=None)
    prediction = model.predict()
    forecast = prediction.forecast
    st.write("Prediction Completed...")
    st.subheader('Forecast Table')
    st.write(forecast)
    st.subheader('Forecast Plot')
    fig2 = plt.figure(figsize = (12,6))
    plt.plot(forecast.Close)
    st.pyplot(fig2,use_container_width=False)
    #st.line_chart(forecast)
# ATOS Modelling ---------------------------------------



# Prophet Modeling------------------------------------------
bt_prophet_start = st.button('Start Prophet Predictive Analysis')
if bt_prophet_start:
    #Select the date and price
    df = df[['Date','Close']]
    #Rename Features
    df = df.rename(columns = {'Date':'ds', 'Close':'y'})
    #import Prophet library
    from fbprophet import Prophet
    from fbprophet.plot import plot_plotly
    #creating the prophet obj
    fbp = Prophet(daily_seasonality = True)
    #Fit or Train the Model
    fbp.fit(df)
    future = fbp.make_future_dataframe(periods=365)
    proph_forecast = fbp.predict(future)
    proph_forecast = proph_forecast[['ds','yhat']]
    st.write(proph_forecast.tail(30))
    #plot_plotly(fbp,forecast)

# Prophet Modeling------------------------------------------

#train_df = pd.DataFrame(df,columns=['Date','Close'])
#print(train_df.shape)
#test_df = pd.DataFrame(df,columns=['Date','Close'][int(len(df)*0.70): int(len(df))])
#print(test_df.shape)

#fig = plt.figure(figsize = (12,6))
#plt.plot(df.Close)
#plt.show()







