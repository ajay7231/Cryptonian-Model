# Prophet is basically a forecasting model that can be used to predict future values of time series data.
# yahoo finance to get the crypto data
import streamlit as st
from datetime import date
import requests

import yfinance as yf
from fbprophet import Prophet

from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

url = 'https://api.exchangerate-api.com/v4/latest/USD'

# Get the data from the API


class RealTimeCurrencyConverter():
    def __init__(self, url):
        self.data = requests.get(url).json()
        self.currencies = self.data['rates']
    # Get the conversion rate

    def convert(self, from_currency, to_currency, amount):
        initial_amount = amount
        # first convert it into USD if it is not in USD.
        # because our base currency is USD
        if from_currency != 'USD':
            amount = amount / self.currencies[from_currency]

        # limiting the precision to 4 decimal places
        amount = round(amount * self.currencies[to_currency], 4)
        return amount


START = "2015-01-01"  # start date
# today's date and convert it into string format
TODAY = date.today().strftime("%Y-%m-%d")

currencyConverter = RealTimeCurrencyConverter(url)

st.title("Crypto Prediction App")  # title of the app

currencies = ["USD",
              "EUR",
              "GBP",
              "AUD",
              "CAD",
              "CHF",
              "CNY",
              "DKK",
              "HKD",
              "INR",
              "JPY",
              "KRW",
              "NZD",
              "PLN",
              "RUB",
              "SEK",
              "SGD",
              "THB",
              "TRY"]


cryptos = ("BTC-USD", "ETH-USD", "BNB-USD",
           "USDT-USD", "HEX-USD", "SOL-USD", "DOGE-USD", "SHIB-USD")

# select the crypto from the select box and return it into selected_cryptos variable
selected_cryptos = st.selectbox("Select dataset for prediction", cryptos)
currencies = st.selectbox("Select currencies", currencies)

# slider to select the number of years of prediction from 1 to 4
n_years = st.slider("Years Of prediction:", 1, 4)

period = n_years * 365  # number of days in a year


@st.cache  # store the donwloaded data in the cache so we don't have to download it again
def load_data(ticker):  # ticker is basically the selected_cryptos variable
    # to download the data from yahoo finance for a specified start and end date
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)  # This puts date in the very first column
    return data


# text to show the loading of data
data_load_state = st.text("Loading data...")

data = load_data(selected_cryptos)  # synchronous call to function
# after the data is loaded, we can plot the raw data
data_load_state.text("Loading data...done!")

st.subheader('Raw Data')  # subheader for raw data
st.write(data.tail())  # display the last 5 rows of the data


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],
                  y=data['Open'], name='crypto_open'))
    fig.add_trace(go.Scatter(x=data['Date'],
                  y=data['Close'], name='crypto_close'))
    fig.layout.update(title_text="Time Series Data",
                      xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_raw_data()

# Forecasting()
df_train = data[['Date', 'Close']]
# we have to rename because prophet expects the column names to be ds and y
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})


m = Prophet()  # Creating a model
m.fit(df_train)  # fitting the training data

# dataframe for the future data ; period is the number of days we want to predict
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)  # predicting the future data

st.subheader('Forecast data')  # subheader for forecast data
st.write(forecast.tail())  # display the last 5 rows of the forecast data


st.write('forecast data')  # display the forecast data
fig1 = plot_plotly(m, forecast)  # plot the forecast data
st.plotly_chart(fig1)   # plot the forecast data

st.write('forecast components')  # display the forecast components
fig2 = m.plot_components(forecast)  # plot the forecast components
st.write(fig2)  # plot the forecast components
