import streamlit as st
import requests
# from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import mplfinance as mpf
import yfinance as yf
from nselib import capital_market as cm
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
st.markdown("""
    <head>
        <link href='https://fonts.googleapis.com/css2?family=Roboto+Condensed&display=swap' rel='stylesheet'>
        <style>
            .custom-font {
                font-family: 'Roboto Condensed', sans-serif;
            }
            .custom-header {
                color: #333333;
                font-weight: bold;
                font-size: 40px;
                margin-bottom: 0;
                text-shadow: 2px 2px #f5f5dc; /* Add text shadow */
            }
            .custom-container {
                text-align: center;
                padding: 10px; /* Decrease padding */
                background-color: #f5f5dc;
                border-radius: 10px;
                border-bottom: 2px solid #cccccc; /* Add bottom border */
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Add box shadow */
            }
            .custom-paragraph {
                font-size: 18px;
                margin-top: 10px;
                line-height: 1.5; /* Adjust line height */
                color: rgba(51, 51, 51, 0.8); /* Slightly lighter color with reduced opacity */
            }
        </style>
    </head>
    <div class='custom-container'>
        <h1 class='custom-header custom-font'>Stock Market Analysis and Prediction App</h1>
        <p class='custom-paragraph custom-font'>Explore market trends, fetch stock prices, and predict future stock prices with ease.</p>
    </div>
""", unsafe_allow_html=True)


def get_stock_info(stock_name):
    try:
        stock = yf.Ticker(stock_name)
        info_dict = stock.info
        ls = ["country", "website", "industry", "priceToBook", "totalCash", "ebitda", "previousClose",
              "payoutRatio", "debtToEquity", "trailingPE"]

        definitions = {
            "country": "The country where the company is headquartered.",
            "website": "The official website of the company.",
            "industry": "The sector in which the company operates.",
            "priceToBook": "The ratio of the company's stock price to its book value per share.",
            "totalCash": "The total cash available to the company.",
            "ebitda": "Earnings Before Interest, Taxes, Depreciation, and Amortization (EBITDA) is a measure of a company's operating performance.",
            "previousClose": "The previous day's closing price of the stock.",
            "payoutRatio": "The ratio of dividends paid to shareholders to the company's net income.",
            "debtToEquity": "The ratio of the company's debt to its equity, indicating its financial leverage.",
            "trailingPE": "The price-to-earnings (P/E) ratio calculated by dividing the current stock price by the earnings per share (EPS) for the past 12 months."
        }

        for i in ls:
            st.write(f"**{i.capitalize()}**: {info_dict.get(i, 'N/A')}")

            if i in definitions:
                st.write(f"[*Definition*: {definitions[i]}]")

    except Exception as e:
        st.error("The data of the desired stock is not available.")

# def get_google_finance_data(ticker, exchange):
#     url = f"https://www.google.com/finance/quote/{ticker}:{exchange}?hl=en"
#     response = requests.get(url)
#     bs = BeautifulSoup(response.text, "html.parser")
#     return float(bs.find(class_="YMlKec fxKbKc").text.strip()[1:].replace(",", ""))
# def get_google_finance_data(ticker,exchange):
#      ticker = yf.Ticker(symbol)
#     current_price = ticker.history(period='1d')['Close'].iloc[-1]
#     return current_price







def get_current_price(symbol):
    try:
        ticker = yf.Ticker(symbol+".NS")
        current_price = ticker.history(period='1d')['Close'].iloc[-1]
        return current_price
    except Exception as e:
        st.error(f"Error fetching current stock price: {e}")
        return None
    



def get_bhav_copy_equities(date):
    try:
        data_day = cm.bhav_copy_equities(date)
        return data_day
    except Exception as e:
        st.error("Data is not available for the desired date.")

def get_stock_details(stock_symbol, date_str):
    try:
        data = cm.bhav_copy_equities(date_str)
        data_1= data[data['SYMBOL'] == stock_symbol]
        if data_1 is not None:
            st.success("Stock details fetched successfully!")
            st.write("### Stock Details for", stock_symbol, "on", date_str)
            st.write(data_1)
        else:
            st.warning(f"No data available for stock symbol '{stock_symbol}' on the selected date.")
    except Exception as e:
        print("Error fetching stock details:", e)
        return None

st.sidebar.markdown("""
    <style>
        .sidebar-header {
            color: #f5f5dc;
            font-family: 'Roboto Condensed', sans-serif;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 20px;
            border-bottom: 2px solid #cccccc;
            padding-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)
st.sidebar.markdown("<p class='sidebar-header'>Features</p>", unsafe_allow_html=True)



equity_list = cm.equity_list()
com_stock_price = st.sidebar.selectbox("Select the official name of the company for stock price:", equity_list['NAME OF COMPANY'], key="stock_price_company_selectbox")
name_com_stock_price = None

if st.sidebar.button("Get Stock Price", key="get_price_button"):
    st.write(f"Company Name: {com_stock_price}")

    symbol_stock_price = equity_list.loc[equity_list['NAME OF COMPANY'] == com_stock_price, 'SYMBOL'].values
    if len(symbol_stock_price) > 0:
        name_com_stock_price = symbol_stock_price[0]

        st.write("Stock Price:")
        # stock_price = get_google_finance_data(name_com_stock_price, "NSE")
        stock_price= get_current_price(name_com_stock_price)
        st.write(f"₹{stock_price}")

        st.write("Stock Information:")
        name_com_stock_price = symbol_stock_price[0] + ".NS"
        get_stock_info(name_com_stock_price)
    else:
        st.write("Please select a valid company name for stock price function.")


date_input = st.sidebar.date_input("Select a date:", datetime.today())

date_str = date_input.strftime("%d-%m-%Y")

if st.sidebar.button("Fetch Bhav Copy", key="fetch_bhav_copy_button"):
    with st.spinner("Fetching Bhav Copy data..."):
        data_day = get_bhav_copy_equities(date_str)
        if not data_day is None:  # Check if DataFrame is not empty
            st.success("Bhav Copy data fetched successfully!")
            st.write("### Bhav Copy for", date_str)
            st.write(data_day)
        else:
            st.warning("No Bhav Copy data available for the selected date. As market was not open.")


stock_symbol = st.sidebar.text_input("Enter stock ticker (e.g., TCS):").upper()


if st.sidebar.button("Fetch Stock Details", key="fetch_stock_details_button"):
    if not stock_symbol:
        st.warning("Please enter a stock ticker.")
    else:
        with st.spinner("Fetching stock details..."):
            get_stock_details(stock_symbol, date_str)

st.set_option('deprecation.showPyplotGlobalUse', False)

def predict_stock_price(company_name, df):
    symbol = df.loc[df['NAME OF COMPANY'] == company_name, 'SYMBOL'].values
    if len(symbol) == 0:
        st.error("Company not found. Please enter a valid company name.")
        return

    name_com = symbol[0] + ".NS"
    try:
        tick = yf.Ticker(name_com)
        historical_data = tick.history(period="1y")
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return

    if historical_data.empty:
        st.error("No historical data found for the specified company.")
        return

    historical_data['Previous Close'] = historical_data['Close'].shift(1)
    historical_data.dropna(inplace=True)

    X = historical_data[['Previous Close']]
    y = historical_data['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)

    future_data = historical_data[['Previous Close']].iloc[-1].values.reshape(1, -1)
    future_price_prediction = model.predict(future_data)[0]

    accuracy_percentage = r2 * 100

    if future_price_prediction > historical_data['Close'].iloc[-1]:
        investment_suggestion = "It is suggested to consider investing in the stock."
    else:
        investment_suggestion = "It is suggested to not invest in the stock."


    plt.figure(figsize=(10, 6))
    plt.plot(historical_data.index, historical_data['Close'], label='Historical Closing Prices')
    plt.axvline(x=historical_data.index[-1], color='r', linestyle='--', linewidth=1, label='End of Historical Data')
    plt.axhline(y=future_price_prediction, color='g', linestyle='--', linewidth=1, label='Predicted Future Price')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.title('Historical Closing Prices and Predicted Future Price')
    plt.legend()
    plt.grid(True)
    st.pyplot()

    st.write(f"Predicted Future Price: {future_price_prediction}")
    st.write(f"Accuracy of the model: {accuracy_percentage:.2f}%")
    st.write(investment_suggestion)


com_prediction = st.sidebar.selectbox("Select the official name of the company for prediction:", equity_list['NAME OF COMPANY'], key="prediction_company_selectbox")

st.sidebar.title('Stock Price Prediction')
if st.sidebar.button('Predict'):
    df = cm.equity_list()
    predict_stock_price(com_prediction, df)

