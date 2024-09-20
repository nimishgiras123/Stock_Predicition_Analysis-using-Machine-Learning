import streamlit as st
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
# import mplfinance as mpf    # we were using this earlier but we shifted to yfinance
import yfinance as yf
import re
import pandas as pd
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

def get_google_finance_data(ticker, exchange):
    url = f"https://www.google.com/finance/quote/{ticker}:{exchange}?hl=en"
    response = requests.get(url)
    bs = BeautifulSoup(response.text, "html.parser")
    return float(bs.find(class_="YMlKec fxKbKc").text.strip()[1:].replace(",", ""))

def get_bhav_copy_equities(date):
    try:
        data_day = cm.bhav_copy_equities(date)
        return data_day
    except Exception as e:
        st.error("Data is not available for the desired date.")

def get_stock_details(stock_symbol, date_str):
    try:
        data = cm.bhav_copy_equities(date_str)
        data_1 = data[data['SYMBOL'] == stock_symbol]

        if not data_1.empty:
            st.success("Stock details fetched successfully!")

            # Display full stock details
            st.write("### Stock Details for", stock_symbol, "on", date_str)
            st.write(data_1)

        else:
            st.warning(f"No data available for stock symbol '{stock_symbol}' on the selected date.")
    except Exception as e:
        print("Error fetching stock details:", e)
        st.error("Error fetching stock details. Please try again.")


def get_stock_data(symbol, period):
    if period == "1w":
        start_date = pd.Timestamp.now() - pd.Timedelta(days=7)
    elif period == "1m":
        start_date = pd.Timestamp.now() - pd.Timedelta(days=30)
    elif period == "1y":
        start_date = pd.Timestamp.now() - pd.Timedelta(days=365)

    end_date = pd.Timestamp.now()

    data = yf.download(symbol, start=start_date, end=end_date)
    return data



def fetch_data():
    pattern = "₹[0-9 ,]+(?:\.\d{1,3})?"
    link_gain = "https://groww.in/markets/top-gainers"
    link_lose = "https://groww.in/markets/top-losers?index=GIDXNIFTY100"

    response_gain = requests.get(link_gain)
    response_lose = requests.get(link_lose)

    bs1 = BeautifulSoup(response_gain.text, "html.parser")
    bs2 = BeautifulSoup(response_lose.text, "html.parser")

    company_name_gain = bs1.find_all(class_="mtp438CompanyName bodyBase")
    prices_gain = bs1.find_all(class_="bodyBaseHeavy")

    company_name_lose = bs2.find_all(class_="mtp438CompanyName bodyBase")
    prices_lose = bs2.find_all(class_="bodyBaseHeavy")

    cname_gain = [i.text for i in company_name_gain]
    sprice_gain = [j.text for j in prices_gain]
    cname_lose = [i.text for i in company_name_lose]
    sprice_lose = [j.text for j in prices_lose]

    sgain = " ".join(sprice_gain)
    slose = " ".join(sprice_lose)
    lsgain = re.findall(pattern, sgain)
    lslose = re.findall(pattern, slose)

    losers = {}
    gainers = {}

    for i in range(10):  # Changed range to 10 because lists might have different lengths
        gainers[cname_gain[i]] = lsgain[i]
        losers[cname_lose[i]] = lslose[i]

    return gainers, losers


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


st.sidebar.title('Stock Details Fetcher')
com_stock_price = st.sidebar.selectbox("Select the official name of the company for stock info:", equity_list['NAME OF COMPANY'], key="stock_price_company_selectbox")
name_com_stock_price = None

if st.sidebar.button("Get Stock Price", key="get_price_button"):
    st.write(f"Company Name: {com_stock_price}")

    symbol_stock_price = equity_list.loc[equity_list['NAME OF COMPANY'] == com_stock_price, 'SYMBOL'].values
    if len(symbol_stock_price) > 0:
        name_com_stock_price = symbol_stock_price[0]

        st.write("Stock Price:")
        stock_price = get_google_finance_data(name_com_stock_price, "NSE")
        st.write(f"₹{stock_price}")

        st.write("Stock Information:")
        name_com_stock_price = symbol_stock_price[0] + ".NS"
        get_stock_info(name_com_stock_price)
    else:
        st.write("Please select a valid company name for stock price function.")


def get_stock_watchlist(selected_stock):
    df = cm.equity_list()

    symbol = df.loc[df['NAME OF COMPANY'] == selected_stock, 'SYMBOL'].values

    if len(symbol) > 0:
        name_com = symbol[0]
        exchange = "NSE"
        url = f"https://www.google.com/finance/quote/{name_com}:{exchange}?hl=en"

        # Make request to the URL
        response = requests.get(url)

        # Parse HTML with BeautifulSoup
        bs = BeautifulSoup(response.text, "html.parser")

        # Find current price
        curr = float(bs.find(class_="YMlKec fxKbKc").text.strip()[1:].replace(",", ""))

        # Find previous day close
        r = bs.find("div", class_="P6K39c").text.strip()

        return curr, r
    else:
        return None, None


def fetch_news():
    url = "https://www.livemint.com/market"
    req = requests.get(url)
    bs = BeautifulSoup(req.text, "html.parser")

    link_pattern = "https?://\S+?(?=\s|$)"

    res = bs.find_all(class_="market-new-common-collection_imgStory__hLGEJ imgStory fl")
    res_str = str(res)

    matches = re.findall(link_pattern, res_str)

    titles = [i.text for i in res]

    return titles, matches

st.sidebar.title('Price List Fetcher')
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


st.sidebar.title('Stock Price Prediction')
com_prediction = st.sidebar.selectbox("Select the official name of the company for prediction:", equity_list['NAME OF COMPANY'], key="prediction_company_selectbox")

if st.sidebar.button('Predict'):
    df = cm.equity_list()
    predict_stock_price(com_prediction, df)


st.sidebar.title("Top gainers and loosers of the Day")
fetch_button = st.sidebar.button("Fetch Data")

if fetch_button:
    with st.spinner('Fetching data...'):
        gainers, losers = fetch_data()

    st.subheader('Top Gainers:')
    gainers_df = pd.DataFrame(list(gainers.items()), columns=['Company', 'Price'])
    st.table(gainers_df)

    st.subheader('Top Losers:')
    losers_df = pd.DataFrame(list(losers.items()), columns=['Company', 'Price'])
    st.table(losers_df)


st.sidebar.title("Latest Stock News")
fetch_button = st.sidebar.button("Fetch News")

# Main content
if fetch_button:
    with st.spinner('Fetching news...'):
        titles, matches = fetch_news()

    st.subheader('Latest Market News:')
    for title, link in zip(titles, matches):
        st.write(f"{title}: {link}")

def get_stock_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data


def main():
    st.sidebar.title('Stock Data Visualization')

    # Input for stock symbol in the sidebar
    name = st.sidebar.text_input('Enter Stock Symbol (e.g., TCS):')
    symbol = f"{name}.NS"

    # Input for date range in the sidebar
    start_date = st.sidebar.date_input('Start Date:', value=pd.to_datetime("2023-02-19"))
    end_date = st.sidebar.date_input('End Date:', value=pd.to_datetime("2024-02-20"))

    # Get stock data
    if st.sidebar.button('Get Data'):
        stock_data = get_stock_data(symbol, start_date, end_date)

        # Plotting
        st.subheader('Stock Price Visuals')

        fig, axs = plt.subplots(4, 1, figsize=(10, 20))

        # Low Price
        axs[0].plot(stock_data['Low'], label='Low', color='blue')
        axs[0].set_title('Low Price')
        axs[0].set_xlabel('Date')
        axs[0].set_ylabel('Price')
        axs[0].grid(True)
        axs[0].spines[['top', 'right']].set_visible(False)

        # High Price
        axs[1].plot(stock_data['High'], label='High', color='green')
        axs[1].set_title('High Price')
        axs[1].set_xlabel('Date')
        axs[1].set_ylabel('Price')
        axs[1].grid(True)
        axs[1].spines[['top', 'right']].set_visible(False)

        # Open Price
        axs[2].plot(stock_data['Open'], label='Open', color='red')
        axs[2].set_title('Open Price')
        axs[2].set_xlabel('Date')
        axs[2].set_ylabel('Price')
        axs[2].grid(True)
        axs[2].spines[['top', 'right']].set_visible(False)

        # Close Price
        axs[3].plot(stock_data['Close'], label='Close', color='purple')
        axs[3].set_title('Close Price')
        axs[3].set_xlabel('Date')
        axs[3].set_ylabel('Price')
        axs[3].grid(True)
        axs[3].spines[['top', 'right']].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)


if __name__ == '__main__':
    main()

def fetch_stock_data(name):
    df = cm.equity_list()
    symbol = df.loc[df['NAME OF COMPANY'] == name, 'SYMBOL'].values

    if len(symbol) > 0:
        name_com = symbol[0]
        exchange = "NSE"
        url = f"https://www.google.com/finance/quote/{name_com}:{exchange}?hl=en"

        response = requests.get(url)
        bs = BeautifulSoup(response.text, "html.parser")

        try:
            curr = float(bs.find(class_="YMlKec fxKbKc").text.strip()[1:].replace(",", ""))
            prev_close = bs.find("div", class_="P6K39c").text.strip()

            return curr, prev_close
        except AttributeError:
            return None, None
    else:
        return None, None


def main():
    st.sidebar.title('Stock Watchlist')

    # Create a watchlist if it doesn't exist in the session state
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = []

    # Sidebar for adding and removing stocks
    st.sidebar.header('Manage Watchlist')

    # Get equity list from NSE
    df = cm.equity_list()
    company_names = df['NAME OF COMPANY'].tolist()

    # Input for adding stock to watchlist
    add_stock = st.sidebar.selectbox('Add Company to Watchlist:', company_names)

    if st.sidebar.button('Add'):
        st.subheader('Current Watchlist:')
        if add_stock not in st.session_state.watchlist:
            st.session_state.watchlist.append(add_stock)

    # Input for removing stock from watchlist
    remove_stock = st.sidebar.selectbox('Remove Company from Watchlist:', st.session_state.watchlist)

    if st.sidebar.button('Remove'):
        st.subheader('Current Watchlist:')
        st.session_state.watchlist.remove(remove_stock)

    # Display watchlist on the main screen
    # st.subheader('Current Watchlist:')
    for stock in st.session_state.watchlist:
        curr_price, prev_close = fetch_stock_data(stock)
        if curr_price and prev_close:
            st.write(f"{stock} - Current Price: ₹{curr_price}, Previous Day Close: {prev_close}")
        else:
            st.write(f"{stock} - Data Not Available")


if __name__ == '__main__':
    main()
