# Stock Market Analysis and Prediction

This project provides an accessible and educational platform for new investors entering the stock market. It integrates live data, machine learning-based prediction models, and educational content into a user-friendly interface.

## Overview

Stock market investing can be daunting for beginners due to a lack of clear guidance and resources. Our project aims to address this gap by combining live stock market insights with educational material and predictive models to empower users with tools for making informed investment decisions.

**Key Features:**
- Real-time stock data collection from APIs
- Interactive data visualizations for trend analysis
- Machine learning models to predict stock price trends
- Community features for collaborative learning and insights
- Comprehensive educational content on stock market strategies

## Problem Statement

The project addresses the issue of limited accessible educational resources for beginner investors. By integrating real-time data and clear, actionable insights, we help users overcome the overwhelming amount of information and empower them to make well-informed decisions.

## Technical Implementation

### Tech Stack:
- **Python**: The core programming language for data analysis, machine learning models, and web scraping.
- **Streamlit**: For creating a simple, interactive UI that supports live visualizations.
- **Pyforest**: Lazy imports for data science libraries such as Pandas, Matplotlib, and Scikit-learn.
- **BeautifulSoup**: For scraping real-time stock data and financial news.
- **NSElib & Nsetools**: Real-time data retrieval from the National Stock Exchange (NSE) of India.
- **Matplotlib & Plotly**: Visualization libraries for creating interactive charts and trend analysis.
- **Scikit-learn & TensorFlow**: For building and training machine learning models that predict stock prices based on historical data.

### Machine Learning Models:
- **Linear Regression**: Used for basic trend forecasting by modeling the relationship between historical stock prices and time.
- **Random Forest**: Provides more robust predictions by using multiple decision trees to capture complex patterns in stock price movement.
- **LSTM (Long Short-Term Memory)**: A type of recurrent neural network (RNN) designed to handle sequential data, ideal for time-series prediction of stock prices.

## Visualizations

The following visualizations were used in the app to help users analyze market trends and performance:

1. **Line Graphs**: Showing stock price movements over time with interactive capabilities.
  
2. **Candlestick Charts**: Visual representation of stock price movements for specific time intervals, including high, low, open, and close values.
   
3. **Moving Average Visualization**: For identifying stock price trends by smoothing out short-term fluctuations.

4. **Prediction Graphs**: Display the results of predictive models, showing projected future stock prices compared to actual historical data.
   

## Results

The app effectively addressed the challenge of providing beginner investors with real-time data and accurate predictions. **Key outcomes**:
- **Prediction Accuracy**: Achieved an average prediction accuracy of **96% to 99.9%** using machine learning models.
- **User Feedback**: Users found the platform highly intuitive, particularly appreciating the educational content alongside the live stock insights.
- **Prediction Models**: The app uses models like LSTM and Random Forest for highly accurate stock trend forecasts.

## Limitations

While this app has a number of strengths, there are some limitations:
- **Data Dependency**: The app relies heavily on real-time data from external APIs. Delays or errors in this data could affect analysis.
- **Model Limitations**: While the machine learning models are accurate, they may not account for sudden, unforeseen events like market crashes or global news.
- **Security Concerns**: Handling live stock data can expose users to security vulnerabilities, which needs constant attention to avoid data breaches.

## Future Work

Planned improvements include:
- Adding more advanced prediction models like deep learning-based architectures.
- Extending educational content to advanced investment strategies.
- Building a stronger user community through enhanced collaborative features.

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.x
- Libraries: streamlit, matplotlib, plotly, scikit-learn, tensorflow

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-market-analysis.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Contributors

- Parth Dadhich
- Vansh Arora
- Shiv Kumar Vijay
- Nimish Gigras

