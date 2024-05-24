# Bitcoin Trend Prediction Using Neural Network 💸

Welcome to our project repository for predicting Bitcoin trends using neural networks! In this project, we aim to forecast the direction of Bitcoin prices by leveraging sentiment analysis, technical indicators, and various other data sources.

## Contributors 🧑‍💻
- Jehoiada Wong
- Axel Nino Nakata

## Overview ℹ️

Bitcoin, a decentralized digital currency, has garnered significant attention as both a mode of transaction and investment. In this project, we utilize neural networks to predict Bitcoin price trends, incorporating sentiment analysis, technical indicators, and additional data sources.

## Preprocessing 🛠️

In the preprocessing phase, we perform several crucial steps to prepare the data for modeling:

- **Data Cleaning:** We carefully clean the raw data to remove any inconsistencies, missing values, or outliers that could adversely affect the model's performance.
- **Feature Engineering:** We engineer new features to capture meaningful relationships and temporal dependencies in the data. This includes creating time lagged variables to incorporate historical information.
- **Encoding:** We encode categorical variables and normalize numerical features to ensure compatibility with the neural network architecture.

## Technical Indicators 📈

We leverage various technical indicators to extract insights from the Bitcoin price data:

- **Stochastic Indicator:** The stochastic oscillator helps us assess the momentum of Bitcoin prices relative to recent price ranges. This indicator is valuable for identifying potential overbought or oversold conditions.
- **Moving Averages:** We calculate both simple and exponential moving averages to smooth out price fluctuations and identify trends in Bitcoin prices over different time periods.

## LSTM-RNN Usage 📈

We employ Long Short-Term Memory (LSTM) recurrent neural networks (RNNs) for modeling sequential data in our project. LSTM networks excel at capturing long-term dependencies in time-series data, making them well-suited for predicting Bitcoin price trends. For a detailed understanding of our LSTM-RNN implementation and its effectiveness in Bitcoin trend prediction, refer to the literature research we have conducted before.

## Data Sources 📊

We incorporate several sources of information for this project:

- **Volume Data:** Trading volume data provides insights into market liquidity and investor participation. High trading volumes often coincide with significant price movements, reflecting increased market activity.
- **Price Action Data:** Analysis of historical price movements helps us identify patterns and trends that may influence future price movements. This includes identifying support and resistance levels, as well as chart patterns such as triangles and head-and-shoulders formations.
- **Fear and Greed Index Data:** We utilize sentiment data from Twitter and Reddit to gauge market sentiment using the Fear and Greed Index. This index measures the overall sentiment of market participants, indicating whether investors are feeling fearful or greedy. By incorporating sentiment analysis, we aim to capture market sentiment and its role in driving price movements.

## Feedback 📝
Feel free to enhance the code, provide feedback, or contribute to the project by opening issues or pull requests. Your input is highly appreciated!
