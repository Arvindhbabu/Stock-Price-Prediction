# Stock Price Prediction with LSTMs

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/) [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
This project implements a stock price prediction model using Long Short-Term Memory (LSTM) neural networks. It fetches historical stock data via yfinance, preprocesses it, trains an LSTM model to predict future closing prices, evaluates the model with RMSE, and visualizes results with Matplotlib. The implementation is in a Jupyter Notebook, suitable for exploring time series forecasting in finance.

LSTMs are effective for sequential data like stock prices due to their ability to capture long-term dependencies. This project provides a practical example of applying deep learning to financial prediction tasks.

## Problem Statement
Develop a stock price prediction model using Long Short-Term Memory (LSTM) networks to forecast future closing prices based on historical data. The system should fetch real-time stock data for a specified ticker (e.g., AAPL) using yfinance, preprocess it with scaling and sequence creation, train an LSTM model, evaluate its performance with RMSE, visualize training loss and predictions, and save the model, demonstrating time series forecasting in financial data analysis.

## Features
- **Data Fetching**: Retrieves historical stock data using yfinance for any ticker.
- **Preprocessing**: Scales data and creates time sequences for LSTM input.
- **Model Building**: Constructs a multi-layer LSTM with dropout for regularization.
- **Training and Evaluation**: Trains the model, predicts on test data, and computes RMSE.
- **Visualization**: Plots training/validation loss and actual vs. predicted prices.
- **Model Saving**: Exports the trained model in HDF5 format.

## Requirements
- Python 3.x
- Jupyter Notebook (or JupyterLab) for running the `.ipynb` file
- Required packages:
  - yfinance
  - numpy
  - pandas
  - scikit-learn
  - tensorflow
  - matplotlib
