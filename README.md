# Macroeconomic Forecasting: ARIMA vs. Zero-Shot Foundation Models

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Optimized-red)
![Statsmodels](https://img.shields.io/badge/Statsmodels-Classical_Baseline-green)

This repository contains the complete quantitative research pipeline for benchmarking classical Box-Jenkins econometric models against state-of-the-art Time-Series Foundation Models (Amazon Chronos) in a highly volatile macroeconomic environment.

## 📖 Abstract Summary
Forecasting macroeconomic indicators is notoriously difficult due to structural breaks, external shocks, and non-stationary volatility. This project rigorously benchmarks the zero-shot predictive performance of a Foundation Model against meticulously tuned ARIMA models. Utilizing India’s monthly export and import data (April 1990 to February 2025), we evaluate both paradigms across a highly volatile 26-month out-of-sample test window (Jan 2022 - Feb 2025).

Evaluated via RMSE and MAE, the models exhibited a compelling divergence. For Exports, the zero-shot Chronos model yielded a superior RMSE (2788.72 vs 2807.77), effectively mitigating extreme prediction errors. However, for Imports, the mathematically grounded ARIMA(3,1,2) baseline drastically outperformed the Foundation Model (RMSE 4442.13 vs 6005.02). The findings offer critical implications for algorithmic systems design, proving that while AI can handle specific tail-risks, classical econometrics remains an indispensable, robust engine for structural economic forecasting.

## 🏗️ Architecture & Methodology

### 1. The Classical Pipeline (Baseline)
We establish a mathematically optimized statistical baseline rather than relying on arbitrary parameters:
* **Stationarity:** Augmented Dickey-Fuller (ADF) testing.
* **Optimization:** AIC-driven selection yielding ARIMA(3,1,3) for exports and ARIMA(3,1,2) for imports.
* **Diagnostics:** Ljung-Box and Jarque-Bera residual validation.

### 2. The AI Pipeline (Challenger)
* **Model:** `amazon/chronos-t5-base`
* **Execution:** Zero-shot probabilistic inference. The model tokenizes the historical time-series data akin to language modeling, generating forecasts without explicit differencing or fine-tuning.

## 📊 Final Benchmarks

**Exports:**
* ARIMA(3,1,3) -> RMSE: 2807.77 | MAE: 2286.24
* Chronos AI   -> RMSE: **2788.72** | MAE: 2293.52

**Imports:**
* ARIMA(3,1,2) -> RMSE: **4442.13** | MAE: **3774.50**
* Chronos AI   -> RMSE: 6005.02 | MAE: 4848.03

## 🚀 Reproduction Instructions
To replicate the findings locally or on Google Colab:

1. Clone this repository.
2. Ensure you have a GPU environment enabled (T4 is sufficient).
3. Install the required dependencies:
   ```bash
   pip install pandas numpy torch matplotlib seaborn statsmodels scipy chronos-forecasting transformers accelerate
