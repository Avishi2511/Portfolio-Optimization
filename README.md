# AI-Powered Portfolio Optimization with XAI 

## Overview

This project optimizes a stock portfolio using Bayesian Optimization and explains the decisions using SHAP (Explainable AI). It finds the best asset allocation to maximize the Sharpe Ratio, ensuring an optimal balance of returns and risk.

## Key Features

1. Bayesian Optimization: Finds the best stock weight distribution.
2. Markowitz’s Modern Portfolio Theory (MPT): Calculates return, risk, and Sharpe ratio.
3. SHAP for Explainability: Explains why certain stocks were given more weight.

## Installation And Uasge

1. Ensure you have Python installed. Then, install the required libraries using:  

```bash
pip install numpy pandas yfinance matplotlib shap bayesian-optimization scipy
```

2. Execute the Python script to optimize the stock weights:

```bash
python main.py
```

## Results 

1. Improved Sharpe Ratio: Higher risk-adjusted returns.
2. Better Risk Management: Diversified stock allocation.
3. Explainable Decisions: SHAP shows why weights changed.

Note : Currently, this project is for education purpose only, not for financial advice.
