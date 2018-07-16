# Modelling CAPM and calculating alpha and beta from historical values.
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader as pdr
from pandas_datareader import data as web
from datetime import date
import numpy as np
import matplotlib.pyplot as plt


# Fetch data for Apple stock prices and for S&P-500 Index from Yahoo Finance:
def capm(start_date, end_date, ticker1, ticker2):
        print("Fetching Data...")
        df = web.DataReader(ticker1, 'iex', start_date, end_date)
        dfb = web.DataReader(ticker2, 'iex', start_date, end_date)
        print("Data Fetch Complete.")

        # create a time-series
        dfsm = pd.DataFrame({'s_close': df['close'], 'b_close': dfb['close']}, index=df.index)

        # compute returns
        dfsm[['s_returns', 'b_returns']] = dfsm[['s_close', 'b_close']]/dfsm[['s_close', 'b_close']].shift(1) - 1
        dfsm = dfsm.dropna()

        covmat = np.cov(dfsm["s_returns"], dfsm["b_returns"])

        # calculate measures now
        beta = covmat[0, 1]/covmat[1, 1]
        alpha = np.mean(dfsm["s_returns"])-beta*np.mean(dfsm["b_returns"])

        # r_squared     = 1.0 - SS_res/SS_tot
        y = beta * dfsm["b_returns"] + alpha
        SS_res = np.sum(np.power(y - dfsm["s_returns"], 2))
        SS_tot = covmat[0, 0]*(len(dfsm) - 1)  # SS_tot is sample_variance*(n-1)

        r_squared = 1.0 - SS_res/SS_tot
        # Volatility for the full time and 1-year momentum
        volatility = np.sqrt(covmat[0, 0])
        momentum = np.prod(1+dfsm["s_returns"].tail(12).values) - 1.0

        # annualize the numbers
        prd = 12.0  # used monthly returns; 12 periods to annualize
        alpha = alpha*prd
        volatility = volatility*np.sqrt(prd)

        print("Beta, alpha, r_squared, volatility, momentum:")
        print(beta, alpha, r_squared, volatility, momentum)

        #     %pylab
        fig, ax = plt.subplots(1, figsize=(8, 6), dpi=80)
        ax.scatter(dfsm["b_returns"], dfsm['s_returns'], label="Data points", s=8)
        beta, alpha = np.polyfit(dfsm["b_returns"], dfsm['s_returns'], deg=1)
        graphX = np.linspace(-0.025, 0.025)
        plt.xlim(-0.025, 0.025)
        ax.plot(graphX, beta*graphX + alpha, color='red', label="CAPM line")

        plt.title('Capital Asset Pricing Model, finding alphas and betas')
        plt.xlabel('Market return $R_m$', fontsize=12)
        plt.ylabel('Stock return $R_i$', fontsize=12)
        plt.text(0.05, 0.05, r'$R_i = \beta * R_m + \alpha$', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()


capm('2017-01-01', '2017-12-31', 'AAPL', 'SPY')