import pandas as pd
import numpy as np
from scipy.stats import norm
from typing import Optional
import investpy
import yfinance as yf
from .european_options_pricing import EuropeanOptionsPricing


class BSMOptionPricing(EuropeanOptionsPricing):
    def __init__(
        self,
        K: float,
        T: float,
        vol: float,
        S0: Optional[float] = None,
        stock_ticker: Optional[str] = None,
        r: Optional[float] = None,
    ):
        super().__init__(S0, K, vol, r, T)
        self.stock_ticker = stock_ticker

        if self.stock_ticker is not None:
            # Fetch stock price using investpy if stock_ticker is provided
            self.S0 = BSMOptionPricing.get_most_recent_S0(self.stock_ticker)
        if self.r is None:
            self._get_r()

        self.bsm_assets()

    @staticmethod
    def get_most_recent_S0(stock_ticker: str) -> float:
        """
        Fetches the most recent closing stock price for a given stock ticker.
        :param stock_ticker: The ticker symbol of the stock (e.g., 'RELIANCE.NS').
        :return: The most recent closing stock price.
        """
        if not stock_ticker.endswith(".NS"):
            stock_ticker += ".NS"
        stock = yf.Ticker(stock_ticker)
        hist = stock.history(period="1d")
        if hist.empty:
            raise ValueError(f"No data found for ticker: {stock_ticker}")
        return hist["Close"].iloc[-1]

    def _get_r(self):
        # Get India government bond data
        bonds = investpy.bonds.get_bond_recent_data(bond="India 10Y")
        self.r = bonds.iloc[-1].Close / 100  # Convert percentage to decimal

    def bsm_assets(self):
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.vol**2) * self.T) / (
            self.vol * np.sqrt(self.T)
        )
        d2 = d1 - self.vol * np.sqrt(self.T)
        self.N_d1 = norm.cdf(d1)
        self.N_d2 = norm.cdf(d2)

    def calculate_call_option_price(self, *args, **kwargs) -> float:
        call_option = (
            self.S0 * self.N_d1 - self.K * np.exp(-self.r * self.T) * self.N_d2
        )
        return call_option
