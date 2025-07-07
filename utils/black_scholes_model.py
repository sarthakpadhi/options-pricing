import pandas as pd
import numpy as np
from scipy.stats import norm
from typing import Optional
import investpy


class EuropeanOptionPricing:
    def __init__(
        self,
        stock_price: float,
        strike_price: float,
        time_to_expiration: float,
        volatility: float,
        risk_free_rate: Optional[float] = None,
    ):
        self.stock_price = stock_price
        self.strike_price = strike_price
        self.time_to_expiration = time_to_expiration
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        
        if self.risk_free_rate is None:
            self._get_risk_free_rate()

        self.bsm_assets()

    def _get_risk_free_rate(self):
        # Get India government bond data
        bonds = investpy.bonds.get_bond_recent_data(bond="India 10Y")
        self.risk_free_rate = (
            bonds.iloc[-1].Close / 100
        )  # Convert percentage to decimal

    def bsm_assets(self):
        d1 = (
            np.log(self.stock_price / self.strike_price)
            + (self.risk_free_rate + 0.5 * self.volatility**2) * self.time_to_expiration
        ) / (self.volatility * np.sqrt(self.time_to_expiration))
        d2 = d1 - self.volatility * np.sqrt(self.time_to_expiration)
        self.N_d1 = norm.cdf(d1)
        self.N_d2 = norm.cdf(d2)

    def calculate_call_option_price(self) -> float:
        call_option = (
            self.stock_price * self.N_d1
            - self.strike_price
            * np.exp(-self.risk_free_rate * self.time_to_expiration)
            * self.N_d2
        )
        return call_option

    def calculate_put_option_price(self) -> float:
        """using put call parity"""
        call_price = self.calculate_call_option_price()
        return (
            call_price
            + self.strike_price * np.exp(-self.risk_free_rate * self.time_to_expiration)
            - self.stock_price
        )
