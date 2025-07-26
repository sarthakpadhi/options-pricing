from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class EuropeanOptionsPricing(ABC):
    def __init__(
        self,
        S0: float,
        K: float,
        vol: float,
        r: float,
        T: float,
        div: Optional[float] = 0,
    ):
        self.S0 = S0
        self.K = K
        self.vol = vol
        self.r = r
        self.T = T
        self.div = div

    @abstractmethod
    def calculate_call_option_price(self, *args, **kwargs) -> float:
        pass

    def calculate_put_option_price(self, *args, **kwargs) -> float:
        """using put call parity"""
        call_price = self.calculate_call_option_price(args, kwargs)
        return (
            call_price
            - self.S0 * np.exp(-self.div * self.T)
            + self.K * np.exp(-self.r * self.T)
        )
