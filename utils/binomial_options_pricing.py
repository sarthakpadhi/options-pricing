from .european_options_pricing import EuropeanOptionsPricing
from typing import Optional
import numpy as np


class BinomialOptionsPricing(EuropeanOptionsPricing):
    def __init__(
        self,
        S0: float,
        K: float,
        vol: float,
        r: float,
        T: float,
        N: int,
        div: Optional[float] = 0,
        u: Optional[float] = None,
    ):
        super().__init__(S0, K, vol, r, T, div)
        self.N = N
        self.del_t = T / N
        self.disc = np.exp(-r * self.del_t)
        self.u = u or np.exp(vol * np.sqrt(self.del_t))
        self.d = 1 / self.u
        self.a = 1 / self.disc
        self.p = (self.a - self.d) / (self.u - self.d)

    def calculate_call_option_price(self, *args, **kwargs) -> float:
        # Implement the binomial pricing logic here
        S = (
            self.S0
            * self.u ** np.arange(0, self.N + 1)
            * self.d ** np.arange(self.N, -1, -1)
        )
        C = np.maximum(S - self.K, 0)
        intermediate_C = []
        for time_step in range(self.N, 0, -1):
            intermediate_C.append(C)
            C = self.disc * (self.p * C[1:] + (1 - self.p) * C[:-1])
        return C[0]
