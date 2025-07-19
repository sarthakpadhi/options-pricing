import numpy as np
from typing import Tuple, Annotated


class MCSimulation:
    """class for Monte Carlo Simulation"""

    def __init__(
        self,
        stock_price: float,
        strike_price: float,
        time_to_expiration: float,
        volatility: float,
        risk_free_rate: float,
        time_steps: int = 1000,
    ):
        """
        Initialize the Monte Carlo Simulation for European Call Option Pricing.

        Args:
            stock_price (float): Current price of the underlying stock (S₀).
            strike_price (float): Strike price of the option (K).
            time_to_expiration (float): Time to expiration in years (T).
            volatility (float): Annualized volatility of the stock (σ).
            risk_free_rate (float): Annual risk-free interest rate (r).
            time_steps (int, optional): Number of discrete time steps for the simulation. changing this doesn't affect the final simulations
            due to Stock assumed to follow a generalised brownian motion
        """
        self.stock_price = stock_price
        self.strike_price = strike_price
        self.time_to_expiration = time_to_expiration
        self.volatility = volatility
        self.risk_free_rate = risk_free_rate
        self.time_steps = time_steps
        self.dt = time_to_expiration / time_steps

    def _simulate(self, num_sim: int = 10000) -> Tuple[float, float]:
        """the actual simulation code"""
        v = self.risk_free_rate - ((self.volatility) ** 2) * 0.5
        Z = np.random.normal(size=(num_sim, self.time_steps))  ## NxM
        delta_lnSt = v * self.dt + self.volatility * np.sqrt(self.dt) * Z  ## NxM
        self.lnSt = np.log(self.stock_price) + np.cumsum(delta_lnSt, axis=1)  ##NxM
        ST = np.exp(self.lnSt[:, -1])  ##Nx1
        call_option_payoff_at_t = np.maximum(ST - self.strike_price, 0)  ##Nx1
        self.call = call_option_payoff_at_t

        discounted_payoffs = (
            np.exp(-self.risk_free_rate * self.time_to_expiration)
            * call_option_payoff_at_t
        )
        self.estimated_call_price_at_0 = np.sum(discounted_payoffs) / num_sim

        self.SE_for_call_price_at_0 = np.std(discounted_payoffs, ddof=1) / np.sqrt(
            num_sim
        )
        return self.estimated_call_price_at_0, self.SE_for_call_price_at_0

    def calculate_call_option_price(self, num_sims: int = 1000):
        self._simulate(num_sims)
        print(f"with a SE of {self.SE_for_call_price_at_0}")
        return self.estimated_call_price_at_0

    def calculate_put_option_price(self) -> float:
        """using put call parity"""
        call_price = self.calculate_call_option_price()
        print(f"with a SE of {self.SE_for_call_price_at_0}")

        return (
            call_price
            + self.strike_price * np.exp(-self.risk_free_rate * self.time_to_expiration)
            - self.stock_price
        )
