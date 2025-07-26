import numpy as np
from typing import Tuple, Annotated
import plotly.graph_objects as go
from .european_options_pricing import EuropeanOptionsPricing


class MCSimulation(EuropeanOptionsPricing):
    """class for Monte Carlo Simulation"""

    def __init__(
        self,
        S0: float,
        K: float,
        T: float,
        vol: float,
        r: float,
        time_steps: int = 1000,
    ):
        """
        Initialize the Monte Carlo Simulation for European Call Option Pricing.

        Args:
            S0 (float): Current price of the underlying stock (S₀).
            K (float): Strike price of the option (K).
            K (float): Time to expiration in years (T).
            vol (float): Annualized vol of the stock (σ).
            r (float): Annual risk-free interest rate (r).
            time_steps (int, optional): Number of discrete time steps for the simulation. changing this doesn't affect the final simulations
            due to Stock assumed to follow a generalised brownian motion
        """
        super().__init__(S0, K, vol, r, T)
        self.time_steps = time_steps
        self.dt = T / time_steps

    def _simulate(self, num_sim: int = 10000) -> Tuple[float, float]:
        """the actual simulation code"""
        v = self.r - ((self.vol) ** 2) * 0.5
        Z = np.random.normal(size=(num_sim, self.time_steps))  ## NxM
        delta_lnSt = v * self.dt + self.vol * np.sqrt(self.dt) * Z  ## NxM
        self.lnSt = np.log(self.S0) + np.cumsum(delta_lnSt, axis=1)  ##NxM
        ST = np.exp(self.lnSt[:, -1])  ##Nx1
        call_option_payoff_at_t = np.maximum(ST - self.K, 0)  ##Nx1
        self.call = call_option_payoff_at_t

        discounted_payoffs = np.exp(-self.r * self.T) * call_option_payoff_at_t
        self.estimated_call_price_at_0 = np.sum(discounted_payoffs) / num_sim

        self.SE_for_call_price_at_0 = np.std(discounted_payoffs, ddof=1) / np.sqrt(
            num_sim
        )
        return self.estimated_call_price_at_0, self.SE_for_call_price_at_0

    def calculate_call_option_price(
        self, *args, num_sims: int = 10000, **kwargs
    ) -> float:
        self._simulate(num_sims)
        return self.estimated_call_price_at_0

    def get_standard_error(self):
        """Get the standard error from the last simulation"""
        return getattr(self, "SE_for_call_price_at_0", 0)

    def create_simulation_histogram(self, num_sims=10000):
        """Create histogram of simulated final stock prices"""
        # Run simulation to get final stock prices
        self._simulate(num_sims)
        ST = np.exp(self.lnSt[:, -1])

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(x=ST, nbinsx=50, name="Simulated Final Prices", opacity=0.7)
        )

        fig.add_vline(
            x=self.S0,
            line_dash="dash",
            line_color="red",
            annotation_text="Current Price",
        )
        fig.add_vline(
            x=self.K,
            line_dash="dash",
            line_color="green",
            annotation_text="Strike Price",
        )

        fig.update_layout(
            title="Distribution of Simulated Final Stock Prices",
            xaxis_title="Final Stock Price (₹)",
            yaxis_title="Frequency",
            template="plotly_white",
            height=400,
        )

        return fig

    def create_price_paths(self, num_paths=100):
        """Create visualization of simulated price paths"""
        # Generate time array
        time_array = np.linspace(0, self.T, self.time_steps)

        # Get some sample paths
        self._simulate(num_paths)
        sample_paths = np.exp(self.lnSt[:num_paths, :])

        fig = go.Figure()

        # Plot sample paths
        for i in range(min(20, num_paths)):  # Show only first 20 paths for clarity
            fig.add_trace(
                go.Scatter(
                    x=time_array,
                    y=sample_paths[i],
                    mode="lines",
                    line=dict(width=1, color="lightblue"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # Add average path
        avg_path = np.mean(sample_paths, axis=0)
        fig.add_trace(
            go.Scatter(
                x=time_array,
                y=avg_path,
                mode="lines",
                line=dict(width=3, color="red"),
                name="Average Path",
            )
        )

        fig.add_hline(
            y=self.K,
            line_dash="dash",
            line_color="green",
            annotation_text="Strike Price",
        )

        fig.update_layout(
            title=f"Sample Stock Price Paths ({min(20, num_paths)} paths shown)",
            xaxis_title="Time (Years)",
            yaxis_title="Stock Price (₹)",
            template="plotly_white",
            height=400,
        )

        return fig
