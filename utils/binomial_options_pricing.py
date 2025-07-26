from .european_options_pricing import EuropeanOptionsPricing
from typing import Optional, List
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


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
        self.intermediate_C = []
        self.intermediate_P = []
        self.stock_prices = [] 

    def calculate_call_option_price(self, *args, **kwargs) -> float:
        S = (
            self.S0
            * self.u ** np.arange(0, self.N + 1)
            * self.d ** np.arange(self.N, -1, -1)
        )
        self.stock_prices = [S.copy()]

        C = np.maximum(S - self.K, 0)
        self.intermediate_C = [C.copy()]

        for time_step in range(self.N, 0, -1):
            S_prev = (
                self.S0
                * self.u ** np.arange(0, time_step)
                * self.d ** np.arange(time_step - 1, -1, -1)
            )
            self.stock_prices.append(S_prev.copy())

            C = self.disc * (self.p * C[1:] + (1 - self.p) * C[:-1])
            self.intermediate_C.append(C.copy())

        self.intermediate_C.reverse()
        self.stock_prices.reverse()

        return self.intermediate_C[0][0]

    def calculate_put_option_price(self, *args, **kwargs) -> float:
        S = (
            self.S0
            * self.u ** np.arange(0, self.N + 1)
            * self.d ** np.arange(self.N, -1, -1)
        )
        self.stock_prices = [S.copy()]

        P = np.maximum(self.K - S, 0)
        self.intermediate_P = [P.copy()]

        for time_step in range(self.N, 0, -1):
            S_prev = (
                self.S0
                * self.u ** np.arange(0, time_step)
                * self.d ** np.arange(time_step - 1, -1, -1)
            )
            self.stock_prices.append(S_prev.copy())

            P = self.disc * (self.p * P[1:] + (1 - self.p) * P[:-1])
            self.intermediate_P.append(P.copy())

        self.intermediate_P.reverse()
        self.stock_prices.reverse()

        return self.intermediate_P[0][0]

    def create_binomial_tree_visualization(
        self, option_type: str = "call"
    ) -> go.Figure:
        """Create a visualization of the binomial tree"""
        if option_type.lower() == "call" and not self.intermediate_C:
            raise ValueError(
                "Call option prices not calculated yet. Run calculate_call_option_price() first."
            )
        elif option_type.lower() == "put" and not self.intermediate_P:
            raise ValueError(
                "Put option prices not calculated yet. Run calculate_put_option_price() first."
            )

        intermediate_values = (
            self.intermediate_C
            if option_type.lower() == "call"
            else self.intermediate_P
        )

        fig = go.Figure()

        for step in range(min(self.N + 1, 10)):  # Limit to 10 steps for readability
            if step < len(intermediate_values):
                values = intermediate_values[step]
                stock_prices = (
                    self.stock_prices[step] if step < len(self.stock_prices) else []
                )

                # Calculate y positions for nodes at this time step
                num_nodes = len(values)
                y_positions = np.linspace(-num_nodes / 2, num_nodes / 2, num_nodes)

                # Add scatter plot for nodes
                hover_text = []
                for i, (value, y_pos) in enumerate(zip(values, y_positions)):
                    stock_price = stock_prices[i] if i < len(stock_prices) else "N/A"
                    hover_text.append(
                        f"Time Step: {step}<br>"
                        f"Stock Price: ₹{stock_price:.2f}<br>"
                        f"{option_type.title()} Value: ₹{value:.4f}"
                    )

                fig.add_trace(
                    go.Scatter(
                        x=[step] * num_nodes,
                        y=y_positions,
                        mode="markers+text",
                        marker=dict(
                            size=12, color=values, colorscale="Viridis", showscale=True
                        ),
                        text=[f"₹{v:.2f}" for v in values],
                        textposition="middle center",
                        textfont=dict(size=8),
                        hovertext=hover_text,
                        hoverinfo="text",
                        name=f"Step {step}",
                        showlegend=False,
                    )
                )

                # Add connecting lines to next step
                if step < min(self.N, 9) and step < len(intermediate_values) - 1:
                    next_values = intermediate_values[step + 1]
                    next_y_positions = np.linspace(
                        -len(next_values) / 2, len(next_values) / 2, len(next_values)
                    )

                    # Connect each node to the next two nodes
                    for i, y_pos in enumerate(y_positions):
                        if i < len(next_y_positions):  # Up movement
                            fig.add_shape(
                                type="line",
                                x0=step,
                                y0=y_pos,
                                x1=step + 1,
                                y1=next_y_positions[i],
                                line=dict(color="lightblue", width=1),
                            )
                        if i + 1 < len(next_y_positions):  # Down movement
                            fig.add_shape(
                                type="line",
                                x0=step,
                                y0=y_pos,
                                x1=step + 1,
                                y1=next_y_positions[i + 1],
                                line=dict(color="lightcoral", width=1),
                            )

        fig.update_layout(
            title=f"Binomial Tree - {option_type.title()} Option Pricing (First 10 Steps)",
            xaxis_title="Time Steps",
            yaxis_title="Node Position",
            template="plotly_white",
            height=600,
            showlegend=False,
            xaxis=dict(tickmode="linear", tick0=0, dtick=1),
        )

        return fig

    def create_convergence_plot(self, max_steps: int = 100) -> go.Figure:
        """Create a plot showing how option prices converge as N increases"""
        call_prices = []
        put_prices = []
        step_counts = range(5, max_steps + 1, 5)

        original_N = self.N

        for n in step_counts:
            # Temporarily change N
            self.N = n
            self.del_t = self.T / n
            self.disc = np.exp(-self.r * self.del_t)
            self.u = np.exp(self.vol * np.sqrt(self.del_t))
            self.d = 1 / self.u
            self.a = 1 / self.disc
            self.p = (self.a - self.d) / (self.u - self.d)

            # Calculate prices
            call_price = self.calculate_call_option_price()
            put_price = self.calculate_put_option_price()

            call_prices.append(call_price)
            put_prices.append(put_price)

        # Restore original N
        self.N = original_N
        self.del_t = self.T / original_N
        self.disc = np.exp(-self.r * self.del_t)
        self.u = np.exp(self.vol * np.sqrt(self.del_t))
        self.d = 1 / self.u
        self.a = 1 / self.disc
        self.p = (self.a - self.d) / (self.u - self.d)

        # Create the plot
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=list(step_counts),
                y=call_prices,
                mode="lines+markers",
                name="Call Option",
                line=dict(color="green", width=2),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=list(step_counts),
                y=put_prices,
                mode="lines+markers",
                name="Put Option",
                line=dict(color="red", width=2),
            )
        )

        fig.update_layout(
            title="Binomial Option Price Convergence",
            xaxis_title="Number of Time Steps (N)",
            yaxis_title="Option Price (₹)",
            template="plotly_white",
            height=400,
            legend=dict(x=0.7, y=0.95),
        )

        return fig

    def get_tree_parameters(self) -> dict:
        """Return the key binomial tree parameters"""
        return {
            "Time Steps (N)": self.N,
            "Time Step Size (Δt)": self.del_t,
            "Up Factor (u)": self.u,
            "Down Factor (d)": self.d,
            "Risk-Neutral Probability (p)": self.p,
            "Discount Factor": self.disc,
        }
