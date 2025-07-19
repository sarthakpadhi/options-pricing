import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from typing import Optional, Tuple, Annotated
import investpy
import plotly.graph_objects as go
import plotly.express as px
from utils.black_scholes_model import EuropeanOptionPricing
from utils.mc_simulations import MCSimulation


def create_payoff_diagram(option_type, strike_price, option_price, spot_range):
    """Create payoff diagram for options"""
    if option_type == "Call":
        payoff = np.maximum(spot_range - strike_price, 0) - option_price
    else:
        payoff = np.maximum(strike_price - spot_range, 0) - option_price

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=spot_range,
            y=payoff,
            mode="lines",
            name=f"{option_type} Option P&L",
            line=dict(width=3),
        )
    )

    fig.add_hline(
        y=0, line_dash="dash", line_color="gray", annotation_text="Break-even"
    )
    fig.add_vline(
        x=strike_price,
        line_dash="dash",
        line_color="red",
        annotation_text="Strike Price",
    )

    fig.update_layout(
        title=f"{option_type} Option Payoff Diagram",
        xaxis_title="Spot Price",
        yaxis_title="Profit/Loss",
        template="plotly_white",
        height=400,
    )

    return fig


def main():
    try:
        st.set_page_config(
            page_title="European Option Pricing Calculator",
            page_icon="📈",
            layout="wide",
        )
    except Exception:
        # Page config already set or running outside Streamlit
        pass

    st.title("📈 European Option Pricing Calculator")
    st.markdown(
        "Calculate Call and Put option prices using Black-Scholes-Merton model or Monte Carlo simulation"
    )

    st.sidebar.header("Input Parameters")

    # Pricing method selection
    pricing_method = st.sidebar.selectbox(
        "Select Pricing Method:",
        ["Black-Scholes Model", "Monte Carlo Simulation", "Compare Both Methods"],
    )

    # Input method selection
    input_method = st.sidebar.radio(
        "Select Input Method:", ["Manual Input", "Fetch from Stock Ticker"]
    )

    if input_method == "Manual Input":
        stock_price = st.sidebar.number_input(
            "Current Stock Price (₹)", min_value=0.01, value=100.0, step=0.01
        )
        stock_ticker = None
        risk_free_rate = (
            st.sidebar.number_input(
                "Risk-Free Rate (%)",
                min_value=0.0,
                max_value=100.0,
                value=6.0,
                step=0.1,
            )
            / 100
        )
    else:
        stock_ticker = st.sidebar.text_input(
            "Stock Ticker (e.g., RELIANCE)", value="RELIANCE"
        )
        stock_price = None
        risk_free_rate = None

        if stock_ticker:
            try:
                with st.spinner("Fetching stock data..."):
                    current_price = EuropeanOptionPricing.get_most_recent_stock_price(
                        stock_ticker
                    )
                    st.sidebar.success(f"Current Price: ₹{current_price:.2f}")
            except Exception as e:
                st.sidebar.error(f"Error fetching data: {str(e)}")

    # Other parameters
    strike_price = st.sidebar.number_input(
        "Strike Price (₹)", min_value=0.01, value=100.0, step=0.01
    )

    time_to_expiration = st.sidebar.number_input(
        "Time to Expiration (Years)", min_value=0.001, value=0.25, step=0.001
    )

    volatility = (
        st.sidebar.number_input(
            "Volatility (%)", min_value=0.1, max_value=200.0, value=20.0, step=0.1
        )
        / 100
    )

    # Monte Carlo specific parameters
    if pricing_method in ["Monte Carlo Simulation", "Compare Both Methods"]:
        st.sidebar.subheader("Monte Carlo Parameters")
        num_simulations = st.sidebar.number_input(
            "Number of Simulations",
            min_value=1000,
            max_value=100000,
            value=10000,
            step=1000,
        )

        time_steps = st.sidebar.number_input(
            "Time Steps", min_value=100, max_value=2000, value=1000, step=100
        )

    # Calculate button
    if st.sidebar.button("Calculate Options", type="primary"):
        try:
            with st.spinner("Calculating option prices..."):

                # Initialize parameters for calculations
                if input_method == "Fetch from Stock Ticker":
                    # Get stock price and risk-free rate
                    option_calculator = EuropeanOptionPricing(
                        strike_price=strike_price,
                        time_to_expiration=time_to_expiration,
                        volatility=volatility,
                        stock_price=stock_price,
                        stock_ticker=stock_ticker,
                        risk_free_rate=risk_free_rate,
                    )
                    actual_stock_price = option_calculator.stock_price
                    actual_risk_free_rate = option_calculator.risk_free_rate
                else:
                    actual_stock_price = stock_price
                    actual_risk_free_rate = risk_free_rate

                if pricing_method == "Black-Scholes Model":
                    # Black-Scholes calculation
                    option_calculator = EuropeanOptionPricing(
                        strike_price=strike_price,
                        time_to_expiration=time_to_expiration,
                        volatility=volatility,
                        stock_price=stock_price,
                        stock_ticker=stock_ticker,
                        risk_free_rate=risk_free_rate,
                    )

                    call_price = option_calculator.calculate_call_option_price()
                    put_price = option_calculator.calculate_put_option_price()

                    # Display results
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Current Stock Price",
                            f"₹{option_calculator.stock_price:.2f}",
                        )
                        st.metric("Strike Price", f"₹{strike_price:.2f}")

                    with col2:
                        st.metric("Call Option Price (BS)", f"₹{call_price:.2f}")
                        st.metric("Put Option Price (BS)", f"₹{put_price:.2f}")

                    with col3:
                        st.metric(
                            "Time to Expiration", f"{time_to_expiration:.3f} years"
                        )
                        st.metric(
                            "Risk-Free Rate",
                            f"{option_calculator.risk_free_rate*100:.2f}%",
                        )

                elif pricing_method == "Monte Carlo Simulation":
                    # Monte Carlo calculation
                    mc_calculator = MCSimulation(
                        stock_price=actual_stock_price,
                        strike_price=strike_price,
                        time_to_expiration=time_to_expiration,
                        volatility=volatility,
                        risk_free_rate=actual_risk_free_rate,
                        time_steps=time_steps,
                    )

                    call_price = mc_calculator.calculate_call_option_price(
                        num_simulations
                    )
                    put_price = mc_calculator.calculate_put_option_price(
                        num_simulations
                    )
                    standard_error = mc_calculator.get_standard_error()

                    # Display results
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Current Stock Price", f"₹{actual_stock_price:.2f}")
                        st.metric("Strike Price", f"₹{strike_price:.2f}")

                    with col2:
                        st.metric("Call Option Price (MC)", f"₹{call_price:.2f}")
                        st.metric("Put Option Price (MC)", f"₹{put_price:.2f}")

                    with col3:
                        st.metric("Standard Error", f"₹{standard_error:.4f}")
                        st.metric("Number of Simulations", f"{num_simulations:,}")

                    # Monte Carlo specific visualizations
                    st.subheader("Monte Carlo Simulation Results")

                    col1, col2 = st.columns(2)
                    with col1:
                        hist_fig = mc_calculator.create_simulation_histogram(
                            num_simulations
                        )
                        st.plotly_chart(hist_fig, use_container_width=True)

                    with col2:
                        paths_fig = mc_calculator.create_price_paths(
                            min(100, num_simulations)
                        )
                        st.plotly_chart(paths_fig, use_container_width=True)

                else:  # Compare Both Methods
                    # Black-Scholes calculation
                    option_calculator = EuropeanOptionPricing(
                        strike_price=strike_price,
                        time_to_expiration=time_to_expiration,
                        volatility=volatility,
                        stock_price=stock_price,
                        stock_ticker=stock_ticker,
                        risk_free_rate=risk_free_rate,
                    )

                    bs_call_price = option_calculator.calculate_call_option_price()
                    bs_put_price = option_calculator.calculate_put_option_price()

                    # Monte Carlo calculation
                    mc_calculator = MCSimulation(
                        stock_price=option_calculator.stock_price,
                        strike_price=strike_price,
                        time_to_expiration=time_to_expiration,
                        volatility=volatility,
                        risk_free_rate=option_calculator.risk_free_rate,
                        time_steps=time_steps,
                    )

                    mc_call_price = mc_calculator.calculate_call_option_price(
                        num_simulations
                    )
                    mc_put_price = mc_calculator.calculate_put_option_price(
                        num_simulations
                    )
                    standard_error = mc_calculator.get_standard_error()

                    # Display comparison
                    st.subheader("Method Comparison")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            "Stock Price", f"₹{option_calculator.stock_price:.2f}"
                        )
                        st.metric("Strike Price", f"₹{strike_price:.2f}")

                    with col2:
                        st.metric("Call Price (BS)", f"₹{bs_call_price:.2f}")
                        st.metric("Put Price (BS)", f"₹{bs_put_price:.2f}")

                    with col3:
                        st.metric("Call Price (MC)", f"₹{mc_call_price:.2f}")
                        st.metric("Put Price (MC)", f"₹{mc_put_price:.2f}")

                    with col4:
                        call_diff = abs(bs_call_price - mc_call_price)
                        put_diff = abs(bs_put_price - mc_put_price)
                        st.metric("Call Price Difference", f"₹{call_diff:.4f}")
                        st.metric("Put Price Difference", f"₹{put_diff:.4f}")

                    # Comparison table
                    comparison_data = {
                        "Method": [
                            "Black-Scholes",
                            "Monte Carlo",
                            "Absolute Difference",
                        ],
                        "Call Price (₹)": [
                            f"{bs_call_price:.4f}",
                            f"{mc_call_price:.4f}",
                            f"{call_diff:.4f}",
                        ],
                        "Put Price (₹)": [
                            f"{bs_put_price:.4f}",
                            f"{mc_put_price:.4f}",
                            f"{put_diff:.4f}",
                        ],
                        "Standard Error": ["-", f"{standard_error:.4f}", "-"],
                    }

                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)

                # Payoff diagrams (common for all methods)
                st.subheader("Payoff Diagrams")

                # Use the appropriate call/put prices for payoff diagrams
                if pricing_method == "Black-Scholes Model":
                    payoff_call_price, payoff_put_price = call_price, put_price
                    payoff_stock_price = option_calculator.stock_price
                elif pricing_method == "Monte Carlo Simulation":
                    payoff_call_price, payoff_put_price = call_price, put_price
                    payoff_stock_price = actual_stock_price
                else:  # Compare both
                    payoff_call_price, payoff_put_price = bs_call_price, bs_put_price
                    payoff_stock_price = option_calculator.stock_price

                # Create spot price range for payoff diagram
                spot_min = payoff_stock_price * 0.7
                spot_max = payoff_stock_price * 1.3
                spot_range = np.linspace(spot_min, spot_max, 100)

                col1, col2 = st.columns(2)

                with col1:
                    call_fig = create_payoff_diagram(
                        "Call", strike_price, payoff_call_price, spot_range
                    )
                    st.plotly_chart(call_fig, use_container_width=True)

                with col2:
                    put_fig = create_payoff_diagram(
                        "Put", strike_price, payoff_put_price, spot_range
                    )
                    st.plotly_chart(put_fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error calculating options: {str(e)}")

    # Information section
    with st.expander("ℹ️ About the Pricing Methods"):
        st.markdown(
            """
        **Black-Scholes-Merton Model:**
        - Analytical solution for European option pricing
        - Assumes constant volatility and risk-free rate
        - No dividends during option's life
        - European-style exercise (only at expiration)
        
        **Monte Carlo Simulation:**
        - Numerical method using random sampling
        - Simulates many possible price paths for the underlying asset
        - More flexible for complex payoffs and path-dependent options
        - Provides standard error estimates for the calculated prices
        - Results converge to Black-Scholes prices with sufficient simulations
        
        **Key Parameters:**
        - **Number of Simulations**: More simulations = higher accuracy but longer computation time
        - **Time Steps**: Discretization of the time to expiration (doesn't affect final price due to geometric Brownian motion assumption)
        - **Standard Error**: Measure of precision in Monte Carlo estimation
        """
        )


if __name__ == "__main__":
    main()
