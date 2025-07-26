import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from typing import Optional, Tuple, Annotated
import investpy
import plotly.graph_objects as go
import plotly.express as px
from utils.black_scholes_model import BSMOptionPricing
from utils.mc_simulations import MCSimulation
from utils.binomial_options_pricing import BinomialOptionsPricing


def create_payoff_diagram(option_type, K, option_price, spot_range):
    """Create payoff diagram for options"""
    if option_type == "Call":
        payoff = np.maximum(spot_range - K, 0) - option_price
    else:
        payoff = np.maximum(K - spot_range, 0) - option_price

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
        x=K,
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
        "Calculate Call and Put option prices using Black-Scholes-Merton model, Monte Carlo simulation, or Binomial Tree model"
    )

    st.sidebar.header("Input Parameters")

    # Pricing method selection
    pricing_method = st.sidebar.selectbox(
        "Select Pricing Method:",
        [
            "Black-Scholes Model", 
            "Monte Carlo Simulation", 
            "Binomial Tree Model",
            "Compare All Methods"
        ],
    )

    # Input method selection
    input_method = st.sidebar.radio(
        "Select Input Method:", ["Manual Input", "Fetch from Stock Ticker"]
    )

    if input_method == "Manual Input":
        S0 = st.sidebar.number_input(
            "Current Stock Price (₹)", min_value=0.01, value=100.0, step=0.01
        )
        stock_ticker = None
        r = (
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
        S0 = None
        r = None

        if stock_ticker:
            try:
                with st.spinner("Fetching stock data..."):
                    current_price = BSMOptionPricing.get_most_recent_S0(
                        stock_ticker
                    )
                    st.sidebar.success(f"Current Price: ₹{current_price:.2f}")
            except Exception as e:
                st.sidebar.error(f"Error fetching data: {str(e)}")

    # Other parameters
    K = st.sidebar.number_input(
        "Strike Price (₹)", min_value=0.01, value=100.0, step=0.01
    )

    T = st.sidebar.number_input(
        "Time to Expiration (Years)", min_value=0.001, value=0.25, step=0.001
    )

    vol = (
        st.sidebar.number_input(
            "Volatility (%)", min_value=0.1, max_value=200.0, value=20.0, step=0.1
        )
        / 100
    )

    # Monte Carlo specific parameters
    if pricing_method in ["Monte Carlo Simulation", "Compare All Methods"]:
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

    # Binomial Tree specific parameters
    if pricing_method in ["Binomial Tree Model", "Compare All Methods"]:
        st.sidebar.subheader("Binomial Tree Parameters")
        N_steps = st.sidebar.number_input(
            "Number of Time Steps (N)",
            min_value=5,
            max_value=500,
            value=50,
            step=5,
        )

    # Calculate button
    if st.sidebar.button("Calculate Options", type="primary"):
        try:
            with st.spinner("Calculating option prices..."):

                # Initialize parameters for calculations
                if input_method == "Fetch from Stock Ticker":
                    # Get stock price and risk-free rate
                    option_calculator = BSMOptionPricing(
                        K=K,
                        T=T,
                        vol=vol,
                        S0=S0,
                        stock_ticker=stock_ticker,
                        r=r,
                    )
                    actual_S0 = option_calculator.S0
                    actual_r = option_calculator.r
                else:
                    actual_S0 = S0
                    actual_r = r

                if pricing_method == "Black-Scholes Model":
                    # Black-Scholes calculation
                    option_calculator = BSMOptionPricing(
                        K=K,
                        T=T,
                        vol=vol,
                        S0=S0,
                        stock_ticker=stock_ticker,
                        r=r,
                    )

                    call_price = option_calculator.calculate_call_option_price()
                    put_price = option_calculator.calculate_put_option_price()

                    # Display results
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Current Stock Price",
                            f"₹{option_calculator.S0:.2f}",
                        )
                        st.metric("Strike Price", f"₹{K:.2f}")

                    with col2:
                        st.metric("Call Option Price (BS)", f"₹{call_price:.2f}")
                        st.metric("Put Option Price (BS)", f"₹{put_price:.2f}")

                    with col3:
                        st.metric(
                            "Time to Expiration", f"{T:.3f} years"
                        )
                        st.metric(
                            "Risk-Free Rate",
                            f"{option_calculator.r*100:.2f}%",
                        )

                elif pricing_method == "Monte Carlo Simulation":
                    # Monte Carlo calculation
                    mc_calculator = MCSimulation(
                        S0=actual_S0,
                        K=K,
                        T=T,
                        vol=vol,
                        r=actual_r,
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
                        st.metric("Current Stock Price", f"₹{actual_S0:.2f}")
                        st.metric("Strike Price", f"₹{K:.2f}")

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

                elif pricing_method == "Binomial Tree Model":
                    # Binomial Tree calculation
                    binomial_calculator = BinomialOptionsPricing(
                        S0=actual_S0,
                        K=K,
                        vol=vol,
                        r=actual_r,
                        T=T,
                        N=N_steps,
                    )

                    call_price = binomial_calculator.calculate_call_option_price()
                    put_price = binomial_calculator.calculate_put_option_price()
                    tree_params = binomial_calculator.get_tree_parameters()

                    # Display results
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Current Stock Price", f"₹{actual_S0:.2f}")
                        st.metric("Strike Price", f"₹{K:.2f}")

                    with col2:
                        st.metric("Call Option Price (Binomial)", f"₹{call_price:.2f}")
                        st.metric("Put Option Price (Binomial)", f"₹{put_price:.2f}")

                    with col3:
                        st.metric("Number of Steps", f"{N_steps}")
                        st.metric("Up Factor (u)", f"{tree_params['Up Factor (u)']:.4f}")

                    # Binomial Tree specific visualizations
                    st.subheader("Binomial Tree Analysis")

                    # Tree parameters table
                    st.subheader("Tree Parameters")
                    params_df = pd.DataFrame(list(tree_params.items()), 
                                           columns=["Parameter", "Value"])
                    params_df["Value"] = params_df["Value"].apply(
                        lambda x: f"{x:.6f}" if isinstance(x, float) else str(x)
                    )
                    st.dataframe(params_df, use_container_width=True)

                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Call Option Tree")
                        call_tree_fig = binomial_calculator.create_binomial_tree_visualization("call")
                        st.plotly_chart(call_tree_fig, use_container_width=True)

                    with col2:
                        st.subheader("Put Option Tree")
                        put_tree_fig = binomial_calculator.create_binomial_tree_visualization("put")
                        st.plotly_chart(put_tree_fig, use_container_width=True)

                    # Convergence plot
                    st.subheader("Convergence Analysis")
                    convergence_fig = binomial_calculator.create_convergence_plot(
                        max_steps=min(100, N_steps * 2)
                    )
                    st.plotly_chart(convergence_fig, use_container_width=True)

                else:  # Compare All Methods
                    # Black-Scholes calculation
                    option_calculator = BSMOptionPricing(
                        K=K,
                        T=T,
                        vol=vol,
                        S0=S0,
                        stock_ticker=stock_ticker,
                        r=r,
                    )

                    bs_call_price = option_calculator.calculate_call_option_price()
                    bs_put_price = option_calculator.calculate_put_option_price()

                    # Monte Carlo calculation
                    mc_calculator = MCSimulation(
                        S0=option_calculator.S0,
                        K=K,
                        T=T,
                        vol=vol,
                        r=option_calculator.r,
                        time_steps=time_steps,
                    )

                    mc_call_price = mc_calculator.calculate_call_option_price(
                        num_simulations
                    )
                    mc_put_price = mc_calculator.calculate_put_option_price(
                        num_simulations
                    )
                    standard_error = mc_calculator.get_standard_error()

                    # Binomial Tree calculation
                    binomial_calculator = BinomialOptionsPricing(
                        S0=option_calculator.S0,
                        K=K,
                        vol=vol,
                        r=option_calculator.r,
                        T=T,
                        N=N_steps,
                    )

                    bin_call_price = binomial_calculator.calculate_call_option_price()
                    bin_put_price = binomial_calculator.calculate_put_option_price()

                    # Display comparison
                    st.subheader("Method Comparison")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            "Stock Price", f"₹{option_calculator.S0:.2f}"
                        )
                        st.metric("Strike Price", f"₹{K:.2f}")

                    with col2:
                        st.metric("Call Price (BSM)", f"₹{bs_call_price:.2f}")
                        st.metric("Put Price (BSM)", f"₹{bs_put_price:.2f}")

                    with col3:
                        st.metric("Call Price (MC)", f"₹{mc_call_price:.2f}")
                        st.metric("Put Price (MC)", f"₹{mc_put_price:.2f}")

                    with col4:
                        st.metric("Call Price (Binomial)", f"₹{bin_call_price:.2f}")
                        st.metric("Put Price (Binomial)", f"₹{bin_put_price:.2f}")

                    # Detailed comparison table
                    st.subheader("Detailed Price Comparison")
                    comparison_data = {
                        "Method": [
                            "Black-Scholes",
                            "Monte Carlo",
                            "Binomial Tree",
                            "BS vs MC Diff",
                            "BS vs Binomial Diff",
                            "MC vs Binomial Diff"
                        ],
                        "Call Price (₹)": [
                            f"{bs_call_price:.4f}",
                            f"{mc_call_price:.4f}",
                            f"{bin_call_price:.4f}",
                            f"{abs(bs_call_price - mc_call_price):.4f}",
                            f"{abs(bs_call_price - bin_call_price):.4f}",
                            f"{abs(mc_call_price - bin_call_price):.4f}"
                        ],
                        "Put Price (₹)": [
                            f"{bs_put_price:.4f}",
                            f"{mc_put_price:.4f}",
                            f"{bin_put_price:.4f}",
                            f"{abs(bs_put_price - mc_put_price):.4f}",
                            f"{abs(bs_put_price - bin_put_price):.4f}",
                            f"{abs(mc_put_price - bin_put_price):.4f}"
                        ],
                        "Standard Error": [
                            "-", 
                            f"{standard_error:.4f}", 
                            "-",
                            "-",
                            "-",
                            "-"
                        ],
                    }

                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)

                    # Additional visualizations for comparison
                    st.subheader("Method Comparison Visualizations")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Price comparison bar chart
                        methods = ['Black-Scholes', 'Monte Carlo', 'Binomial Tree']
                        call_prices = [bs_call_price, mc_call_price, bin_call_price]
                        put_prices = [bs_put_price, mc_put_price, bin_put_price]
                        
                        comparison_fig = go.Figure(data=[
                            go.Bar(name='Call Options', x=methods, y=call_prices),
                            go.Bar(name='Put Options', x=methods, y=put_prices)
                        ])
                        comparison_fig.update_layout(
                            title="Option Prices by Method",
                            yaxis_title="Price (₹)",
                            barmode='group',
                            template="plotly_white",
                            height=400
                        )
                        st.plotly_chart(comparison_fig, use_container_width=True)
                    
                    with col2:
                        # Convergence plot for binomial
                        convergence_fig = binomial_calculator.create_convergence_plot(
                            max_steps=min(100, N_steps * 2)
                        )
                        st.plotly_chart(convergence_fig, use_container_width=True)

                # Payoff diagrams (common for all methods)
                st.subheader("Payoff Diagrams")

                # Use the appropriate call/put prices for payoff diagrams
                if pricing_method == "Black-Scholes Model":
                    payoff_call_price, payoff_put_price = call_price, put_price
                    payoff_S0 = option_calculator.S0
                elif pricing_method == "Monte Carlo Simulation":
                    payoff_call_price, payoff_put_price = call_price, put_price
                    payoff_S0 = actual_S0
                elif pricing_method == "Binomial Tree Model":
                    payoff_call_price, payoff_put_price = call_price, put_price
                    payoff_S0 = actual_S0
                else:  # Compare all methods
                    payoff_call_price, payoff_put_price = bs_call_price, bs_put_price
                    payoff_S0 = option_calculator.S0

                # Create spot price range for payoff diagram
                spot_min = payoff_S0 * 0.7
                spot_max = payoff_S0 * 1.3
                spot_range = np.linspace(spot_min, spot_max, 100)

                col1, col2 = st.columns(2)

                with col1:
                    call_fig = create_payoff_diagram(
                        "Call", K, payoff_call_price, spot_range
                    )
                    st.plotly_chart(call_fig, use_container_width=True)

                with col2:
                    put_fig = create_payoff_diagram(
                        "Put", K, payoff_put_price, spot_range
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
        
        **Binomial Tree Model:**
        - Discrete-time model that approximates the continuous Black-Scholes model
        - Uses a recombining tree structure with up and down movements
        - Converges to Black-Scholes prices as the number of time steps increases
        - More intuitive and easier to understand than Black-Scholes
        - Can handle early exercise features (American options) with modifications
        - Allows for easy visualization of the option pricing process
        
        **Key Parameters:**
        - **Number of Simulations (MC)**: More simulations = higher accuracy but longer computation time
        - **Time Steps (MC)**: Discretization of the time to expiration
        - **Number of Steps (Binomial)**: More steps = better approximation to Black-Scholes
        - **Standard Error (MC)**: Measure of precision in Monte Carlo estimation
        - **Up/Down Factors (Binomial)**: Determine the magnitude of price movements in each step
        - **Risk-Neutral Probability (Binomial)**: Probability of upward movement in risk-neutral world
        
        **Convergence:**
        - Binomial trees converge to Black-Scholes prices as N → ∞
        - Monte Carlo simulations converge to Black-Scholes prices as number of simulations → ∞
        - All three methods should give similar results for European options under the same assumptions
        """
        )

    # Additional analysis section
    with st.expander("📊 Advanced Analysis Tips"):
        st.markdown(
            """
        **Choosing the Right Method:**
        
        - **Black-Scholes**: Best for quick analytical results, standard European options
        - **Monte Carlo**: Best for complex payoffs, path-dependent options, or when you need confidence intervals
        - **Binomial Tree**: Best for understanding the pricing mechanism, American options, or when you want to see the tree structure
        
        **Recommended Parameters:**
        
        - **Binomial Steps**: 50-100 steps for good accuracy, 200+ for high precision
        - **MC Simulations**: 10,000+ for reliable results, 50,000+ for high precision
        - **MC Time Steps**: 1000+ steps for smooth paths
        
        **Interpreting Results:**
        
        - Small differences between methods are normal due to discretization and sampling errors
        - Binomial prices should get closer to Black-Scholes as you increase the number of steps
        - Monte Carlo standard error gives you confidence in the simulation results
        - If methods give very different results, check your input parameters
        
        **Visualization Benefits:**
        
        - **Binomial Tree**: Shows how option value evolves through time and different stock price scenarios
        - **Convergence Plot**: Demonstrates how binomial prices approach the theoretical Black-Scholes price
        - **MC Histograms**: Show the distribution of final stock prices from the simulations
        - **MC Price Paths**: Illustrate the random walk behavior of stock prices
        """
        )


if __name__ == "__main__":
    main()