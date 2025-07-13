import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from typing import Optional
import investpy
import plotly.graph_objects as go
import plotly.express as px
from utils.black_scholes_model import EuropeanOptionPricing

def create_payoff_diagram(option_type, strike_price, option_price, spot_range):
    """Create payoff diagram for options"""
    if option_type == "Call":
        payoff = np.maximum(spot_range - strike_price, 0) - option_price
    else:  
        payoff = np.maximum(strike_price - spot_range, 0) - option_price
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=spot_range,
        y=payoff,
        mode='lines',
        name=f'{option_type} Option P&L',
        line=dict(width=3)
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Break-even")
    fig.add_vline(x=strike_price, line_dash="dash", line_color="red", annotation_text="Strike Price")
    
    fig.update_layout(
        title=f'{option_type} Option Payoff Diagram',
        xaxis_title='Spot Price',
        yaxis_title='Profit/Loss',
        template='plotly_white',
        height=400
    )
    
    return fig

def main():
    try:
        st.set_page_config(
            page_title="European Option Pricing Calculator",
            page_icon="📈",
            layout="wide"
        )
    except Exception:
        # Page config already set or running outside Streamlit
        pass
    
    st.title("📈 European Option Pricing Calculator")
    st.markdown("Calculate Call and Put option prices using the Black-Scholes-Merton model")
    
    
    st.sidebar.header("Input Parameters")
    
    # Input method selection
    input_method = st.sidebar.radio(
        "Select Input Method:",
        ["Manual Input", "Fetch from Stock Ticker"]
    )
    
    if input_method == "Manual Input":
        stock_price = st.sidebar.number_input(
            "Current Stock Price (₹)", 
            min_value=0.01, 
            value=100.0, 
            step=0.01
        )
        stock_ticker = None
        risk_free_rate = st.sidebar.number_input(
            "Risk-Free Rate (%)", 
            min_value=0.0, 
            max_value=100.0, 
            value=6.0, 
            step=0.1
        ) / 100
    else:
        stock_ticker = st.sidebar.text_input(
            "Stock Ticker (e.g., RELIANCE)", 
            value="RELIANCE"
        )
        stock_price = None
        risk_free_rate = None
        
        if stock_ticker:
            try:
                with st.spinner("Fetching stock data..."):
                    current_price = EuropeanOptionPricing.get_most_recent_stock_price(stock_ticker)
                    st.sidebar.success(f"Current Price: ₹{current_price:.2f}")
            except Exception as e:
                st.sidebar.error(f"Error fetching data: {str(e)}")
    
    # Other parameters
    strike_price = st.sidebar.number_input(
        "Strike Price (₹)", 
        min_value=0.01, 
        value=100.0, 
        step=0.01
    )
    
    time_to_expiration = st.sidebar.number_input(
        "Time to Expiration (Years)", 
        min_value=0.001, 
        value=0.25, 
        step=0.001
    )
    
    volatility = st.sidebar.number_input(
        "Volatility (%)", 
        min_value=0.1, 
        max_value=200.0, 
        value=20.0, 
        step=0.1
    ) / 100
    
    # Calculate button
    if st.sidebar.button("Calculate Options", type="primary"):
        try:
            with st.spinner("Calculating option prices..."):
                # Create option pricing object
                option_calculator = EuropeanOptionPricing(
                    strike_price=strike_price,
                    time_to_expiration=time_to_expiration,
                    volatility=volatility,
                    stock_price=stock_price,
                    stock_ticker=stock_ticker,
                    risk_free_rate=risk_free_rate
                )
                
                # Calculate option prices
                call_price = option_calculator.calculate_call_option_price()
                put_price = option_calculator.calculate_put_option_price()
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Current Stock Price", f"₹{option_calculator.stock_price:.2f}")
                    st.metric("Strike Price", f"₹{strike_price:.2f}")
                
                with col2:
                    st.metric("Call Option Price", f"₹{call_price:.2f}")
                    st.metric("Put Option Price", f"₹{put_price:.2f}")
                
                with col3:
                    st.metric("Time to Expiration", f"{time_to_expiration:.3f} years")
                    st.metric("Risk-Free Rate", f"{option_calculator.risk_free_rate*100:.2f}%")
        
                # Payoff diagrams
                st.subheader("Payoff Diagrams")
                
                # Create spot price range for payoff diagram
                spot_min = option_calculator.stock_price * 0.7
                spot_max = option_calculator.stock_price * 1.3
                spot_range = np.linspace(spot_min, spot_max, 100)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    call_fig = create_payoff_diagram("Call", strike_price, call_price, spot_range)
                    st.plotly_chart(call_fig, use_container_width=True)
                
                with col2:
                    put_fig = create_payoff_diagram("Put", strike_price, put_price, spot_range)
                    st.plotly_chart(put_fig, use_container_width=True)
                
                # Summary table
                st.subheader("Summary")
                summary_data = {
                    'Parameter': ['Stock Price', 'Strike Price', 'Time to Expiration', 'Volatility', 'Risk-Free Rate', 'Call Price', 'Put Price'],
                    'Value': [
                        f"₹{option_calculator.stock_price:.2f}",
                        f"₹{strike_price:.2f}",
                        f"{time_to_expiration:.3f} years",
                        f"{volatility*100:.2f}%",
                        f"{option_calculator.risk_free_rate*100:.2f}%",
                        f"₹{call_price:.2f}",
                        f"₹{put_price:.2f}"
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error calculating options: {str(e)}")
    
    # Information section
    with st.expander("ℹ️ About the Black-Scholes-Merton Model"):
        st.markdown("""
        The Black-Scholes-Merton model is used to calculate the theoretical price of European options. 
        
        **Key Assumptions:**
        - Constant volatility and risk-free rate during the option's life
        - No dividends during the option's life
        - European-style exercise (only at expiration)
        - Efficient markets with no transaction costs
    
        """)

if __name__ == "__main__":
    main()