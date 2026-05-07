import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Stock Forecast Dashboard",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

html, body, [class*="css"] {
    background-color: #0E1117;
    color: white;
    font-family: 'Arial';
}

.main {
    background-color: #0E1117;
}

.stMetric {
    background-color: #FFFFFF;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #DDD;
    color: black;
}

div[data-testid="stSidebar"] {
    background-color: #161A25;
             min-height: 55px;
        font-size: 18px;
}

h1, h2, h3 {
    color: #00D4FF;
}

.reliability-card {
    padding: 20px;
    border-radius: 12px;
    margin: 10px 0;
}

</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.title("📈 AI Stock Forecast Dashboard")

st.markdown("""
Analyze PSX (Pakistan Stock Exchange) stocks using AI forecasting, technical indicators, 
and interactive financial charts.
""")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Configuration")

# =========================
# PSX TICKER DROPDOWN
# =========================
PSX_TICKERS = {
    "-- Select a stock --": "",
    
    "Custom ticker...": "__custom__",
    # ── Banks ──────────────────────────────────────────────
    "ABL.KA   — Allied Bank Limited":               "ABL.KA",
    "AKBL.KA  — Askari Bank Limited":               "AKBL.KA",
    "BAFL.KA  — Bank Alfalah Limited":              "BAFL.KA",
    "BAHL.KA  — Bank AL Habib Limited":             "BAHL.KA",
    "BOP.KA   — Bank of Punjab":                    "BOP.KA",
    "FABL.KA  — Faysal Bank Limited":               "FABL.KA",
    "HBL.KA   — Habib Bank Limited":                "HBL.KA",
    "HMB.KA   — Habib Metropolitan Bank":           "HMB.KA",
    "MCB.KA   — MCB Bank Limited":                  "MCB.KA",
    "MEBL.KA  — Meezan Bank Limited":               "MEBL.KA",
    "NBP.KA   — National Bank of Pakistan":         "NBP.KA",
    "SCBPL.KA — Standard Chartered Bank Pak":       "SCBPL.KA",
    "UBL.KA   — United Bank Limited":               "UBL.KA",

    # ── Oil & Gas ──────────────────────────────────────────
    "APL.KA   — Attock Petroleum Limited":          "APL.KA",
    "ATRL.KA  — Attock Refinery Limited":           "ATRL.KA",
    "CNERGY.KA — Cnergyico PK Limited":             "CNERGY.KA",
    "MARI.KA  — Mari Energies Limited":             "MARI.KA",
    "OGDC.KA  — Oil & Gas Dev Company":             "OGDC.KA",
    "POL.KA   — Pakistan Oilfields Limited":        "POL.KA",
    "PPL.KA   — Pakistan Petroleum Limited":        "PPL.KA",
    "PSO.KA   — Pakistan State Oil":                "PSO.KA",

    # ── Fertilizer ────────────────────────────────────────
    "EFERT.KA  — Engro Fertilizers Limited":        "EFERT.KA",
    "FATIMA.KA — Fatima Fertilizer Company":        "FATIMA.KA",
    "FFC.KA    — Fauji Fertilizer Company":         "FFC.KA",

    # ── Cement ────────────────────────────────────────────
    "BWCL.KA  — Bestway Cement Limited":            "BWCL.KA",
    "CHCC.KA  — Cherat Cement Company":             "CHCC.KA",
    "DGKC.KA  — D.G. Khan Cement Company":          "DGKC.KA",
    "FCCL.KA  — Fauji Cement Company":              "FCCL.KA",
    "KOHC.KA  — Kohat Cement Company":              "KOHC.KA",
    "LUCK.KA  — Lucky Cement Limited":              "LUCK.KA",
    "MLCF.KA  — Maple Leaf Cement Factory":         "MLCF.KA",
    "PIOC.KA  — Pioneer Cement Limited":            "PIOC.KA",
    "POWER.KA — Power Cement Limited":              "POWER.KA",

    # ── Power ──────────────────────────────────────────────
    "HUBC.KA  — The Hub Power Company":             "HUBC.KA",
    "KAPCO.KA — Kot Addu Power Company":            "KAPCO.KA",
    "KEL.KA   — K-Electric Limited":                "KEL.KA",
    "NPL.KA   — Nishat Power Limited":              "NPL.KA",

    # ── Technology ────────────────────────────────────────
    "AIRLINK.KA — Air Link Communication":          "AIRLINK.KA",
    "SYS.KA    — Systems Limited":                  "SYS.KA",
    "TRG.KA    — TRG Pakistan Limited":             "TRG.KA",
    "PTC.KA    — Pakistan Telecom Company":         "PTC.KA",

    # ── Pharma ────────────────────────────────────────────
    "ABOT.KA   — Abbott Laboratories Pakistan":     "ABOT.KA",
    "AGP.KA    — AGP Limited":                      "AGP.KA",
    "CPHL.KA   — Citi Pharma Ltd":                  "CPHL.KA",
    "GLAXO.KA  — GlaxoSmithKline Pakistan":         "GLAXO.KA",
    "HALEON.KA — Haleon Pakistan Limited":          "HALEON.KA",
    "HINOON.KA — Highnoon Laboratories":            "HINOON.KA",
    "SEARL.KA  — The Searle Company":               "SEARL.KA",

    # ── Automobiles ───────────────────────────────────────
    "ATLH.KA  — Atlas Honda Limited":               "ATLH.KA",
    "GAL.KA   — Ghandhara Automobiles":             "GAL.KA",
    "GHNI.KA  — Ghandhara Industries":              "GHNI.KA",
    "HCAR.KA  — Honda Atlas Cars Pakistan":         "HCAR.KA",
    "INDU.KA  — Indus Motor Company":               "INDU.KA",
    "MTL.KA   — Millat Tractors Limited":           "MTL.KA",
    "SAZEW.KA — Sazgar Engineering Works":          "SAZEW.KA",

    # ── Engro Group ───────────────────────────────────────
    "ENGROH.KA — Engro Holdings Limited":           "ENGROH.KA",
    "EPCL.KA   — Engro Polymer":                    "EPCL.KA",

    # ── Consumer / FMCG ───────────────────────────────────
    "COLG.KA   — Colgate-Palmolive Pakistan":       "COLG.KA",
    "FFL.KA    — Fauji Foods Limited":              "FFL.KA",
    "NATF.KA   — National Foods Limited":           "NATF.KA",
    "NESTLE.KA — Nestle Pakistan Limited":          "NESTLE.KA",
    "UPFL.KA   — Unilever Pakistan Foods":          "UPFL.KA",

    # ── Textiles ──────────────────────────────────────────
    "GADT.KA  — Gadoon Textile Mills":              "GADT.KA",
    "ILP.KA   — Interloop Limited":                 "ILP.KA",
    "KTML.KA  — Kohinoor Textile Mills":            "KTML.KA",
    "NML.KA   — Nishat Mills Limited":              "NML.KA",

    # ── Insurance ─────────────────────────────────────────
    "AICL.KA  — Adamjee Insurance Company":         "AICL.KA",

    # ── Miscellaneous ─────────────────────────────────────
    "AHCL.KA  — Arif Habib Corporation":            "AHCL.KA",
    "DCR.KA   — Dolmen City REIT":                  "DCR.KA",
    "ISL.KA   — International Steels Limited":      "ISL.KA",
    "LCI.KA   — Lucky Core Industries":             "LCI.KA",
    "MUREB.KA — Murree Brewery Company":            "MUREB.KA",
    "PAEL.KA  — Pak Elektron Limited":              "PAEL.KA",
    "PKGS.KA  — Packages Limited":                  "PKGS.KA",
    "PSX.KA   — Pakistan Stock Exchange":           "PSX.KA",
    "SNGP.KA  — Sui Northern Gas Pipelines":        "SNGP.KA",
    "SSGC.KA  — Sui Southern Gas Company":          "SSGC.KA",
    "SHFA.KA  — Shifa International Hospitals":     "SHFA.KA",
    "PAKT.KA  — Pakistan Tobacco Company":          "PAKT.KA",

}

selected_label = st.sidebar.selectbox(
    "Select PSX Stock",
    list(PSX_TICKERS.keys())
)

selected_value = PSX_TICKERS[selected_label]

if selected_value == "__custom__":
    ticker = st.sidebar.text_input(
        "Enter custom ticker (e.g. ENGRO.KA)",
        "HBL.KA"
    ).upper()
elif selected_value == "":
    st.sidebar.warning("Please select a stock to continue.")
    st.stop()
else:
    ticker = selected_value


period = st.sidebar.selectbox(
    "Historical Range",
    ["1y", "2y", "5y", "10y"]
)

prediction_days = st.sidebar.slider(
    "Forecast Days",
    30,
    365,
    90
)

backtest_days = st.sidebar.slider(
    "Backtest Window (days)",
    15,
    90,
    30,
    help="How many recent days to use for evaluating model accuracy"
)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data(symbol, range_period):

    data = yf.download(
        symbol,
        period=range_period,
        auto_adjust=True,
        progress=False
    )

    return data

try:

    df = load_data(ticker, period)

    # Fix MultiIndex issue
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)

    if df.empty:
        st.error("No stock data found.")
        st.stop()

    # =========================
    # COMPANY INFO
    # =========================
    stock = yf.Ticker(ticker)
    info = stock.info

    company_name = info.get("longName", ticker)

    st.subheader(company_name)

    # =========================
    # METRICS
    # =========================
    current_price = float(df['Close'].iloc[-1])
    previous_price = float(df['Close'].iloc[-2])

    change = current_price - previous_price
    percent_change = (change / previous_price) * 100
    volume = int(df['Volume'].iloc[-1])

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Current Price", f"${current_price:.2f}")
    col2.metric("Daily Change", f"{percent_change:.2f}%", delta=f"{change:.2f}")
    col3.metric("Volume", f"{volume:,}")
    col4.metric("Market Cap", f"{info.get('marketCap', 'N/A')}")

    # =========================
    # TABS
    # =========================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Candlestick",
        "Technical Analysis",
        "AI Forecast",
        "Model Evaluation",
        "Company Info"
    ])

    # =========================================================
    # TAB 1 — CANDLESTICK
    # =========================================================
    with tab1:

        fig_candle = go.Figure(data=[go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close']
        )])

        fig_candle.update_layout(
            title=f"{ticker} Candlestick Chart",
            template="plotly_dark",
            height=700,
            hovermode="x unified"
        )

        st.plotly_chart(fig_candle, use_container_width=True)
        st.markdown("""
### What is a Candlestick Chart?

Candlesticks show daily market activity:

- Open price
- Close price
- Highest price
- Lowest price

Green candles usually indicate price increases,
while red candles indicate price decreases.
""")
        # Volume Chart
        fig_volume = go.Figure()

        fig_volume.add_trace(go.Bar(
            x=df['Date'],
            y=df['Volume'],
            name='Volume'
        ))

        fig_volume.update_layout(
            title="Trading Volume",
            template="plotly_dark",
            height=300
        )

        st.plotly_chart(fig_volume, use_container_width=True)

    # =========================================================
    # TAB 2 — TECHNICAL ANALYSIS
    # =========================================================
    with tab2:

        # Moving Averages
        df['MA50'] = df['Close'].rolling(50).mean()
        df['MA200'] = df['Close'].rolling(200).mean()

        fig_ma = go.Figure()

        fig_ma.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close Price'))
        fig_ma.add_trace(go.Scatter(x=df['Date'], y=df['MA50'], name='MA50'))
        fig_ma.add_trace(go.Scatter(x=df['Date'], y=df['MA200'], name='MA200'))

        fig_ma.update_layout(
            title="Moving Averages",
            template="plotly_dark",
            height=600,
            hovermode="x unified"
        )
        st.markdown("""
### What are Moving Averages?

Moving averages smooth out short-term price fluctuations to reveal 
the overall market trend.

- MA50 → medium-term trend
- MA200 → long-term trend

When MA50 crosses above MA200, it may indicate bullish momentum.
When MA50 drops below MA200, it may indicate bearish momentum.
""")
        st.plotly_chart(fig_ma, use_container_width=True)

        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.ewm(com=13, min_periods=14).mean()
        avg_loss = loss.ewm(com=13, min_periods=14).mean()

        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI'))

        fig_rsi.update_layout(
            title="RSI Indicator",
            template="plotly_dark",
            height=400,
            hovermode="x unified"
        )
        st.markdown("""
### What is RSI?

RSI (Relative Strength Index) measures market momentum and helps identify 
whether a stock may be overbought or oversold.

- RSI above 70 → potentially overbought
- RSI below 30 → potentially oversold
- RSI around 50 → neutral momentum
""")
        st.plotly_chart(fig_rsi, use_container_width=True)

    # =========================================================
    # TAB 3 — AI FORECAST
    # =========================================================
    with tab3:

        df_train = df[['Date', 'Close']].copy()
        df_train.columns = ['ds', 'y']
        df_train['ds'] = pd.to_datetime(df_train['ds'])
        df_train['ds'] = df_train['ds'].dt.tz_localize(None)
        df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')
        df_train.dropna(inplace=True)

        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )

        model.fit(df_train)

        future = model.make_future_dataframe(periods=prediction_days)
        forecast = model.predict(future)

        st.subheader("AI Forecast")

        st.warning("""
⚠️ Prophet models trend and seasonality patterns only. 
It cannot account for earnings surprises, macro events, or market sentiment.
Treat this as a statistical baseline — not a financial prediction.
See the **Model Evaluation** tab to understand how reliable this forecast is.
""")

        fig_forecast = plot_plotly(model, forecast)
        fig_forecast.update_layout(template="plotly_dark", height=700)
        st.plotly_chart(fig_forecast, use_container_width=True)

        pred_price = float(forecast['yhat'].iloc[-1])
        diff = pred_price - current_price
        perc = (diff / current_price) * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Current Price", f"${current_price:.2f}")
        c2.metric(f"Predicted ({prediction_days}d)", f"${pred_price:.2f}")
        c3.metric("Expected Change", f"{perc:.2f}%", delta=f"{diff:.2f}")

        if perc > 10:
            signal = "🚀 Strong Bullish"
        elif perc > 3:
            signal = "📈 Bullish"
        elif perc < -10:
            signal = "🔻 Strong Bearish"
        elif perc < -3:
            signal = "📉 Bearish"
        else:
            signal = "⚖ Neutral"

        st.success(f"AI Forecast Signal: {signal}")

        st.subheader("Forecast Components")
        fig_components = model.plot_components(forecast)
        st.pyplot(fig_components)

    # =========================================================
    # TAB 4 — MODEL EVALUATION
    # =========================================================
    with tab4:

        st.subheader("📊 Model Reliability Evaluation")

        st.markdown("""
This tab answers one question: **how much should you trust this forecast?**  
We evaluate the model by testing it on data it has never seen — the most recent historical days.
""")

        # ── Prepare data ──────────────────────────────────────
        df_eval = df[['Date', 'Close']].copy()
        df_eval.columns = ['ds', 'y']
        df_eval['ds'] = pd.to_datetime(df_eval['ds']).dt.tz_localize(None)
        df_eval['y'] = pd.to_numeric(df_eval['y'], errors='coerce')
        df_eval.dropna(inplace=True)

        if len(df_eval) < backtest_days + 30:
            st.error("Not enough historical data for the selected backtest window. Try a longer historical range or reduce the backtest window.")
            st.stop()

        cutoff = len(df_eval) - backtest_days
        train_bt = df_eval.iloc[:cutoff]
        actual_bt = df_eval.iloc[cutoff:].reset_index(drop=True)

        with st.spinner("Running backtest — training model on historical data..."):
            m_bt = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05,
                interval_width=0.95          # wider CI — needed for multi-day horizon
            )
            m_bt.fit(train_bt)
            future_bt = m_bt.make_future_dataframe(periods=backtest_days)
            fc_bt = m_bt.predict(future_bt)

        predicted_vals = fc_bt['yhat'].iloc[-backtest_days:].values
        actual_vals    = actual_bt['y'].values
        lower_vals     = fc_bt['yhat_lower'].iloc[-backtest_days:].values
        upper_vals     = fc_bt['yhat_upper'].iloc[-backtest_days:].values

        # ── Error Metrics ─────────────────────────────────────
        mae  = np.mean(np.abs(predicted_vals - actual_vals))
        mape = np.mean(np.abs((predicted_vals - actual_vals) / actual_vals)) * 100
        rmse = np.sqrt(np.mean((predicted_vals - actual_vals) ** 2))

        # How often did actual price fall inside the confidence interval?
        within_ci = np.mean(
            (actual_vals >= lower_vals) & (actual_vals <= upper_vals)
        ) * 100

        # Uncertainty band width at end of backtest period
        avg_band     = np.mean(upper_vals - lower_vals)
        avg_band_pct = (avg_band / current_price) * 100

        # Direction accuracy — did model get up/down right?
        actual_direction   = np.diff(actual_vals) > 0
        pred_direction     = np.diff(predicted_vals) > 0
        direction_accuracy = np.mean(actual_direction == pred_direction) * 100

        # Direction accuracy context label
        if direction_accuracy >= 60:
            dir_label = "✅ Better than random"
            dir_note  = "The model has some directional edge."
        elif direction_accuracy >= 50:
            dir_label = "⚠️ Marginally above random"
            dir_note  = "Directional edge is weak — barely better than a coin flip."
        else:
            dir_label = "❌ Worse than random"
            dir_note  = f"At {direction_accuracy:.1f}%, flipping a coin would have been more accurate. The model has no reliable directional signal for this stock."

        # ── Reliability Label ─────────────────────────────────
        # Base on MAPE but also penalise if direction accuracy is sub-random
        if mape < 5 and direction_accuracy >= 55:
            reliability_label = "🟢 High"
            reliability_color = "#00C897"
            reliability_note  = "Model tracked prices closely during the backtest period."
        elif mape < 15 and direction_accuracy >= 50:
            reliability_label = "🟡 Moderate"
            reliability_color = "#FFD700"
            reliability_note  = "Model captured the general trend but had noticeable errors."
        elif direction_accuracy < 50:
            reliability_label = "🔴 Low — Direction Worse Than Random"
            reliability_color = "#FF4B4B"
            reliability_note  = f"Direction accuracy was {direction_accuracy:.1f}% — below 50%. This model provides no reliable signal for this stock in this period. Use with extreme caution."
        else:
            reliability_label = "🔴 Low"
            reliability_color = "#FF4B4B"
            reliability_note  = "Large errors in backtest. Treat forecast directionally only."

        # ── Section 1: Summary Card ───────────────────────────
        st.markdown(f"""
<div style="background: #1A1F2E; border-left: 5px solid {reliability_color}; 
            padding: 20px; border-radius: 10px; margin-bottom: 20px;">
    <h3 style="color: {reliability_color}; margin: 0;">
        Overall Reliability: {reliability_label}
    </h3>
    <p style="color: #CCCCCC; margin-top: 8px;">{reliability_note}</p>
    <p style="color: #888; font-size: 0.85em; margin: 0;">
        Based on a {backtest_days}-day backtest — the model was trained on all data 
        except the last {backtest_days} days, then its predictions were compared to 
        what actually happened.
    </p>
</div>
""", unsafe_allow_html=True)

        # ── Section 2: Metrics Row ────────────────────────────
        st.subheader("Error Metrics")

        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "MAPE",
            f"{mape:.1f}%",
            help="Mean Absolute Percentage Error — average % the model was off. Lower is better."
        )
        m2.metric(
            "MAE",
            f"${mae:.2f}",
            help="Mean Absolute Error — average dollar error per day. Lower is better."
        )
        m3.metric(
            "RMSE",
            f"${rmse:.2f}",
            help="Root Mean Square Error — penalises large errors more. Lower is better."
        )
        m4.metric(
            "Direction Accuracy",
            f"{direction_accuracy:.1f}%",
            help="How often the model correctly predicted whether price would go up or down. 50% = coin flip."
        )

        # Direction accuracy callout — always shown, since it's the most honest signal
        if direction_accuracy >= 60:
            st.success(f"✅ Direction Accuracy: {dir_label} — {dir_note}")
        elif direction_accuracy >= 50:
            st.warning(f"⚠️ Direction Accuracy: {dir_label} — {dir_note}")
        else:
            st.error(f"❌ Direction Accuracy: {dir_label} — {dir_note}")

        st.markdown("---")

        # ── Section 3: Confidence Interval Quality ────────────
        st.subheader("Confidence Interval Quality")

        st.caption("""
Prophet's confidence interval is calibrated for short-horizon forecasts. 
Over a multi-day backtest, the actual price often drifts outside the band — 
this reflects genuine forecast uncertainty, not a model bug. 
We use a 95% CI width to give the model the best chance of capturing actual prices.
""")

        ci1, ci2 = st.columns(2)

        ci1.metric(
            "Actual Price Inside CI (95%)",
            f"{within_ci:.1f}%",
            help="How often the real price fell within the 95% uncertainty band during the backtest."
        )
        ci2.metric(
            "Avg Uncertainty Band",
            f"±${avg_band/2:.2f}  ({avg_band_pct/2:.1f}%)",
            help="Average half-width of the confidence interval in dollar and % terms."
        )

        if within_ci >= 75:
            st.success(f"✅ CI captured actual price {within_ci:.1f}% of the time — well-calibrated for a multi-day forecast.")
        elif within_ci >= 40:
            st.warning(f"⚠️ CI captured actual price {within_ci:.1f}% of the time — the model underestimates how far prices can move.")
        else:
            st.error(f"❌ CI captured actual price {within_ci:.1f}% of the time — the uncertainty band is too narrow for this stock's volatility. The model's confidence is not justified.")

        st.markdown("---")

        # ── Section 4: Backtest Chart ─────────────────────────
        st.subheader("Backtest: Predicted vs Actual")

        fig_bt = go.Figure()

        fig_bt.add_trace(go.Scatter(
            x=actual_bt['ds'],
            y=actual_vals,
            name="Actual Price",
            line=dict(color="#00D4FF", width=2)
        ))

        fig_bt.add_trace(go.Scatter(
            x=actual_bt['ds'],
            y=predicted_vals,
            name="Predicted Price",
            line=dict(color="#FF6B35", width=2, dash="dash")
        ))

        fig_bt.add_trace(go.Scatter(
            x=list(actual_bt['ds']) + list(actual_bt['ds'])[::-1],
            y=list(upper_vals) + list(lower_vals)[::-1],
            fill="toself",
            fillcolor="rgba(255, 107, 53, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Confidence Interval"
        ))

        fig_bt.update_layout(
            title=f"{ticker} — Last {backtest_days} Days: Model vs Reality",
            template="plotly_dark",
            height=500,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig_bt, use_container_width=True)

        # ── Section 5: Daily Error Chart ──────────────────────
        st.subheader("Daily Prediction Error")

        daily_error = predicted_vals - actual_vals
        daily_error_pct = (daily_error / actual_vals) * 100

        error_colors = ["#FF4B4B" if e > 0 else "#00C897" for e in daily_error]

        fig_err = go.Figure()

        fig_err.add_trace(go.Bar(
            x=actual_bt['ds'],
            y=daily_error_pct,
            name="Error %",
            marker_color=error_colors
        ))

        fig_err.add_hline(y=0, line_color="white", line_width=1)

        fig_err.update_layout(
            title="Daily Error (Predicted − Actual) as % of Actual Price",
            template="plotly_dark",
            height=350,
            yaxis_title="Error %"
        )

        st.plotly_chart(fig_err, use_container_width=True)

        st.caption("🔴 Red bars = model overestimated  |  🟢 Green bars = model underestimated")

        st.markdown("---")

        # ── Section 6: Plain English Interpretation ───────────
        st.subheader("📋 How to Interpret These Results")

        st.markdown(f"""
| Metric | Your Result | Benchmark | What It Means |
|--------|-------------|-----------|---------------|
| MAPE | {mape:.1f}% | < 5% great, < 15% ok | On average, predictions were off by {mape:.1f}% of the actual price |
| MAE | ${mae:.2f} | Lower is better | In dollar terms, average daily error was ${mae:.2f} |
| RMSE | ${rmse:.2f} | Lower is better | Penalises large single-day errors more heavily than MAE |
| Direction Accuracy | {direction_accuracy:.1f}% | > 50% = better than random | The model called up/down correctly {direction_accuracy:.1f}% of the time |
| CI Coverage (95%) | {within_ci:.1f}% | > 75% = well-calibrated | Actual price was inside the shaded band {within_ci:.1f}% of the time |
""")

        st.info(f"""
**Bottom line for {ticker}:**  
Prophet is a trend + seasonality model. It works best for stocks with 
stable long-term trends and predictable seasonal patterns.

A direction accuracy of **{direction_accuracy:.1f}%** {"is better than random — the model has some edge." if direction_accuracy > 50 else "is worse than a coin flip — the model has no reliable edge for this stock in this period."}  

It cannot predict sudden moves driven by earnings, macro events, or sentiment shifts.  
{"The forecast direction may have some value." if direction_accuracy > 55 else "Treat the forecast as a rough trend baseline only, not a directional signal."}
""")

    # =========================================================
    # TAB 5 — COMPANY INFO
    # =========================================================
    with tab5:

        st.subheader("Company Information")

        st.write(f"**Sector:** {info.get('sector', 'N/A')}")
        st.write(f"**Industry:** {info.get('industry', 'N/A')}")
        st.write(f"**Country:** {info.get('country', 'N/A')}")
        st.write(f"**Website:** {info.get('website', 'N/A')}")

        st.subheader("Business Summary")
        st.write(info.get('longBusinessSummary', 'N/A'))

# =========================
# ERROR HANDLING
# =========================
except Exception as e:
    st.error(f"An error occurred: {e}")

# =========================
# DISCLAIMER
# =========================
st.warning("""
This application provides statistical forecasts based on historical 
market data and technical indicators.

It should not be considered financial advice or used as the sole basis 
for investment decisions.
""")