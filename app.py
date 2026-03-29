"""
ATIS v4.0 — Trading Dashboard
Professional Dash app with TradingView-style chart, trade signal panel,
model consensus, news sentiment feed, and trade journal.
"""
import sys
import json
import threading
import time as _time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import html, dcc, Input, Output, State, callback_context, no_update
import dash_bootstrap_components as dbc

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.settings import (
    PROCESSED_DIR, MODELS_DIR, DASH_HOST, DASH_PORT,
    DASH_DEBUG, TARGET_ACCURACY, CONFIDENCE_GATE,
)
from src.signals.strike_selector import StrikeSelector
from src.signals.trade_manager import TradeManager
from src.news.news_fetcher import NewsFetcher
from src.news.finbert_agent import FinBERTAgent

# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def load_chart_data(timeframe="1min", n_bars=500):
    """Load OHLCV data for charting."""
    tf_map = {"1m": "1min", "5m": "5min", "15m": "15min", "1H": "1H", "1D": "1H"}
    tf = tf_map.get(timeframe, "1min")
    path = PROCESSED_DIR / f"nifty_{tf}_clean.parquet"
    if not path.exists():
        path = PROCESSED_DIR / "nifty_1min_clean.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.tail(n_bars)


def load_model_results():
    """Load training/backtest results."""
    # Priority 1: Final Resync Report
    final_path = MODELS_DIR / "final_report.json"
    if final_path.exists():
        with open(final_path) as f:
            return {"models": json.load(f)}

    # Priority 2: Backtest Report
    report_path = MODELS_DIR / "backtest_report.json"
    if report_path.exists():
        with open(report_path) as f:
            return json.load(f)
    
    training_path = MODELS_DIR / "training_report.json"
    if training_path.exists():
        try:
            with open(training_path) as f:
                data = json.load(f)
                if isinstance(data, dict) and "final_holdout" in data:
                    return data["final_holdout"]
                elif isinstance(data, list) and len(data) > 0:
                    return data[-1]
        except:
            pass
    return {}


def load_trade_journal():
    """Load trade log CSV."""
    path = Path(ROOT) / "data" / "trades" / "trade_log.csv"
    if path.exists() and path.stat().st_size > 10:
        return pd.read_csv(path)
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════
# CHART BUILDER
# ═══════════════════════════════════════════════════════════════════

def build_candlestick_chart(df, timeframe="1m", overlays=None):
    """Build professional TradingView-style candlestick chart."""
    if df.empty:
        return go.Figure()

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.65, 0.20, 0.15],
        subplot_titles=("NIFTY 50", "Volume", "RSI"),
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df["timestamp"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        increasing=dict(line=dict(color="#26a69a"), fillcolor="#26a69a"),
        decreasing=dict(line=dict(color="#ef5350"), fillcolor="#ef5350"),
        name="NIFTY", whiskerwidth=0.5,
    ), row=1, col=1)

    # EMAs
    if overlays is None:
        overlays = ["ema_21", "ema_50", "ema_200"]
    colors = {"ema_9": "#ff9800", "ema_21": "#2196f3", "ema_50": "#e91e63",
              "ema_200": "#9c27b0", "ema_13": "#00bcd4"}
    for ema in overlays:
        if ema in df.columns:
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df[ema], name=ema.upper(),
                line=dict(width=1.5, color=colors.get(ema, "#888")),
                opacity=0.8,
            ), row=1, col=1)

    # Bollinger Bands
    if "bb_upper" in df.columns and "bb_lower" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["bb_upper"], name="BB Upper",
            line=dict(width=0.8, dash="dot", color="#78909c"), opacity=0.5,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["bb_lower"], name="BB Lower",
            line=dict(width=0.8, dash="dot", color="#78909c"),
            fill="tonexty", fillcolor="rgba(120,144,156,0.08)", opacity=0.5,
        ), row=1, col=1)

    # Volume bars
    vol_colors = ["#26a69a" if c >= o else "#ef5350"
                  for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(
        x=df["timestamp"], y=df["volume"],
        marker_color=vol_colors, name="Volume", opacity=0.7,
    ), row=2, col=1)

    # RSI
    if "rsi_14" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["rsi_14"], name="RSI 14",
            line=dict(width=1.5, color="#ff9800"),
        ), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red",
                      opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green",
                      opacity=0.5, row=3, col=1)

    # Signal Overlays (Triangle Markers)
    if "signal" in df.columns:
        buy_signals = df[df["signal"] == "BULLISH"]
        sell_signals = df[df["signal"] == "BEARISH"]
        
        fig.add_trace(go.Scatter(
            x=buy_signals["timestamp"], y=buy_signals["low"] * 0.9995,
            mode="markers", marker=dict(symbol="triangle-up", size=12, color="#26a69a"),
            name="BUY", hoverinfo="text", text="Supervisor BUY Signal"
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=sell_signals["timestamp"], y=sell_signals["high"] * 1.0005,
            mode="markers", marker=dict(symbol="triangle-down", size=12, color="#ef5350"),
            name="SELL", hoverinfo="text", text="Supervisor SELL Signal"
        ), row=1, col=1)

    # Risk/Reward Levels (Horizontal Lines) for Active Trade
    # (Simplified: we'll use fig.add_hline for the dashboard version)

    # Layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0e17",
        plot_bgcolor="#0a0e17",
        font=dict(family="Inter, sans-serif", color="#e0e0e0"),
        height=700,
        margin=dict(l=60, r=20, t=40, b=20),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=10)),
        hovermode="x unified",
        uirevision="constant_state"  # Crucial for silent refresh (preserves zoom level)
    )

    # Crosshair and Rangebreaks (Remove weekend/overnight gaps)
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),  # Hide weekends
            dict(bounds=[15.5, 9.25], pattern="hour"),  # Hide 15:30 to 09:15
        ],
        showgrid=True, gridwidth=0.5, gridcolor="#1a2030",

        showspikes=True, spikecolor="#666", spikethickness=0.5,
        spikemode="across", spikesnap="cursor",
    )
    fig.update_yaxes(
        showgrid=True, gridwidth=0.5, gridcolor="#1a2030",
        showspikes=True, spikecolor="#666", spikethickness=0.5,
    )

    return fig


# ═══════════════════════════════════════════════════════════════════
# DASHBOARD COMPONENTS
# ═══════════════════════════════════════════════════════════════════

def make_signal_card(signal=None, active_trade=None):
    """Build the trade signal card with real-time locking."""
    if active_trade:
        status = active_trade.get("status", "")
        status_color = {"LOCKED": "#ff9800", "WIN": "#4caf50",
                        "LOSS": "#f44336", "TIMEOUT": "#9e9e9e"}.get(status, "#666")
        status_icon = {"LOCKED": "🔒", "WIN": "✅", "LOSS": "❌",
                       "TIMEOUT": "⏰"}.get(status, "")

        # Live P&L (estimated from exit or current tracking)
        entry = active_trade.get("entry", 0)
        exit_p = active_trade.get("exit_price")
        pnl = active_trade.get("pnl", 0)
        pnl_pct = active_trade.get("pnl_percent", 0)
        pnl_color = "#4caf50" if pnl >= 0 else "#f44336"

        # Border style — pulsing for LOCKED, solid for resolved
        border_style = f"2px solid {status_color}"

        header_text = f"{status_icon} {status}"
        if status == "LOCKED":
            header_text += " — Live Tracking"
        elif status in ("WIN", "LOSS"):
            header_text += f" — P&L: ₹{pnl:.2f} ({pnl_pct:+.1f}%)"

        body_children = [
            html.Div([
                html.Span(active_trade.get("symbol", ""), className="fw-bold",
                          style={"fontSize": "1.3rem", "color": "#e0e0e0"}),
                html.Span(f"  ({active_trade.get('direction', '')})",
                          style={"color": "#9e9e9e"}),
            ]),
            html.Div(f"Trade ID: {active_trade.get('id', '')}",
                     style={"color": "#555", "fontSize": "0.8rem", "fontFamily": "monospace"}),
            html.Hr(style={"borderColor": "#333"}),
            _kv_row("Entry", f"₹{entry:.2f}", "#2196f3"),
            _kv_row("SL", f"₹{active_trade.get('sl', 0):.2f}", "#f44336"),
            _kv_row("T1", f"₹{active_trade.get('t1', 0):.2f}",
                     "#4caf50" if active_trade.get("t1_hit") else "#666",
                     "✅" if active_trade.get("t1_hit") else "⬜"),
            _kv_row("T2", f"₹{active_trade.get('t2', 0):.2f}",
                     "#4caf50" if active_trade.get("t2_hit") else "#666",
                     "✅" if active_trade.get("t2_hit") else "⬜"),
            _kv_row("T3", f"₹{active_trade.get('t3', 0):.2f}",
                     "#4caf50" if active_trade.get("t3_hit") else "#666",
                     "✅" if active_trade.get("t3_hit") else "⬜"),
            html.Hr(style={"borderColor": "#333"}),
            _kv_row("Confidence", f"{active_trade.get('confidence', 0):.0%}", "#ff9800"),
        ]

        # Show P&L for resolved trades
        if status in ("WIN", "LOSS", "TIMEOUT"):
            body_children.append(html.Hr(style={"borderColor": "#333"}))
            body_children.append(
                html.Div([
                    html.Span("Result: ", style={"color": "#999"}),
                    html.Span(f"₹{pnl:.2f} ({pnl_pct:+.1f}%)",
                              style={"color": pnl_color, "fontWeight": "bold",
                                     "fontSize": "1.2rem", "fontFamily": "monospace"}),
                ], style={"textAlign": "center", "padding": "8px 0"})
            )

        return dbc.Card([
            dbc.CardHeader(
                html.H5(header_text, className="mb-0", style={"color": status_color}),
                style={"backgroundColor": "#111827", "borderBottom": border_style}
            ),
            dbc.CardBody(body_children, style={"backgroundColor": "#0d1117"}),
        ], style={"backgroundColor": "#0d1117", "border": f"1px solid {status_color}",
                   "borderRadius": "12px"})


    # No active trade
    if signal:
        direction_color = "#4caf50" if signal.get("direction") == "BULLISH" else "#f44336"
        emoji = "🟢" if signal.get("direction") == "BULLISH" else "🔴"
        return dbc.Card([
            dbc.CardHeader(
                html.H5(f"{emoji} NEW SIGNAL", className="mb-0",
                         style={"color": direction_color}),
                style={"backgroundColor": "#111827",
                       "borderBottom": f"2px solid {direction_color}"}
            ),
            dbc.CardBody([
                html.Div(signal.get("symbol", ""), className="fw-bold",
                         style={"fontSize": "1.3rem", "color": "#e0e0e0"}),
                html.Hr(style={"borderColor": "#333"}),
                _kv_row("Entry", f"₹{signal.get('entry', 0):.2f}", "#2196f3"),
                _kv_row("SL", f"₹{signal.get('sl', 0):.2f}", "#f44336"),
                *[_kv_row(f"T{i+1}", f"₹{t:.2f}", "#4caf50")
                  for i, t in enumerate(signal.get("targets", []))],
                _kv_row("Confidence", f"{signal.get('confidence', 0):.0%}", "#ff9800"),
            ], style={"backgroundColor": "#0d1117"}),
        ], style={"backgroundColor": "#0d1117", "border": "1px solid #21262d",
                   "borderRadius": "12px"})

    # Waiting state
    return dbc.Card([
        dbc.CardBody([
            html.Div("⏳ WAITING", style={"fontSize": "1.5rem", "color": "#9e9e9e",
                                           "textAlign": "center", "padding": "30px"}),
            html.P("Models analyzing... waiting for high-confidence setup",
                   style={"color": "#666", "textAlign": "center"}),
        ], style={"backgroundColor": "#0d1117"}),
    ], style={"backgroundColor": "#0d1117", "border": "1px solid #21262d",
               "borderRadius": "12px"})


def _kv_row(label, value, color="#e0e0e0", icon=""):
    return html.Div([
        html.Span(f"{label}: ", style={"color": "#9e9e9e", "width": "80px",
                                        "display": "inline-block"}),
        html.Span(value, style={"color": color, "fontWeight": "bold",
                                 "fontFamily": "monospace", "fontSize": "1.1rem"}),
        html.Span(f" {icon}", style={"marginLeft": "8px"}) if icon else "",
    ], style={"padding": "4px 0"})


def make_model_gauges(results):
    """Build model accuracy gauges."""
    model_results = results.get("model_results", results.get("models", {}))
    if isinstance(model_results, list):
        items = [(r["name"], r.get("f1", r.get("accuracy", 0))) for r in model_results]
    elif isinstance(model_results, dict):
        items = [(k, v.get("f1", v.get("accuracy", 0)) if isinstance(v, dict) else v)
                 for k, v in model_results.items()]
    else:
        items = []

    if not items:
        return html.Div("No model results yet", style={"color": "#666", "padding": "20px"})

    gauges = []
    for name, score in items:
        color = "#4caf50" if score >= TARGET_ACCURACY else "#ff9800" if score >= 0.5 else "#f44336"
        pct = score * 100
        gauges.append(
            html.Div([
                html.Div([
                    html.Span(name.replace("_", " ").title(),
                              style={"color": "#ccc", "fontSize": "0.8rem"}),
                    html.Span(f"{pct:.1f}%",
                              style={"color": color, "fontWeight": "bold", "float": "right"}),
                ]),
                html.Div(
                    html.Div(style={
                        "width": f"{min(pct, 100):.0f}%", "height": "6px",
                        "backgroundColor": color, "borderRadius": "3px",
                        "transition": "width 0.3s ease",
                    }),
                    style={"backgroundColor": "#1a2030", "borderRadius": "3px",
                           "height": "6px", "marginTop": "4px"},
                ),
            ], style={"marginBottom": "12px"})
        )

    return html.Div(gauges)


def make_news_panel(headlines=None):
    """Build news sentiment panel."""
    if not headlines:
        return html.Div([
            html.P("📡 News feed inactive", style={"color": "#666"}),
            html.P("Fill API keys in config/settings.py or wait for RSS fetch",
                   style={"color": "#444", "fontSize": "0.85rem"}),
        ])

    items = []
    for h in headlines[:8]:
        score = h.get("sentiment", 0)
        color = "#4caf50" if score > 0.2 else "#f44336" if score < -0.2 else "#9e9e9e"
        impact = "⚡" if h.get("high_impact") else ""
        items.append(html.Div([
            html.Span(f"{impact} ", style={"color": "#ff9800"}) if impact else "",
            html.Span(h.get("title", "")[:80], style={"color": "#ccc", "fontSize": "0.85rem"}),
            html.Span(f" [{score:+.2f}]",
                      style={"color": color, "fontSize": "0.8rem", "fontFamily": "monospace"}),
        ], style={"padding": "6px 0", "borderBottom": "1px solid #1a2030"}))

    return html.Div(items)


def make_equity_curve(journal_df):
    """Build equity curve from trade journal."""
    if journal_df.empty or "pnl" not in journal_df.columns:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#0a0e17",
            plot_bgcolor="#0a0e17", height=250,
            annotations=[dict(text="No trades yet", showarrow=False,
                              font=dict(size=16, color="#666"))],
        )
        return fig

    journal_df["cum_pnl"] = journal_df["pnl"].cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(journal_df))), y=journal_df["cum_pnl"],
        fill="tozeroy",
        fillcolor="rgba(38,166,154,0.15)",
        line=dict(color="#26a69a", width=2),
        name="Equity",
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0a0e17",
        plot_bgcolor="#0a0e17", height=250,
        margin=dict(l=40, r=10, t=20, b=20),
        yaxis_title="Cumulative P&L (₹)",
        showlegend=False,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════
# DASH APP
# ═══════════════════════════════════════════════════════════════════

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap",
    ],
    title="ATIS v4.0 — Trading Dashboard",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

app.layout = dbc.Container([
    # ── HEADER ──────────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H2("⚡ ATIS v4.0", className="mb-0",
                         style={"fontWeight": "700", "background": "linear-gradient(90deg, #4caf50, #2196f3)",
                                "-webkit-background-clip": "text", "-webkit-text-fill-color": "transparent"}),
                html.Small("Autonomous Trading Intelligence System",
                           style={"color": "#666"}),
                dbc.Button("🧠 Model Analysis", id="btn-thinking", size="sm", color="primary",
                           outline=True, className="mt-2 ms-3")
            ], style={"display": "flex", "alignItems": "center"}),
        ], width=6),
        dbc.Col([
            html.Div([
                html.Div(id="live-clock", style={"color": "#9e9e9e", "textAlign": "right",
                                                   "fontFamily": "monospace", "fontSize": "1.1rem"}),
                html.Div(id="market-status", style={"textAlign": "right", "fontSize": "0.9rem"}),
            ]),
        ], width=6),
    ], className="py-3", style={"borderBottom": "1px solid #21262d"}),

    # ── TIMEFRAME SELECTOR ──────────────────────────────────
    dbc.Row([
        dbc.Col([
            dbc.ButtonGroup([
                dbc.Button("1m", id="tf-1m", outline=True, color="light", size="sm",
                           n_clicks=0, className="me-1"),
                dbc.Button("5m", id="tf-5m", outline=True, color="primary", size="sm",
                           n_clicks=0, className="me-1", active=True),
                dbc.Button("15m", id="tf-15m", outline=True, color="light", size="sm",
                           n_clicks=0, className="me-1"),
                dbc.Button("1H", id="tf-1H", outline=True, color="light", size="sm",
                           n_clicks=0, className="me-1"),
            ]),
            dbc.ButtonGroup([
                dbc.Button("EMA", id="ov-ema", outline=True, color="info", size="sm",
                           n_clicks=0, className="ms-3 me-1", active=True),
                dbc.Button("BB", id="ov-bb", outline=True, color="info", size="sm",
                           n_clicks=0, className="me-1"),
                dbc.Button("Vol", id="ov-vol", outline=True, color="info", size="sm",
                           n_clicks=0),
            ]),
        ]),
    ], className="py-2"),

    # ── MAIN LAYOUT ─────────────────────────────────────────
    dbc.Row([
        # Chart (left)
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id="main-chart", config={
                        "scrollZoom": True,
                        "displayModeBar": True,
                        "modeBarButtonsToAdd": ["drawline", "drawopenpath",
                                                 "drawcircle", "eraseshape"],
                        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                    }),
                ], style={"padding": "0"}),
            ], style={"backgroundColor": "#0a0e17", "border": "1px solid #21262d",
                       "borderRadius": "12px"}),
        ], width=8),

        # Signal + Controls (right)
        dbc.Col([
            # Trade Signal Card
            html.Div(id="signal-card"),
            html.Br(),

            # VIX Regime
            dbc.Card([
                dbc.CardHeader(html.H6("🌡️ VIX Regime", className="mb-0"),
                               style={"backgroundColor": "#111827"}),
                dbc.CardBody(id="vix-gauge",
                             style={"backgroundColor": "#0d1117"}),
            ], className="mb-3", style={"border": "1px solid #21262d", "borderRadius": "12px"}),

            # News Alert
            dbc.Card([
                dbc.CardHeader(html.H6("📰 News Sentiment", className="mb-0"),
                               style={"backgroundColor": "#111827"}),
                dbc.CardBody(id="news-panel",
                             style={"backgroundColor": "#0d1117", "maxHeight": "250px",
                                    "overflowY": "auto"}),
            ], style={"border": "1px solid #21262d", "borderRadius": "12px"}),
        ], width=4),
    ], className="mt-3"),

    # ── BOTTOM ROW ──────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H6("📊 Model Consensus", className="mb-0"),
                               style={"backgroundColor": "#111827"}),
                dbc.CardBody(id="model-gauges",
                             style={"backgroundColor": "#0d1117"}),
            ], style={"border": "1px solid #21262d", "borderRadius": "12px"}),
        ], width=4),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H6("📈 Equity Curve", className="mb-0"),
                               style={"backgroundColor": "#111827"}),
                dbc.CardBody([
                    dcc.Graph(id="equity-chart", config={"displayModeBar": False}),
                ], style={"backgroundColor": "#0d1117", "padding": "0"}),
            ], style={"border": "1px solid #21262d", "borderRadius": "12px"}),
        ], width=4),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H6("📋 Trade Journal", className="mb-0"),
                               style={"backgroundColor": "#111827"}),
                dbc.CardBody(id="trade-journal",
                             style={"backgroundColor": "#0d1117", "maxHeight": "300px",
                                    "overflowY": "auto"}),
            ], style={"border": "1px solid #21262d", "borderRadius": "12px"}),
        ], width=4),
    ], className="mt-3 mb-4"),

    # ── INTERVALS ───────────────────────────────────────────
    dcc.Interval(id="interval-fast", interval=1000, n_intervals=0),  # 1s Silent Refresh
    dcc.Interval(id="interval-slow", interval=30000, n_intervals=0),
    dcc.Store(id="timeframe-store", data="5m"),
    dcc.Store(id="engine-state", data={}),
    
    # ── MODALS ──────────────────────────────────────────────
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("🧠 Autonomous Engine Analysis")),
        dbc.ModalBody(id="model-thinking-body", style={"backgroundColor": "#0d1117"}),
    ], id="modal-thinking", size="lg", is_open=False, style={"backgroundColor": "transparent"}),

], fluid=True, style={"backgroundColor": "#0a0e17", "minHeight": "100vh",
                        "fontFamily": "Inter, sans-serif"})

# ═══════════════════════════════════════════════════════════════════
# CALLBACKS
# ═══════════════════════════════════════════════════════════════════

@app.callback(
    Output("modal-thinking", "is_open"),
    [Input("btn-thinking", "n_clicks")],
    [State("modal-thinking", "is_open")],
    prevent_initial_call=True
)
def toggle_modal(n1, is_open):
    if n1:
        return not is_open
    return is_open

@app.callback(
    Output("timeframe-store", "data"),
    [Input("tf-1m", "n_clicks"), Input("tf-5m", "n_clicks"),
     Input("tf-15m", "n_clicks"), Input("tf-1H", "n_clicks")],
    prevent_initial_call=True,
)
def update_timeframe(*clicks):
    ctx = callback_context
    if not ctx.triggered:
        return "5m"
    btn = ctx.triggered[0]["prop_id"].split(".")[0]
    return btn.replace("tf-", "")


def _load_engine_state():
    """Load the latest engine state from the live engine JSON export."""
    state_path = Path(ROOT) / "data" / "trades" / "engine_state.json"
    if state_path.exists():
        try:
            with open(state_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _build_thinking_panel(engine_state):
    """Build the Model Thinking panel content from engine state."""
    if not engine_state or "predictions" not in engine_state:
        return html.Div([
            html.P("⏳ Waiting for Live Engine data...",
                   style={"color": "#9e9e9e", "textAlign": "center", "padding": "40px"}),
            html.P("Start the engine: python live_engine_atis.py",
                   style={"color": "#555", "textAlign": "center", "fontFamily": "monospace"}),
        ])

    preds = engine_state.get("predictions", {})
    direction = engine_state.get("direction", "SIDEWAYS")
    confidence = engine_state.get("confidence", 0)
    spot = engine_state.get("spot", 0)
    latency = engine_state.get("latency_seconds", 0)

    label_map = {0: "BEARISH 🔴", 1: "SIDEWAYS ⚪", 2: "BULLISH 🟢"}
    model_descriptions = {
        "trend_catboost": ("📈 Trend (CatBoost)", "EMA crossovers, ADX, SuperTrend signals"),
        "fibo_xgboost": ("📐 Fibonacci (XGBoost)", "Fib retracement/extension levels, pivot zones"),
        "candle_catboost": ("🕯️ Candle (CatBoost)", "Candlestick patterns, body/wick ratios"),
        "trap_xgboost": ("🪤 Trap (XGBoost)", "Bull/bear trap detection, false breakouts"),
        "lgbm": ("⚡ LGBM (Ensemble)", "Full 256-feature gradient boosted ensemble"),
        "lstm": ("🧠 LSTM (Neural)", "60-bar sequential memory, temporal patterns"),
        "finbert": ("📰 FinBERT (Sentiment)", "News headline sentiment aggregation"),
    }

    rows = []
    for name, pred_val in preds.items():
        display_name, reasoning = model_descriptions.get(name, (name, "Custom model"))
        pred_label = label_map.get(pred_val, str(pred_val))
        color = "#4caf50" if pred_val == 2 else "#f44336" if pred_val == 0 else "#9e9e9e"

        rows.append(html.Div([
            html.Div([
                html.Span(display_name, style={"color": "#e0e0e0", "fontWeight": "600",
                                                "fontSize": "0.95rem"}),
                html.Span(f"  →  {pred_label}", style={"color": color, "fontWeight": "bold",
                                                         "fontFamily": "monospace", "fontSize": "1rem"}),
            ]),
            html.P(reasoning, style={"color": "#777", "fontSize": "0.8rem", "marginBottom": "4px",
                                      "paddingLeft": "12px"}),
        ], style={"padding": "8px 0", "borderBottom": "1px solid #1a2030"}))

    # Supervisor Summary
    sup_color = "#4caf50" if confidence >= 0.7 else "#ff9800" if confidence >= 0.5 else "#f44336"
    dir_emoji = "🟢" if direction == "BULLISH" else "🔴" if direction == "BEARISH" else "⚪"

    supervisor_section = html.Div([
        html.Hr(style={"borderColor": "#333"}),
        html.H6(f"🏛️ SUPERVISOR (L3 Meta-Learner) → {dir_emoji} {direction}",
                style={"color": sup_color, "fontWeight": "700"}),
        html.Div([
            html.Span("Confidence: ", style={"color": "#999"}),
            html.Span(f"{confidence:.1%}", style={"color": sup_color, "fontWeight": "bold",
                                                    "fontSize": "1.3rem", "fontFamily": "monospace"}),
        ]),
        html.P(f"The Supervisor aggregates all {len(preds)} model votes using a trained "
               f"Logistic Regression meta-learner. It weighs each model's historical accuracy "
               f"to produce a final confidence-weighted prediction.",
               style={"color": "#777", "fontSize": "0.8rem", "marginTop": "8px"}),
        html.Div([
            html.Span(f"Spot: ₹{spot:,.2f}", style={"color": "#ccc", "marginRight": "20px"}),
            html.Span(f"Latency: {latency:.3f}s", style={"color": "#ccc"}),
        ], style={"marginTop": "8px"}),
    ], style={"padding": "10px 0"})

    return html.Div([*rows, supervisor_section])


@app.callback(
    [Output("main-chart", "figure"),
     Output("signal-card", "children"),
     Output("live-clock", "children"),
     Output("market-status", "children"),
     Output("vix-gauge", "children"),
     Output("model-thinking-body", "children")],
    [Input("interval-fast", "n_intervals"),
     Input("timeframe-store", "data")],
)
def update_main(n, timeframe):
    now = datetime.now()
    clock = now.strftime("%H:%M:%S")

    # Market status
    hour = now.hour
    minute = now.minute
    mins = hour * 60 + minute
    if 9 * 60 + 15 <= mins <= 15 * 60 + 30 and now.weekday() < 5:
        market_status = html.Span("🟢 MARKET OPEN", style={"color": "#4caf50"})
    else:
        market_status = html.Span("🔴 MARKET CLOSED", style={"color": "#f44336"})

    # Load chart data
    df = load_chart_data(timeframe, n_bars=500)
    fig = build_candlestick_chart(df, timeframe)

    # Load live engine state
    engine_state = _load_engine_state()

    # Signal card (from engine state)
    signal = engine_state.get("signal")
    active_trade = engine_state.get("active_trade")
    signal_card = make_signal_card(signal=signal, active_trade=active_trade)

    # Model Thinking Panel
    thinking_panel = _build_thinking_panel(engine_state)

    # VIX gauge
    vix_val = 0.15
    if not df.empty and "vix_proxy" in df.columns:
        vix_val = df["vix_proxy"].iloc[-1]
        if pd.isna(vix_val):
            vix_val = 0.15

    vix_pct = float(vix_val) * 100
    if vix_pct > 25:
        vix_color, vix_label = "#f44336", "🔴 HIGH VOLATILITY"
    elif vix_pct > 15:
        vix_color, vix_label = "#ff9800", "🟡 MEDIUM"
    else:
        vix_color, vix_label = "#4caf50", "🟢 LOW VOLATILITY"

    vix_gauge = html.Div([
        html.Div(f"VIX Proxy: {vix_pct:.1f}%",
                 style={"color": vix_color, "fontWeight": "bold",
                         "fontSize": "1.2rem", "textAlign": "center"}),
        html.Div(vix_label, style={"color": "#999", "textAlign": "center",
                                     "fontSize": "0.9rem"}),
        html.Div(
            html.Div(style={
                "width": f"{min(vix_pct * 2, 100):.0f}%", "height": "8px",
                "backgroundColor": vix_color, "borderRadius": "4px",
            }),
            style={"backgroundColor": "#1a2030", "borderRadius": "4px",
                   "height": "8px", "marginTop": "10px"},
        ),
    ])

    return fig, signal_card, clock, market_status, vix_gauge, thinking_panel


@app.callback(
    [Output("model-gauges", "children"),
     Output("equity-chart", "figure"),
     Output("trade-journal", "children"),
     Output("news-panel", "children")],
    Input("interval-slow", "n_intervals"),
)
def update_slow(n):
    # Model gauges
    results = load_model_results()
    gauges = make_model_gauges(results)

    # Equity curve
    journal_df = load_trade_journal()
    equity_fig = make_equity_curve(journal_df)

    # Trade journal table
    if not journal_df.empty:
        cols = ["id", "symbol", "direction", "entry", "sl", "status", "pnl"]
        display_cols = [c for c in cols if c in journal_df.columns]
        if display_cols:
            rows = journal_df[display_cols].tail(10).to_dict("records")
            table = dbc.Table([
                html.Thead(html.Tr([html.Th(c, style={"color": "#999", "borderColor": "#333"})
                                     for c in display_cols])),
                html.Tbody([
                    html.Tr([
                        html.Td(str(row.get(c, "")),
                                style={"color": "#4caf50" if c == "pnl" and row.get(c, 0) > 0
                                       else "#f44336" if c == "pnl" else "#ccc",
                                       "borderColor": "#333", "fontSize": "0.85rem"})
                        for c in display_cols
                    ]) for row in rows
                ]),
            ], bordered=True, dark=True, hover=True, size="sm",
               style={"marginBottom": "0"})
        else:
            table = html.P("No trade data", style={"color": "#666"})
    else:
        table = html.P("No trades yet", style={"color": "#666"})

    # News panel (Real-time Integration)
    try:
        fetcher = NewsFetcher()
        agent = FinBERTAgent()
        raw_news = fetcher.fetch_all()
        scored_news = []
        for n in raw_news[:5]:
            sentiment = agent.predict_sentiment(n["title"])
            n["sentiment"] = sentiment
            scored_news.append(n)
        news_panel = make_news_panel(scored_news)
    except:
        news_panel = make_news_panel()

    return gauges, equity_fig, table, news_panel


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

server = app.server

if __name__ == "__main__":
    print("=" * 50)
    print("  ATIS v4.0 — Trading Dashboard")
    print(f"  http://{DASH_HOST}:{DASH_PORT}")
    print("=" * 50)
    app.run(host=DASH_HOST, port=DASH_PORT, debug=DASH_DEBUG)
