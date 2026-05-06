# eToro AI Trading Agent

An AI-powered trading agent for eToro Agent Portfolios. Uses Claude to dynamically select and execute trading strategies based on real-time market conditions.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Dashboard (dashboard/index.html)                │
│  Browser UI — shows portfolio, positions, logs   │
└─────────────────┬───────────────────────────────┘
                  │ HTTP
┌─────────────────▼───────────────────────────────┐
│  API Server (api/server.py - FastAPI)            │
│  Exposes /portfolio /cycles /agent/start etc.    │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│  AI Trading Agent (agent/trading_agent.py)       │
│  Claude analyses market data → decides trades   │
└────────┬──────────────────────┬─────────────────┘
         │                      │
  eToro API              Anthropic API
  (trades)               (decisions)
```

## Quick Start

### 1. Prerequisites
- Python 3.12+
- An eToro account with Agent Portfolios enabled (min $200 funded)
- An Anthropic API key

### 2. Get your credentials
1. In eToro desktop: **Side menu → Agent Portfolios → Create Portfolio**
2. Fund it (minimum $200)
3. Copy your `API Key` and `User Key`
4. Get your Anthropic key from https://console.anthropic.com

### 3. Configure
```bash
cp .env.example .env
# Edit .env and fill in your keys
```

### 4. Install & run locally
```bash
pip install -r requirements.txt
cd api && python server.py
```

Then open `dashboard/index.html` in your browser.

---

## Deploy to Cloud

### Option A: Railway (easiest, free tier)
```bash
railway login
railway init
railway up
```

### Option B: Fly.io
```bash
fly launch
fly secrets set ETORO_API_KEY=xxx ETORO_USER_KEY=xxx ANTHROPIC_API_KEY=xxx
fly deploy
```

### Option C: AWS / GCP / Azure (Docker)
```bash
docker build -t etoro-agent .
docker run -d \
  -e ETORO_API_KEY=xxx \
  -e ETORO_USER_KEY=xxx \
  -e ANTHROPIC_API_KEY=xxx \
  -p 8000:8000 \
  etoro-agent
```

---

## How the AI decides

Every cycle (default: hourly), Claude receives:
- Your current portfolio (cash, open positions, P&L)
- Real-time market data for 10 tracked stocks
- Current timestamp

Claude then picks from 4 strategies:
| Strategy | When used |
|---|---|
| `momentum` | Strong upward trends, breakouts |
| `mean_reversion` | Oversold stocks bouncing back |
| `defensive` | High volatility, uncertain markets |
| `hold` | No clear edge — preserves cash |

### Risk guardrails (hardcoded)
- **Max 10%** of portfolio per stock
- **Min 15% cash buffer** always maintained
- **Min trade size**: $50
- Trades only from a defined watchlist
- Demo mode available for testing

---

## Customisation

Edit `agent/trading_agent.py`:
```python
WATCHLIST = ["AAPL", "MSFT", ...]  # Change which stocks to watch
MAX_POSITION_PCT = 0.10            # Max allocation per stock (10%)
MIN_TRADE_AMOUNT = 50              # Minimum trade in USD
RUN_INTERVAL_SECS = 3600          # How often to run (seconds)
```

To change the AI's strategy prompt, edit `SYSTEM_PROMPT` in `trading_agent.py`.

---

## Testing (Demo Mode)

eToro provides demo endpoints. Change the base URL in `trading_agent.py`:
```python
ETORO_BASE = "https://public-api.etoro.com/api/v1/trading/execution/demo"
```

This lets you run the full agent without risking real money.

---

## ⚠️ Risk Disclaimer

This software is for educational and personal use. Automated trading carries significant financial risk. Past performance does not guarantee future results. Always start with small amounts and monitor the agent closely.
