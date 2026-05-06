"""
eToro AI Trading Agent
Dynamically selects strategy based on market conditions using Claude AI.
"""

import os
import json
import uuid
import time
import logging
from datetime import datetime, timezone
from typing import Optional
import requests
import anthropic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("agent.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
ETORO_BASE      = "https://public-api.etoro.com/api/v1"
ETORO_API_KEY   = os.environ["ETORO_API_KEY"]    # x-api-key
ETORO_USER_KEY  = os.environ["ETORO_USER_KEY"]   # x-user-key
ANTHROPIC_KEY   = os.environ["ANTHROPIC_API_KEY"]

# Trading limits – edit to match your risk appetite
MAX_POSITION_PCT   = 0.10   # max 10% of portfolio per stock
MIN_TRADE_AMOUNT   = 50     # USD
RUN_INTERVAL_SECS  = 3600   # run every hour
WATCHLIST          = [      # tickers to track
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "TSLA", "META", "JPM", "V", "UNH"
]

# ── eToro API helpers ────────────────────────────────────────────────────────

def _headers() -> dict:
    return {
        "x-api-key":    ETORO_API_KEY,
        "x-user-key":   ETORO_USER_KEY,
        "x-request-id": str(uuid.uuid4()),
        "Content-Type": "application/json",
    }


def etoro_get(path: str) -> dict:
    url = f"{ETORO_BASE}{path}"
    r = requests.get(url, headers=_headers(), timeout=15)
    r.raise_for_status()
    return r.json()


def etoro_post(path: str, body: dict) -> dict:
    url = f"{ETORO_BASE}{path}"
    r = requests.post(url, headers=_headers(), json=body, timeout=15)
    r.raise_for_status()
    return r.json()


# ── Portfolio helpers ────────────────────────────────────────────────────────

def get_portfolio() -> dict:
    """Return current positions and cash balance."""
    return etoro_get("/portfolio")


def get_available_cash(portfolio: dict) -> float:
    """Calculate free cash from portfolio data."""
    return float(portfolio.get("availableCash", 0))


def get_open_positions(portfolio: dict) -> list[dict]:
    """Extract open equity positions."""
    return [
        p for p in portfolio.get("positions", [])
        if p.get("isBuy") and p.get("positionType") == "Stock"
    ]


def get_instrument_id(ticker: str) -> Optional[int]:
    """Resolve a ticker symbol to an eToro instrumentId."""
    try:
        data = etoro_get(f"/market-data/search?internalSymbolFull={ticker}")
        instruments = data.get("instruments", [])
        if instruments:
            return instruments[0]["instrumentId"]
    except Exception as e:
        log.warning(f"Could not resolve {ticker}: {e}")
    return None


def get_market_data(ticker: str) -> dict:
    """Fetch recent price/rate data for a ticker."""
    try:
        iid = get_instrument_id(ticker)
        if not iid:
            return {}
        return etoro_get(f"/market-data/instruments/{iid}/rates")
    except Exception as e:
        log.warning(f"Market data error for {ticker}: {e}")
        return {}


# ── Trade execution ─────────────────────────────────────────────────────────

def open_position(instrument_id: int, amount_usd: float, is_buy: bool = True) -> dict:
    """Open a market order by dollar amount."""
    body = {
        "InstrumentId": instrument_id,
        "Amount": round(amount_usd, 2),
        "Leverage": 1,
        "IsBuy": is_buy,
    }
    log.info(f"{'BUY' if is_buy else 'SELL'} {amount_usd:.2f} USD of instrument {instrument_id}")
    return etoro_post("/trading/execution/market-open-orders/by-amount", body)


def close_position(position_id: int, units: Optional[float] = None) -> dict:
    """Close a position fully or partially."""
    body = {"UnitsToDeduct": units}  # None = full close
    log.info(f"CLOSE position {position_id}" + (f" ({units} units)" if units else " (full)"))
    return etoro_post(f"/trading/execution/market-close-orders/positions/{position_id}", body)


# ── AI decision engine ───────────────────────────────────────────────────────

def build_market_snapshot(portfolio: dict) -> dict:
    """Gather market data for the watchlist."""
    snapshot = {}
    for ticker in WATCHLIST:
        data = get_market_data(ticker)
        if data:
            snapshot[ticker] = {
                "lastPrice":  data.get("lastPrice"),
                "change1d":   data.get("dailyChange"),
                "change1w":   data.get("weeklyChange"),
                "change1m":   data.get("monthlyChange"),
                "high52w":    data.get("high52Week"),
                "low52w":     data.get("low52Week"),
            }
    return snapshot


SYSTEM_PROMPT = """You are an expert AI portfolio manager running inside an eToro Agent Portfolio.
Your goal is to manage a stock portfolio dynamically, picking the best strategy for current conditions.

You will receive:
1. Current portfolio state (cash, open positions, equity)
2. Real-time market data for a watchlist of stocks
3. Today's date and time

Your response MUST be valid JSON with this structure:
{
  "strategy": "momentum|mean_reversion|defensive|hold",
  "rationale": "Brief explanation of chosen strategy and market read",
  "actions": [
    {
      "action": "buy|sell|close",
      "ticker": "AAPL",
      "amount_usd": 500,
      "reason": "Short reason"
    }
  ],
  "risk_level": "low|medium|high",
  "next_review": "1h|4h|24h"
}

Rules:
- Never allocate more than 10% of portfolio equity to a single stock
- Always keep at least 15% in cash as a buffer
- Only trade stocks in the provided watchlist
- For sells/closes, reference ticker symbols only (positions will be looked up)
- If no action is warranted, return an empty actions array
- Be concise but specific in your reasoning
"""


def ask_claude(portfolio: dict, market_data: dict) -> dict:
    """Send portfolio state to Claude and get back trading decisions."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    positions = get_open_positions(portfolio)
    cash      = get_available_cash(portfolio)
    equity    = float(portfolio.get("totalEquity", cash))

    user_msg = f"""
Current time: {datetime.now(timezone.utc).isoformat()}

=== PORTFOLIO STATE ===
Available Cash: ${cash:,.2f}
Total Equity:   ${equity:,.2f}
Cash Buffer:    {(cash/equity*100):.1f}%

Open Positions ({len(positions)}):
{json.dumps(positions, indent=2)}

=== MARKET DATA ===
{json.dumps(market_data, indent=2)}

Based on this data, decide what actions to take. Remember to stay within the risk rules.
Return ONLY valid JSON.
"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}]
    )

    raw = response.content[0].text.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ── Action executor ──────────────────────────────────────────────────────────

def execute_actions(decisions: dict, portfolio: dict, equity: float) -> list[dict]:
    """Execute the trades Claude decided on."""
    results = []
    positions_by_ticker = {
        p.get("instrumentName", "").upper(): p
        for p in get_open_positions(portfolio)
    }

    for action in decisions.get("actions", []):
        ticker     = action["action"].upper() if "ticker" not in action else action["ticker"].upper()
        act_type   = action["action"].lower()
        amount_usd = action.get("amount_usd", 0)
        ticker     = action.get("ticker", "").upper()

        try:
            if act_type == "buy":
                # Safety: cap at MAX_POSITION_PCT of equity
                max_allowed = equity * MAX_POSITION_PCT
                safe_amount = min(amount_usd, max_allowed)
                if safe_amount < MIN_TRADE_AMOUNT:
                    log.info(f"Skipping BUY {ticker}: amount too small (${safe_amount:.2f})")
                    continue
                iid = get_instrument_id(ticker)
                if not iid:
                    log.warning(f"Skipping BUY {ticker}: could not resolve instrument ID")
                    continue
                result = open_position(iid, safe_amount, is_buy=True)
                results.append({"action": "buy", "ticker": ticker, "amount": safe_amount, "result": result})

            elif act_type in ("sell", "close"):
                pos = positions_by_ticker.get(ticker)
                if not pos:
                    log.warning(f"Cannot close {ticker}: no open position found")
                    continue
                result = close_position(pos["positionId"])
                results.append({"action": "close", "ticker": ticker, "result": result})

        except requests.HTTPError as e:
            log.error(f"Trade failed for {ticker}: {e.response.text}")
            results.append({"action": act_type, "ticker": ticker, "error": str(e)})
        except Exception as e:
            log.error(f"Unexpected error for {ticker}: {e}")
            results.append({"action": act_type, "ticker": ticker, "error": str(e)})

    return results


# ── Run loop ─────────────────────────────────────────────────────────────────

def run_cycle():
    log.info("=" * 60)
    log.info("Starting trading cycle")

    try:
        portfolio    = get_portfolio()
        equity       = float(portfolio.get("totalEquity", 0))
        market_data  = build_market_snapshot(portfolio)

        log.info(f"Portfolio equity: ${equity:,.2f} | Watching {len(market_data)} instruments")

        decisions = ask_claude(portfolio, market_data)
        log.info(f"Strategy: {decisions.get('strategy')} | Risk: {decisions.get('risk_level')}")
        log.info(f"Rationale: {decisions.get('rationale')}")
        log.info(f"Actions planned: {len(decisions.get('actions', []))}")

        results = execute_actions(decisions, portfolio, equity)

        # Save cycle log
        cycle_log = {
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "equity":     equity,
            "decisions":  decisions,
            "results":    results,
        }
        with open("cycle_log.jsonl", "a") as f:
            f.write(json.dumps(cycle_log) + "\n")

        log.info(f"Cycle complete. {len(results)} trades executed.")
        return cycle_log

    except Exception as e:
        log.error(f"Cycle failed: {e}", exc_info=True)
        return {"error": str(e)}


def main():
    log.info("eToro AI Trading Agent starting...")
    while True:
        run_cycle()
        log.info(f"Sleeping {RUN_INTERVAL_SECS}s until next cycle...")
        time.sleep(RUN_INTERVAL_SECS)


if __name__ == "__main__":
    main()
