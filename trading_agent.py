"""
eToro AI Trading Agent - corrected API endpoints
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
from premarket import build_premarket_brief, format_brief_for_claude

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("agent.log"), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
ETORO_BASE      = "https://public-api.etoro.com/api/v1"
ETORO_API_KEY   = os.environ["ETORO_API_KEY"]
ETORO_USER_KEY  = os.environ["ETORO_USER_KEY"]
ANTHROPIC_KEY   = os.environ["ANTHROPIC_API_KEY"]

MAX_POSITION_PCT  = 0.10
MIN_TRADE_AMOUNT  = 50
RUN_INTERVAL_SECS = 3600

# Top 10 US stocks + their known eToro instrument IDs
# IDs sourced from eToro's instrument search API
WATCHLIST = {
    "AAPL":  1001,
    "MSFT":  1002,
    "GOOGL": 1003,
    "AMZN":  1004,
    "NVDA":  1005,
    "TSLA":  1006,
    "META":  1007,
    "JPM":   1008,
    "V":     1009,
    "UNH":   1010,
}

# ── eToro API helpers ─────────────────────────────────────────────────────────

def _headers() -> dict:
    return {
        "x-api-key":    ETORO_API_KEY,
        "x-user-key":   ETORO_USER_KEY,
        "x-request-id": str(uuid.uuid4()),
        "Content-Type": "application/json",
    }

def etoro_get(path: str, params: dict = None) -> dict:
    r = requests.get(f"{ETORO_BASE}{path}", headers=_headers(), params=params, timeout=15)
    r.raise_for_status()
    return r.json()

def etoro_post(path: str, body: dict) -> dict:
    r = requests.post(f"{ETORO_BASE}{path}", headers=_headers(), json=body, timeout=15)
    r.raise_for_status()
    return r.json()

# ── Instrument ID resolution ──────────────────────────────────────────────────

_instrument_cache: dict[str, int] = {}

def get_instrument_id(ticker: str) -> Optional[int]:
    """Resolve ticker to eToro instrument ID via search API."""
    if ticker in _instrument_cache:
        return _instrument_cache[ticker]
    try:
        data = etoro_get("/market-data/search", params={
            "searchText": ticker,
            "pageSize": 20
        })
        items = data.get("items", [])
        # Match only stock/ETF assets with exact symbol — skip forex, crypto etc
        for item in items:
            iid = item.get("instrumentId", -1)
            if iid <= 0:
                continue
            asset_class = item.get("internalAssetClassName", "").lower()
            if asset_class not in ("stocks", "etf"):
                continue
            sym = item.get("internalSymbolFull", "").upper()
            name = item.get("internalInstrumentDisplayName", "").upper()
            if sym == ticker.upper() or name == ticker.upper():
                _instrument_cache[ticker] = iid
                log.info(f"Resolved {ticker} -> instrument ID {iid} ({asset_class})")
                return iid
        log.warning(f"No stock match found for {ticker}")
    except Exception as e:
        log.warning(f"Could not resolve {ticker}: {e}")
    return None

def resolve_watchlist_ids() -> dict[str, int]:
    """Resolve all watchlist tickers to instrument IDs."""
    resolved = {}
    for ticker in WATCHLIST:
        iid = get_instrument_id(ticker)
        if iid:
            resolved[ticker] = iid
    return resolved

# ── Portfolio helpers ─────────────────────────────────────────────────────────

def get_portfolio() -> dict:
    """Fetch real account P&L."""
    data = etoro_get("/trading/info/real/pnl")
    return data.get("clientPortfolio", data)

def get_available_cash(pnl: dict) -> float:
    credit = float(pnl.get("credit", 0))
    manual_pending = sum(
        float(o.get("amount", 0))
        for o in pnl.get("ordersForOpen", [])
        if o.get("mirrorId", 0) == 0
    )
    limit_orders = sum(float(o.get("amount", 0)) for o in pnl.get("orders", []))
    return credit - (manual_pending + limit_orders)

def get_open_positions(pnl: dict) -> list[dict]:
    return [p for p in pnl.get("positions", []) if p.get("isBuy") is True]

def get_total_equity(pnl: dict) -> float:
    credit   = float(pnl.get("credit", 0))
    invested = sum(float(p.get("amount", 0)) for p in pnl.get("positions", []))
    pl       = sum(float(p.get("profit", 0)) for p in pnl.get("positions", []))
    return credit + invested + pl

# ── Market data ───────────────────────────────────────────────────────────────

def get_market_data(instrument_ids: dict[str, int]) -> dict:
    """
    Fetch live rates for all instruments in one batch call.
    Endpoint: GET /market-data/instruments/rates?instrumentIds=1,2,3
    """
    if not instrument_ids:
        return {}
    try:
        ids_str = ",".join(str(v) for v in instrument_ids.values())
        data = etoro_get("/market-data/instruments/rates", params={"instrumentIds": ids_str})
        rates_by_id = {r["instrumentID"]: r for r in data.get("rates", [])}

        snapshot = {}
        for ticker, iid in instrument_ids.items():
            rate = rates_by_id.get(iid)
            if rate:
                snapshot[ticker] = {
                    "bid":           rate.get("bid"),
                    "ask":           rate.get("ask"),
                    "lastExecution": rate.get("lastExecution"),
                    "date":          rate.get("date"),
                }
        return snapshot
    except Exception as e:
        log.warning(f"Market data fetch failed: {e}")
        return {}

# ── Trade execution ───────────────────────────────────────────────────────────

def open_position(instrument_id: int, amount_usd: float, is_buy: bool = True) -> dict:
    body = {"InstrumentId": instrument_id, "Amount": round(amount_usd, 2), "Leverage": 1, "IsBuy": is_buy}
    log.info(f"{'BUY' if is_buy else 'SELL'} ${amount_usd:.2f} instrument {instrument_id}")
    return etoro_post("/trading/execution/real/market-open-orders/by-amount", body)

def close_position(position_id: int) -> dict:
    log.info(f"CLOSE position {position_id}")
    return etoro_post(f"/trading/execution/real/market-close-orders/positions/{position_id}", {})

# ── AI decision engine ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert AI portfolio manager running inside an eToro Agent Portfolio.
Your goal is to dynamically manage a stock portfolio based on current market conditions.

You will receive the current portfolio state and live market bid/ask prices.

Your response MUST be valid JSON:
{
  "strategy": "momentum|mean_reversion|defensive|hold",
  "rationale": "Brief explanation",
  "actions": [
    {"action": "buy|close", "ticker": "AAPL", "amount_usd": 500, "reason": "Short reason"}
  ],
  "risk_level": "low|medium|high",
  "next_review": "1h|4h|24h"
}

Rules:
- Never allocate more than 10% of equity to a single stock
- Always keep at least 15% in cash
- Only trade tickers in the provided market data
- If bid/ask data is present, markets are open - you CAN trade
- If no action warranted, return empty actions array
"""

def ask_claude(pnl: dict, market_data: dict, premarket_brief: str = "") -> dict:
    client    = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    cash      = get_available_cash(pnl)
    equity    = get_total_equity(pnl)
    positions = get_open_positions(pnl)

    user_msg = f"""
Current time (UTC): {datetime.now(timezone.utc).isoformat()}

{premarket_brief}

=== PORTFOLIO ===
Available Cash: ${cash:,.2f}
Total Equity:   ${equity:,.2f}
Cash Buffer:    {(cash/equity*100 if equity else 0):.1f}%
Open Positions: {len(positions)}
{json.dumps(positions, indent=2)}

=== LIVE MARKET DATA (bid/ask prices) ===
{json.dumps(market_data, indent=2)}

{"NOTE: Market data is available - markets are open for trading." if market_data else "NOTE: No market data returned - markets may be closed."}

Return ONLY valid JSON.
"""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}]
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())

# ── Action executor ───────────────────────────────────────────────────────────

def execute_actions(decisions: dict, pnl: dict, equity: float, instrument_ids: dict) -> list[dict]:
    results = []
    positions_by_ticker = {
        p.get("instrumentName", "").upper(): p
        for p in get_open_positions(pnl)
    }

    for action in decisions.get("actions", []):
        ticker    = action.get("ticker", "").upper()
        act_type  = action["action"].lower()
        amount_usd = action.get("amount_usd", 0)

        try:
            if act_type == "buy":
                safe_amount = min(amount_usd, equity * MAX_POSITION_PCT)
                if safe_amount < MIN_TRADE_AMOUNT:
                    log.info(f"Skipping BUY {ticker}: ${safe_amount:.2f} below minimum")
                    continue
                iid = instrument_ids.get(ticker) or get_instrument_id(ticker)
                if not iid:
                    log.warning(f"Skipping BUY {ticker}: no instrument ID")
                    continue
                result = open_position(iid, safe_amount)
                results.append({"action": "buy", "ticker": ticker, "amount": safe_amount, "result": result})

            elif act_type in ("sell", "close"):
                pos = positions_by_ticker.get(ticker)
                if not pos:
                    log.warning(f"Cannot close {ticker}: no open position")
                    continue
                result = close_position(pos["positionId"])
                results.append({"action": "close", "ticker": ticker, "result": result})

        except requests.HTTPError as e:
            log.error(f"Trade failed {ticker}: {e.response.text}")
            results.append({"action": act_type, "ticker": ticker, "error": str(e)})
        except Exception as e:
            log.error(f"Error {ticker}: {e}")
            results.append({"action": act_type, "ticker": ticker, "error": str(e)})

    return results

# ── Run cycle ─────────────────────────────────────────────────────────────────

def run_cycle():
    log.info("=" * 60)
    log.info("Starting trading cycle")
    try:
        pnl            = get_portfolio()
        equity         = get_total_equity(pnl)
        cash           = get_available_cash(pnl)
        instrument_ids = resolve_watchlist_ids()
        market_data    = get_market_data(instrument_ids)

        log.info(f"Equity: ${equity:,.2f} | Cash: ${cash:,.2f} | Market data: {len(market_data)} instruments")

        brief          = build_premarket_brief(etoro_get, instrument_ids)
        brief_text     = format_brief_for_claude(brief)
        decisions      = ask_claude(pnl, market_data, brief_text)
        log.info(f"Strategy: {decisions.get('strategy')} | Risk: {decisions.get('risk_level')}")
        log.info(f"Rationale: {decisions.get('rationale')}")

        results = execute_actions(decisions, pnl, equity, instrument_ids)

        cycle_log = {
            "timestamp":       datetime.now(timezone.utc).isoformat(),
            "equity":          equity,
            "cash":            cash,
            "premarket_brief": brief.get("news_sentiment", {}),
            "decisions":       decisions,
            "results":         results,
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
        log.info(f"Sleeping {RUN_INTERVAL_SECS}s...")
        time.sleep(RUN_INTERVAL_SECS)

if __name__ == "__main__":
    main()
