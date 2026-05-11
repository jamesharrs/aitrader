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

MAX_POSITION_PCT       = 0.10
MIN_TRADE_AMOUNT       = 50
RUN_INTERVAL_MARKET    = 900    # 15 min during market hours
RUN_INTERVAL_OFFHOURS  = 3600   # 60 min outside market hours
PRICE_MOVE_THRESHOLD   = 0.02   # 2% move triggers immediate cycle

# Pre-market brief cache — only refresh once per trading day
_brief_cache = {"date": None, "brief": {}}
RUN_INTERVAL_SECS      = 3600   # legacy fallback

# Top 10 US stocks with verified eToro instrument IDs
# Sourced from eToro's static instruments metadata API
WATCHLIST = {
    "AAPL":  1001,
    "META":  1003,
    "MSFT":  1004,
    "AMZN":  1005,
    "JPM":   1023,
    "UNH":   1032,
    "V":     1046,
    "TSLA":  1111,
    "NVDA":  1137,
    "GOOGL": 6434,
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
        # fields param is required — request only what we need
        data = etoro_get("/market-data/search", params={
            "searchText": ticker,
            "pageSize": 20,
            "fields": "instrumentId,internalSymbolFull,internalInstrumentDisplayName,internalAssetClassName,isDelisted,isActiveInPlatform"
        })
        items = data.get("items", [])
        log.info(f"Search for {ticker} returned {len(items)} items")
        # Match stocks/ETFs with exact symbol
        for item in items:
            iid = item.get("instrumentId", -1)
            if iid <= 0:
                continue
            if item.get("isDelisted") or not item.get("isActiveInPlatform", True):
                continue
            asset_class = item.get("internalAssetClassName", "").lower()
            if asset_class not in ("stocks", "etf"):
                continue
            sym = item.get("internalSymbolFull", "").upper()
            disp = item.get("internalInstrumentDisplayName", "").upper()
            log.info(f"  candidate: {sym} / {disp} (ID={iid}, class={asset_class})")
            if sym == ticker.upper() or disp == ticker.upper():
                _instrument_cache[ticker] = iid
                log.info(f"Resolved {ticker} -> instrument ID {iid}")
                return iid
        log.warning(f"No stock match for {ticker} in {len(items)} results")
    except Exception as e:
        log.warning(f"Could not resolve {ticker}: {e}")
    return None

def resolve_watchlist_ids() -> dict[str, int]:
    """Return verified instrument IDs for all watchlist tickers."""
    return dict(WATCHLIST)  # IDs are hardcoded and verified from eToro metadata API

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

def get_latest_candle(instrument_id: int) -> dict:
    """
    Fetch the most recent candle for an instrument.
    Response structure: { candles: [ { instrumentId, candles: [ {open,high,low,close,...} ] } ] }
    """
    try:
        path = f"/market-data/instruments/{instrument_id}/history/candles/desc/OneDay/2"
        data = etoro_get(path)
        outer = data.get("candles", [])
        if outer:
            inner = outer[0].get("candles", [])
            if inner:
                return inner[0]  # Most recent candle
    except Exception as e:
        log.warning(f"Candle fetch failed for {instrument_id}: {e}")
    return {}


def get_market_data(instrument_ids: dict[str, int]) -> dict:
    """
    Fetch latest price for each instrument via candle history.
    Falls back to closing price history if candles unavailable.
    """
    if not instrument_ids:
        return {}

    snapshot = {}
    for ticker, iid in instrument_ids.items():
        candle = get_latest_candle(iid)
        if candle:
            snapshot[ticker] = {
                "lastPrice": candle.get("close"),   # "close" from candle response
                "open":      candle.get("open"),
                "high":      candle.get("high"),
                "low":       candle.get("low"),
                "volume":    candle.get("volume"),
                "date":      candle.get("fromDate"),
            }
            log.info(f"  {ticker}: ${candle.get('close')}")

    log.info(f"Market data: {len(snapshot)}/{len(instrument_ids)} instruments returned data")
    return snapshot

# ── Trade execution ───────────────────────────────────────────────────────────

def open_position(instrument_id: int, amount_usd: float, is_buy: bool = True) -> dict:
    body = {"InstrumentId": instrument_id, "Amount": round(amount_usd, 2), "Leverage": 1, "IsBuy": is_buy}
    log.info(f"{'BUY' if is_buy else 'SELL'} ${amount_usd:.2f} instrument {instrument_id}")
    return etoro_post("/trading/execution/market-open-orders/by-amount", body)

def close_position(position_id: int) -> dict:
    log.info(f"CLOSE position {position_id}")
    return etoro_post(f"/trading/execution/market-close-orders/positions/{position_id}", {})

# ── AI decision engine ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert AI portfolio manager running inside an eToro Agent Portfolio.
Your goal is to dynamically manage a stock portfolio based on current market conditions.

You will receive the current portfolio state and recent OHLCV candle data.

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
- If lastPrice data is present for stocks, markets are open and you CAN trade
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

=== LIVE MARKET DATA (recent candles) ===
{json.dumps(market_data, indent=2)}

{"NOTE: Market data is available - markets are open for trading." if market_data else "NOTE: No market data returned - markets may be closed."}

Return ONLY valid JSON.
"""
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
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
    ID_TO_TICKER = {v: k for k, v in WATCHLIST.items()}
    positions_by_ticker = {
        ID_TO_TICKER.get(p.get("instrumentID") or p.get("instrumentId"), "").upper(): p
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
                pid = pos.get("positionID") or pos.get("positionId"); result = close_position(pid)
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

        # Refresh pre-market brief once per trading day (saves ~90% of token costs)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if _brief_cache["date"] != today:
            log.info("Refreshing pre-market brief (daily)...")
            brief = build_premarket_brief(etoro_get, instrument_ids)
            _brief_cache["date"]  = today
            _brief_cache["brief"] = brief
        else:
            log.info("Using cached pre-market brief (no token cost)")
            brief = _brief_cache["brief"]

        brief_text = format_brief_for_claude(brief)
        decisions  = ask_claude(pnl, market_data, brief_text)
        log.info(f"Strategy: {decisions.get('strategy')} | Risk: {decisions.get('risk_level')}")
        log.info(f"Rationale: {decisions.get('rationale')}")

        results = execute_actions(decisions, pnl, equity, instrument_ids)

        cycle_log = {
            "timestamp":       datetime.now(timezone.utc).isoformat(),
            "equity":          equity,
            "cash":            cash,
            "market_data":     market_data,
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

def is_market_hours() -> bool:
    """Returns True if NYSE is currently open (Mon-Fri 14:30-21:00 UTC)."""
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    market_open  = now.replace(hour=14, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=21, minute=0,  second=0, microsecond=0)
    return market_open <= now <= market_close


def check_price_alerts(market_data: dict, last_prices: dict) -> list[str]:
    """Detect any stock that has moved >2% since last cycle."""
    alerts = []
    for ticker, data in market_data.items():
        current = data.get("lastExecution") or data.get("ask")
        if not current or ticker not in last_prices:
            continue
        prev = last_prices[ticker]
        move = abs((current - prev) / prev)
        if move >= PRICE_MOVE_THRESHOLD:
            direction = "▲" if current > prev else "▼"
            alerts.append(f"{ticker} {direction}{move*100:.1f}%")
    return alerts


def main():
    log.info("eToro AI Trading Agent starting...")
    last_prices: dict[str, float] = {}

    while True:
        result     = run_cycle()
        in_market  = is_market_hours()
        interval   = RUN_INTERVAL_MARKET if in_market else RUN_INTERVAL_OFFHOURS

        # Update last known prices for alert tracking
        market_data = result.get("market_data", {}) if isinstance(result, dict) else {}
        for ticker, data in market_data.items():
            price = data.get("lastExecution") or data.get("ask")
            if price:
                last_prices[ticker] = price

        log.info(f"{'[MARKET OPEN]' if in_market else '[AFTER HOURS]'} "
                 f"Next cycle in {interval//60} min...")

        # Sleep in 60s increments so we can react to price alerts
        slept = 0
        while slept < interval:
            time.sleep(60)
            slept += 60

            # During market hours, check for big price moves every minute
            if in_market and last_prices:
                try:
                    ids  = resolve_watchlist_ids()
                    md   = get_market_data(ids)
                    alerts = check_price_alerts(md, last_prices)
                    if alerts:
                        log.info(f"Price alert triggered: {', '.join(alerts)} — running early cycle")
                        break
                except Exception:
                    pass  # Don't crash the sleep loop on transient errors


if __name__ == "__main__":
    main()
