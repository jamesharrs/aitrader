"""
eToro AI Trading Agent - corrected API endpoints
"""

import os
import json
import uuid
import time
import logging
import sqlite3
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

MAX_POSITION_PCT       = 0.08   # hard cap: no single position > 8% of equity
MAX_TICKER_PCT         = 0.25   # hard cap: no single ticker > 25% of equity across all positions
MIN_CASH_PCT           = 0.20   # hard floor: always keep 20% cash (raised from 15%)
MIN_TRADE_AMOUNT       = 50
STOP_LOSS_PCT          = 0.05   # auto-close any position down >5%
MAX_BUYS_PER_CYCLE     = 1      # max 1 buy per cycle — prevents cash blowout
RUN_INTERVAL_MARKET    = 900    # 15 min during market hours
RUN_INTERVAL_OFFHOURS  = 3600   # 60 min outside market hours
PRICE_MOVE_THRESHOLD   = 0.02   # 2% move triggers immediate cycle
BRIEF_REFRESH_THRESHOLD = 0.03  # 3% intraday move triggers mid-session brief refresh

# Conviction-based sizing — reduced to preserve cash buffer
CONVICTION_SIZE = {"high": 0.05, "medium": 0.03, "low": 0.02}  # % of equity

# Pre-market brief cache — only refresh once per trading day
def _load_brief_cache() -> dict:
    try:
        with open("brief_cache.json") as f:
            return json.load(f)
    except Exception:
        return {"date": None, "brief": {}}

def _save_brief_cache(date: str, brief: dict):
    try:
        with open("brief_cache.json", "w") as f:
            json.dump({"date": date, "brief": brief}, f)
    except Exception as e:
        log.warning(f"Could not save brief cache: {e}")
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
    """
    Equity = cash + net value of all positions.
    Net value = units × openRate + profit (as reported by eToro API).
    We use amount + profit since amount = cost basis of current units.
    """
    credit   = float(pnl.get("credit", 0))
    net_pos  = sum(
        float(p.get("amount", 0)) + float(p.get("profit", 0))
        for p in pnl.get("positions", [])
    )
    return credit + net_pos

# ── Market data ───────────────────────────────────────────────────────────────

def get_candle_history(instrument_id: int, count: int = 6) -> list[dict]:
    """
    Fetch last N daily candles for an instrument (oldest → newest).
    Response: { candles: [ { instrumentId, candles: [...] } ] }
    """
    try:
        path = f"/market-data/instruments/{instrument_id}/history/candles/desc/OneDay/{count}"
        data = etoro_get(path)
        outer = data.get("candles", [])
        if outer:
            inner = outer[0].get("candles", [])
            # API returns desc order; reverse to get oldest → newest
            return list(reversed(inner))
    except Exception as e:
        log.warning(f"Candle history fetch failed for {instrument_id}: {e}")
    return []


def get_market_data(instrument_ids: dict[str, int]) -> dict:
    """
    Fetch 5-day rolling price context for each instrument.
    Returns lastPrice, 5-day close history, simple trend direction, and volume signal.
    """
    if not instrument_ids:
        return {}

    snapshot = {}
    for ticker, iid in instrument_ids.items():
        candles = get_candle_history(iid, count=6)
        if candles:
            latest = candles[-1]
            closes = [c.get("close") for c in candles if c.get("close")]

            # Simple 5-day trend: compare last close to 5-day-ago close
            pct_change_5d = None
            if len(closes) >= 2:
                pct_change_5d = round((closes[-1] - closes[0]) / closes[0] * 100, 2)

            # Volume signal: today vs average of prior days
            volumes = [c.get("volume", 0) for c in candles]
            avg_vol = sum(volumes[:-1]) / max(len(volumes) - 1, 1) if len(volumes) > 1 else None
            vol_ratio = round(volumes[-1] / avg_vol, 2) if avg_vol and avg_vol > 0 else None

            snapshot[ticker] = {
                "lastPrice":    latest.get("close"),
                "open":         latest.get("open"),
                "high":         latest.get("high"),
                "low":          latest.get("low"),
                "volume":       latest.get("volume"),
                "date":         latest.get("fromDate"),
                "closes_5d":    closes,          # [oldest … newest] — shows trend
                "pct_change_5d": pct_change_5d,  # positive = uptrend
                "volume_ratio": vol_ratio,        # >1.2 = high volume, <0.8 = low volume
            }
            log.info(f"  {ticker}: ${latest.get('close')} | 5d: {pct_change_5d}% | vol_ratio: {vol_ratio}")

    log.info(f"Market data: {len(snapshot)}/{len(instrument_ids)} instruments returned data")
    return snapshot

# ── Trade execution ───────────────────────────────────────────────────────────

def open_position(instrument_id: int, amount_usd: float, is_buy: bool = True) -> dict:
    body = {"InstrumentId": instrument_id, "Amount": round(amount_usd, 2), "Leverage": 1, "IsBuy": is_buy}
    log.info(f"{'BUY' if is_buy else 'SELL'} ${amount_usd:.2f} instrument {instrument_id}")
    return etoro_post("/trading/execution/market-open-orders/by-amount", body)

def close_position(position_id: int, instrument_id: int = None) -> dict:
    log.info(f"CLOSE position {position_id} instrument {instrument_id}")
    body = {"InstrumentId": instrument_id, "UnitsToDeduct": None}
    return etoro_post(f"/trading/execution/market-close-orders/positions/{position_id}", body)

# ── AI decision engine ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert AI portfolio manager running inside an eToro Agent Portfolio.
Your goal is to dynamically manage a stock portfolio to generate returns above the S&P 500.

You receive: portfolio state, 5-day rolling OHLCV data per stock, and a pre-market briefing.

Your response MUST be valid JSON:
{
  "strategy": "momentum|mean_reversion|defensive|hold",
  "rationale": "Specific explanation citing price action and market context",
  "actions": [
    {
      "action": "buy|close",
      "ticker": "AAPL",
      "amount_usd": 500,
      "confidence": "high|medium|low",
      "reason": "Specific reason citing data"
    }
  ],
  "risk_level": "low|medium|high",
  "next_review": "1h|4h|24h",
  "hold_cash_reason": "Only populate if actions is empty — specific signal you are waiting for"
}

HARD RULES (enforced in code — violations are blocked automatically):
- Max 8% of equity per single position
- Max 25% of equity in any one ticker across all positions combined
- Min 20% cash buffer at all times — DO NOT recommend buys if cash is near this floor
- Stop-losses at -5% are auto-executed before you are called — you will not see those positions
- Max 1 buy per cycle — recommend only your single highest-conviction buy
- Min trade size $50

DECISION FRAMEWORK:
- Use closes_5d to identify trend: rising = uptrend, falling = downtrend
- pct_change_5d > +3%: momentum candidate (buy if volume_ratio > 1.0)
- pct_change_5d < -3%: mean-reversion candidate (buy only if no fundamental reason for decline)
- volume_ratio > 1.2 with rising price = confirmed momentum; with falling price = distribution (avoid/close)
- Prefer CLOSING losing or stale positions over buying new ones when cash is below 30%
- Do NOT re-buy a ticker you just closed in the same session unless conditions have materially changed

IMPORTANT — Cash discipline:
The portfolio has previously blown through its cash buffer by over-trading. 
Preserving 20%+ cash is a primary objective — it enables you to act on real opportunities.
If cash is below 25%, your default should be to CLOSE a weak position rather than buy.
Only recommend a buy when you have a genuinely strong signal AND sufficient cash headroom.
"""


def ask_claude(pnl: dict, market_data: dict, premarket_brief: str = "") -> dict:
    client    = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    cash      = get_available_cash(pnl)
    equity    = get_total_equity(pnl)
    positions = get_open_positions(pnl)

    # Summarise positions for the prompt — include P&L context
    pos_summary = []
    for p in positions:
        profit   = float(p.get("profit", 0))
        amount   = float(p.get("amount", 0))   # cost basis of current units
        net_val  = amount + profit
        pct      = (profit / amount * 100) if amount else 0
        pos_summary.append({
            "ticker":     p.get("instrumentName", "?"),
            "cost_basis": round(amount, 2),
            "net_value":  round(net_val, 2),
            "profit":     round(profit, 2),
            "pct":        round(pct, 2),
            "units":      p.get("units"),
            "openRate":   p.get("openRate"),
        })

    user_msg = f"""
Current time (UTC): {datetime.now(timezone.utc).isoformat()}

{premarket_brief}

=== PORTFOLIO STATE ===
Available Cash: ${cash:,.2f}
Total Equity:   ${equity:,.2f}
Cash Buffer:    {(cash/equity*100 if equity else 0):.1f}%
Open Positions ({len(positions)}):
{json.dumps(pos_summary, indent=2)}

=== MARKET DATA (5-day rolling context) ===
Fields: lastPrice, open, high, low, closes_5d (oldest→newest), pct_change_5d (5d%), volume_ratio (vs avg)
{json.dumps(market_data, indent=2)}

{"NOTE: Market data present — markets are open. You can execute trades now." if market_data else "NOTE: No market data — markets may be closed."}

Analyse the trend data carefully before deciding. Return ONLY valid JSON.
"""
    response = client.messages.create(
        model="claude-sonnet-4-6",
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

    # Track per-ticker invested amount for concentration cap
    ticker_invested = {}
    for p in get_open_positions(pnl):
        t = ID_TO_TICKER.get(p.get("instrumentID") or p.get("instrumentId"), "").upper()
        ticker_invested[t] = ticker_invested.get(t, 0) + float(p.get("amount", 0))

    # Re-fetch live cash so we can check buffer before each buy
    live_cash = get_available_cash(pnl)
    buys_this_cycle = 0

    # Process closes first — they free up cash before any buys
    for action in sorted(decisions.get("actions", []), key=lambda a: 0 if a["action"] in ("close","sell") else 1):
        ticker   = action.get("ticker", "").upper()
        act_type = action["action"].lower()
        amount_usd = action.get("amount_usd", 0)

        try:
            if act_type == "buy":
                # ── Fix 4: cap buys at 1 per cycle ──
                if buys_this_cycle >= MAX_BUYS_PER_CYCLE:
                    log.info(f"Skipping BUY {ticker}: already bought {MAX_BUYS_PER_CYCLE} this cycle")
                    continue

                # ── Fix 5: conviction-based sizing (reduced) ──
                confidence = action.get("confidence", "medium")
                size_pct = CONVICTION_SIZE.get(confidence, 0.03)
                conviction_cap = equity * size_pct
                safe_amount = min(amount_usd or conviction_cap, equity * MAX_POSITION_PCT, conviction_cap)

                if safe_amount < MIN_TRADE_AMOUNT:
                    log.info(f"Skipping BUY {ticker}: ${safe_amount:.2f} below minimum")
                    continue

                # ── Fix 1: pre-buy cash buffer check ──
                projected_cash = live_cash - safe_amount
                if equity > 0 and (projected_cash / equity) < MIN_CASH_PCT:
                    log.info(f"Skipping BUY {ticker}: would leave cash at {projected_cash/equity:.1%} (min {MIN_CASH_PCT:.0%})")
                    continue

                # ── Fix 2: per-ticker concentration cap ──
                already_invested = ticker_invested.get(ticker, 0)
                if equity > 0 and (already_invested + safe_amount) / equity > MAX_TICKER_PCT:
                    log.info(f"Skipping BUY {ticker}: concentration would reach {(already_invested+safe_amount)/equity:.1%} (max {MAX_TICKER_PCT:.0%})")
                    continue

                iid = instrument_ids.get(ticker) or get_instrument_id(ticker)
                if not iid:
                    log.warning(f"Skipping BUY {ticker}: no instrument ID")
                    continue

                result = open_position(iid, safe_amount)
                results.append({"action": "buy", "ticker": ticker, "amount": safe_amount, "confidence": confidence, "result": result})
                live_cash -= safe_amount
                ticker_invested[ticker] = already_invested + safe_amount
                buys_this_cycle += 1

            elif act_type in ("sell", "close"):
                pos = positions_by_ticker.get(ticker)
                if not pos:
                    log.warning(f"Cannot close {ticker}: no open position")
                    continue
                pid = pos.get("positionID") or pos.get("positionId")
                iid = pos.get("instrumentID") or pos.get("instrumentId") or WATCHLIST.get(ticker)
                result = close_position(pid, iid)
                freed = float(pos.get("amount", 0)) + float(pos.get("profit", 0))
                live_cash += freed
                results.append({"action": "close", "ticker": ticker, "result": result})

        except requests.HTTPError as e:
            log.error(f"Trade failed {ticker}: {e.response.text}")
            results.append({"action": act_type, "ticker": ticker, "error": str(e)})
        except Exception as e:
            log.error(f"Error {ticker}: {e}")
            results.append({"action": act_type, "ticker": ticker, "error": str(e)})

    return results


# ── Stop-loss enforcement ─────────────────────────────────────────────────────

def enforce_stop_losses(pnl: dict) -> list[dict]:
    """
    Auto-close any position that is down more than STOP_LOSS_PCT.
    Runs before Claude is called so the portfolio state Claude sees is clean.
    """
    ID_TO_TICKER = {v: k for k, v in WATCHLIST.items()}
    closed = []
    for p in get_open_positions(pnl):
        profit   = float(p.get("profit", 0))
        invested = float(p.get("amount", 0))
        if invested <= 0:
            continue
        pct = profit / invested
        if pct <= -STOP_LOSS_PCT:
            ticker = ID_TO_TICKER.get(p.get("instrumentID") or p.get("instrumentId"), "?")
            pid    = p.get("positionID") or p.get("positionId")
            iid    = p.get("instrumentID") or p.get("instrumentId") or WATCHLIST.get(ticker)
            log.warning(f"STOP-LOSS: closing {ticker} at {pct:.1%} (${profit:+.2f})")
            try:
                close_position(pid, iid)
                closed.append({"ticker": ticker, "pct": pct, "profit": profit})
            except Exception as e:
                log.error(f"Stop-loss close failed for {ticker}: {e}")
    return closed



def init_decision_db():
    """Create SQLite table for tracking decision accuracy."""
    conn = sqlite3.connect("decisions.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT,
            ticker      TEXT,
            action      TEXT,
            confidence  TEXT,
            price_at_decision REAL,
            reason      TEXT,
            strategy    TEXT,
            next_price  REAL,       -- filled in on following cycle
            pct_outcome REAL        -- positive = good call
        )
    """)
    conn.commit()
    conn.close()

def record_decisions(decisions: dict, market_data: dict):
    """Log each trade decision with entry price for future accuracy scoring."""
    conn = sqlite3.connect("decisions.db")
    ts = datetime.now(timezone.utc).isoformat()
    strategy = decisions.get("strategy", "hold")
    for action in decisions.get("actions", []):
        ticker = action.get("ticker", "").upper()
        price = (market_data.get(ticker) or {}).get("lastPrice")
        conn.execute("""
            INSERT INTO decisions (timestamp, ticker, action, confidence, price_at_decision, reason, strategy)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (ts, ticker, action.get("action"), action.get("confidence"), price, action.get("reason"), strategy))
    conn.commit()
    conn.close()

def update_decision_outcomes(market_data: dict):
    """Fill in next_price and pct_outcome for decisions that are missing them."""
    conn = sqlite3.connect("decisions.db")
    rows = conn.execute(
        "SELECT id, ticker, price_at_decision FROM decisions WHERE next_price IS NULL AND price_at_decision IS NOT NULL"
    ).fetchall()
    for row_id, ticker, entry_price in rows:
        current = (market_data.get(ticker) or {}).get("lastPrice")
        if current and entry_price:
            pct = round((current - entry_price) / entry_price * 100, 2)
            conn.execute("UPDATE decisions SET next_price=?, pct_outcome=? WHERE id=?", (current, pct, row_id))
    conn.commit()
    conn.close()

# ── Run cycle ─────────────────────────────────────────────────────────────────

def run_cycle():
    log.info("=" * 60)
    log.info("Starting trading cycle")
    try:
        init_decision_db()
        pnl            = get_portfolio()
        equity         = get_total_equity(pnl)
        cash           = get_available_cash(pnl)
        instrument_ids = resolve_watchlist_ids()
        market_data    = get_market_data(instrument_ids)

        log.info(f"Equity: ${equity:,.2f} | Cash: ${cash:,.2f} | Market data: {len(market_data)} instruments")

        # ── Fix 3: enforce stop-losses before Claude sees the portfolio ──
        stop_loss_closes = enforce_stop_losses(pnl)
        if stop_loss_closes:
            log.info(f"Stop-losses fired: {stop_loss_closes}")
            # Re-fetch portfolio so Claude sees the cleaned state
            pnl  = get_portfolio()
            cash = get_available_cash(pnl)

        # Update outcomes for any prior decisions now that new prices are in
        if market_data:
            update_decision_outcomes(market_data)

        # Daily brief — with mid-session refresh if any stock moved >3% since morning
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        _cache = _load_brief_cache()

        needs_refresh = _cache["date"] != today
        if not needs_refresh and market_data:
            # Check if any stock has moved significantly since the brief was generated
            morning_prices = _cache.get("morning_prices", {})
            for ticker, data in market_data.items():
                current = data.get("lastPrice") or 0
                morning = morning_prices.get(ticker) or 0
                if morning and current and abs((current - morning) / morning) >= BRIEF_REFRESH_THRESHOLD:
                    log.info(f"Mid-session brief refresh triggered by {ticker} move")
                    needs_refresh = True
                    break

        if needs_refresh:
            log.info("Refreshing pre-market brief...")
            brief = build_premarket_brief(etoro_get, instrument_ids)
            morning_prices = {t: d.get("lastPrice") for t, d in market_data.items()}
            _save_brief_cache(today, brief)
            # Store morning prices alongside cache for mid-session drift detection
            try:
                with open("brief_cache.json") as f:
                    cached = json.load(f)
                cached["morning_prices"] = morning_prices
                with open("brief_cache.json", "w") as f:
                    json.dump(cached, f)
            except Exception:
                pass
        else:
            log.info("Using cached pre-market brief (no token cost)")
            brief = _cache["brief"]

        brief_text = format_brief_for_claude(brief)
        decisions  = ask_claude(pnl, market_data, brief_text)
        log.info(f"Strategy: {decisions.get('strategy')} | Risk: {decisions.get('risk_level')}")
        log.info(f"Rationale: {decisions.get('rationale')}")
        if not decisions.get("actions") and decisions.get("hold_cash_reason"):
            log.info(f"Hold reason: {decisions.get('hold_cash_reason')}")

        # Record decisions with entry prices for accuracy tracking
        record_decisions(decisions, market_data)

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
