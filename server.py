"""
eToro Agent API Server
Serves the dashboard and exposes API endpoints.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import json, os, threading, time
from datetime import datetime, timezone
from typing import Optional
from trading_agent import run_cycle, get_portfolio, get_available_cash, get_open_positions

app = FastAPI(title="eToro AI Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── State ─────────────────────────────────────────────────────────────────────
agent_state = {
    "running":      False,
    "last_cycle":   None,
    "cycle_count":  0,
    "status":       "idle",
}

# ── Models ────────────────────────────────────────────────────────────────────
class ManualTradeRequest(BaseModel):
    ticker: str
    action: str
    amount_usd: Optional[float] = None


# ── Dashboard ─────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def dashboard():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    try:
        with open(html_path) as f:
            content = f.read()
        content = content.replace(
            "API_URL = 'http://localhost:8000'",
            "API_URL = window.location.origin"
        )
        return HTMLResponse(content=content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Dashboard not found</h1>", status_code=404)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


@app.get("/agent/status")
def agent_status():
    from trading_agent import is_market_hours
    return {
        **agent_state,
        "uptime":       "available" if agent_state["running"] else "stopped",
        "market_open":  is_market_hours(),
        "cycle_interval": "15 min" if is_market_hours() else "60 min",
    }


@app.post("/agent/start")
def start_agent(background_tasks: BackgroundTasks):
    if agent_state["running"]:
        raise HTTPException(400, "Agent already running")
    agent_state["running"] = True
    agent_state["status"]  = "running"
    background_tasks.add_task(_run_agent_loop)
    return {"message": "Agent started"}


@app.post("/agent/stop")
def stop_agent():
    agent_state["running"] = False
    agent_state["status"]  = "idle"
    return {"message": "Agent stopping"}


@app.post("/agent/run-now")
def run_now(background_tasks: BackgroundTasks):
    background_tasks.add_task(_single_cycle)
    return {"message": "Cycle triggered"}


@app.get("/portfolio")
def portfolio():
    # Reverse map: instrumentID → ticker
    ID_TO_TICKER = {
        1001: 'AAPL', 1003: 'META', 1004: 'MSFT', 1005: 'AMZN',
        1023: 'JPM',  1032: 'UNH',  1046: 'V',    1111: 'TSLA',
        1137: 'NVDA', 6434: 'GOOGL'
    }
    try:
        data      = get_portfolio()
        cash      = get_available_cash(data)
        raw_pos   = get_open_positions(data)

        # Normalise positions to what the dashboard expects
        positions = []
        for p in raw_pos:
            iid     = p.get("instrumentID") or p.get("instrumentId")
            ticker  = ID_TO_TICKER.get(iid, f"ID:{iid}")
            pl      = p.get("unrealizedPnL", {}).get("pnL", 0)
            invested = float(p.get("amount", 0))
            positions.append({
                "positionId":     p.get("positionID"),
                "instrumentName": ticker,
                "instrumentId":   iid,
                "invested":       invested,
                "profit":         pl,
                "units":          p.get("units", 0),
                "openRate":       p.get("openRate"),
                "isBuy":          p.get("isBuy"),
            })

        equity = float(data.get("credit", 0)) + sum(
            float(p.get("amount", 0)) + float(p.get("unrealizedPnL", {}).get("pnL", 0))
            for p in raw_pos
        )
        return {
            "cash":          cash,
            "totalEquity":   equity,
            "positions":     positions,
            "positionCount": len(positions),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/cycles")
def get_cycles(limit: int = 20):
    cycles = []
    try:
        with open("cycle_log.jsonl") as f:
            for line in f:
                line = line.strip()
                if line:
                    cycles.append(json.loads(line))
        return {"cycles": cycles[-limit:], "total": len(cycles)}
    except FileNotFoundError:
        return {"cycles": [], "total": 0}



@app.get("/debug/close/{position_id}")
def debug_close(position_id: int):
    """Test close endpoint and return full error detail from eToro."""
    import requests as req
    from trading_agent import _headers, ETORO_BASE
    url = f"{ETORO_BASE}/trading/execution/market-close-orders/positions/{position_id}"
    try:
        r = req.post(url, headers=_headers(), json={"UnitsToDeduct": None}, timeout=15)
        return {"status": r.status_code, "body": r.json()}
    except Exception as e:
        return {"error": str(e)}

@app.post("/trade/close/{position_id}")
def close_by_id(position_id: int):
    """Close a specific position by its positionId directly."""
    from trading_agent import close_position
    try:
        result = close_position(position_id)
        return {"message": f"Closed position {position_id}", "result": result}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/trade/manual")
def manual_trade(req: ManualTradeRequest):
    from trading_agent import get_instrument_id, open_position, close_position, WATCHLIST
    ID_TO_TICKER = {v: k for k, v in WATCHLIST.items()}
    TICKER_TO_ID = WATCHLIST

    if req.action == "buy":
        if not req.amount_usd:
            raise HTTPException(400, "amount_usd required for buy")
        iid = TICKER_TO_ID.get(req.ticker.upper()) or get_instrument_id(req.ticker)
        if not iid:
            raise HTTPException(404, f"Instrument not found: {req.ticker}")
        result = open_position(iid, req.amount_usd)
        return {"message": f"Bought ${req.amount_usd} of {req.ticker}", "result": result}

    elif req.action == "close":
        # Fetch normalised positions (already resolved to tickers)
        port = portfolio()
        norm_positions = port["positions"]
        target = req.ticker.upper()
        # Find first matching position
        pos = next((p for p in norm_positions if p["instrumentName"] == target), None)
        if not pos:
            raise HTTPException(404, f"No open position for {target}")
        position_id = pos["positionId"]
        result = close_position(position_id)
        return {"message": f"Closed {target} position {position_id}", "result": result}

    raise HTTPException(400, f"Unknown action: {req.action}")


# ── Debug endpoint ────────────────────────────────────────────────────────────
@app.get("/debug/market")
def debug_market():
    """Diagnose instrument resolution and market data."""
    from trading_agent import resolve_watchlist_ids, get_market_data, etoro_get, is_market_hours
    from datetime import datetime, timezone
    results = {
        "utc_time":     datetime.now(timezone.utc).isoformat(),
        "market_hours": is_market_hours(),
    }

    # Test raw rates endpoint with known IDs
    try:
        ids_str = "1001,1003,1004,1005,1023,1032,1046,1111,1137,6434"
        raw_rates = etoro_get("/market-data/instruments/rates", params={"instrumentIds": ids_str})
        results["rates_raw"] = raw_rates
    except Exception as e:
        results["rates_error"] = str(e)

    # Test candle endpoint with daily interval (works outside hours)
    try:
        daily = etoro_get("/market-data/instruments/1001/history/candles/desc/OneDay/2")
        results["aapl_daily_candles"] = daily
    except Exception as e:
        results["aapl_daily_error"] = str(e)

    # Resolved IDs
    try:
        ids = resolve_watchlist_ids()
        results["instrument_ids"] = ids
    except Exception as e:
        results["ids_error"] = str(e)

    # Full market data via our function
    try:
        ids = resolve_watchlist_ids()
        md = get_market_data(ids)
        results["market_data"] = md
        results["market_data_count"] = len(md)
    except Exception as e:
        results["market_data_error"] = str(e)

    return results


# ── Background tasks ──────────────────────────────────────────────────────────
def _run_agent_loop():
    from trading_agent import (
        RUN_INTERVAL_MARKET, RUN_INTERVAL_OFFHOURS,
        is_market_hours, resolve_watchlist_ids,
        get_market_data, check_price_alerts
    )
    last_prices: dict = {}

    while agent_state["running"]:
        _single_cycle()
        if not agent_state["running"]:
            break

        in_market = is_market_hours()
        interval  = RUN_INTERVAL_MARKET if in_market else RUN_INTERVAL_OFFHOURS

        # Update last known prices
        result = agent_state.get("last_cycle") or {}
        for ticker, data in (result.get("market_data") or {}).items():
            price = data.get("lastPrice")
            if price:
                last_prices[ticker] = price

        # Sleep in 60s chunks, checking for price alerts during market hours
        slept = 0
        while slept < interval and agent_state["running"]:
            time.sleep(60)
            slept += 60

            if in_market and last_prices:
                try:
                    ids    = resolve_watchlist_ids()
                    md     = get_market_data(ids)
                    alerts = check_price_alerts(md, last_prices)
                    if alerts:
                        break  # Trigger early cycle
                except Exception:
                    pass


def _single_cycle():
    agent_state["status"] = "running"
    result = run_cycle()
    agent_state["last_cycle"]  = result
    agent_state["cycle_count"] += 1
    agent_state["status"] = "idle" if agent_state["running"] else "stopped"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
