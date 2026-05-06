"""
eToro Agent API Server
Exposes endpoints for the dashboard to poll and for manual control.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
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

# ── State ────────────────────────────────────────────────────────────────────
agent_state = {
    "running":      False,
    "last_cycle":   None,
    "cycle_count":  0,
    "status":       "idle",   # idle | running | paused | error
}
_cycle_thread: Optional[threading.Thread] = None


# ── Models ───────────────────────────────────────────────────────────────────
class ManualTradeRequest(BaseModel):
    ticker: str
    action: str       # buy | close
    amount_usd: Optional[float] = None


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


@app.get("/agent/status")
def agent_status():
    return {
        **agent_state,
        "uptime": "available" if agent_state["running"] else "stopped",
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
    return {"message": "Agent stopping after current cycle"}


@app.post("/agent/run-now")
def run_now(background_tasks: BackgroundTasks):
    """Trigger an immediate cycle."""
    background_tasks.add_task(_single_cycle)
    return {"message": "Cycle triggered"}


@app.get("/portfolio")
def portfolio():
    try:
        data      = get_portfolio()
        cash      = get_available_cash(data)
        positions = get_open_positions(data)
        return {
            "cash":          cash,
            "totalEquity":   data.get("totalEquity"),
            "positions":     positions,
            "positionCount": len(positions),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/cycles")
def get_cycles(limit: int = 20):
    """Return recent cycle logs."""
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


@app.post("/trade/manual")
def manual_trade(req: ManualTradeRequest):
    """Execute a manual trade outside of the agent cycle."""
    from trading_agent import (
        get_instrument_id, open_position, close_position, get_portfolio, get_open_positions
    )
    portfolio = get_portfolio()
    if req.action == "buy":
        if not req.amount_usd:
            raise HTTPException(400, "amount_usd required for buy")
        iid = get_instrument_id(req.ticker)
        if not iid:
            raise HTTPException(404, f"Instrument not found: {req.ticker}")
        result = open_position(iid, req.amount_usd)
        return {"message": f"Bought ${req.amount_usd} of {req.ticker}", "result": result}

    elif req.action == "close":
        positions = get_open_positions(portfolio)
        pos = next((p for p in positions if req.ticker.upper() in str(p.get("instrumentName","")).upper()), None)
        if not pos:
            raise HTTPException(404, f"No open position for {req.ticker}")
        result = close_position(pos["positionId"])
        return {"message": f"Closed {req.ticker} position", "result": result}

    raise HTTPException(400, f"Unknown action: {req.action}")


# ── Background tasks ─────────────────────────────────────────────────────────

def _run_agent_loop():
    from trading_agent import RUN_INTERVAL_SECS
    while agent_state["running"]:
        _single_cycle()
        if agent_state["running"]:
            time.sleep(RUN_INTERVAL_SECS)


def _single_cycle():
    agent_state["status"] = "running"
    result = run_cycle()
    agent_state["last_cycle"]  = result
    agent_state["cycle_count"] += 1
    agent_state["status"] = "idle" if agent_state["running"] else "stopped"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
