"""
Pre-Market Intelligence Module
Gathers news, sentiment, and price action context before each trading cycle.
"""

import os
import logging
import requests
from datetime import datetime, timezone, timedelta
from typing import Optional
import anthropic

log = logging.getLogger(__name__)

ANTHROPIC_KEY = os.environ["ANTHROPIC_API_KEY"]

# ── News & Sentiment via web search ──────────────────────────────────────────

def fetch_market_news() -> list[dict]:
    """
    Use Claude with web search to pull today's pre-market news and sentiment.
    Returns structured headlines + sentiment per ticker.
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    today  = datetime.now(timezone.utc).strftime("%B %d, %Y")

    prompt = f"""Today is {today}. Search for the latest pre-market news and sentiment for these US stocks:
AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META, JPM, V, UNH

Also search for:
- Overall US market pre-market sentiment today
- Any major macro events (Fed, CPI, earnings, geopolitical)
- S&P 500 and Nasdaq futures direction

Return ONLY valid JSON in this exact structure:
{{
  "market_overview": {{
    "futures_sentiment": "bullish|bearish|neutral",
    "sp500_futures_change_pct": 0.0,
    "nasdaq_futures_change_pct": 0.0,
    "key_macro_events": ["event1", "event2"],
    "overall_summary": "2-3 sentence market overview"
  }},
  "stocks": {{
    "AAPL": {{
      "sentiment": "bullish|bearish|neutral",
      "premarket_change_pct": 0.0,
      "key_headlines": ["headline1", "headline2"],
      "catalyst": "earnings|upgrade|downgrade|news|none",
      "analyst_consensus": "buy|hold|sell|mixed"
    }}
  }},
  "trading_approach": {{
    "recommended_strategy": "momentum|mean_reversion|defensive|hold",
    "confidence": "high|medium|low",
    "sectors_to_favour": ["tech", "finance"],
    "sectors_to_avoid": [],
    "key_risks": ["risk1", "risk2"],
    "daily_thesis": "2-3 sentence thesis for today's trading session"
  }}
}}

Include all 10 stocks. Use real data from your search. Return ONLY the JSON, no other text."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": prompt}]
        )

        # Extract text from response (may include tool use blocks)
        raw = ""
        for block in response.content:
            if hasattr(block, "text"):
                raw += block.text

        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        import json
        return json.loads(raw)

    except Exception as e:
        log.warning(f"Pre-market news fetch failed: {e}")
        return {}


def get_candle_data(etoro_get_fn, instrument_id: int, interval: str = "OneHour", count: int = 24) -> list[dict]:
    """
    Fetch recent OHLCV candles for an instrument.
    Useful for pre-market gap analysis and overnight price action.
    """
    try:
        path = f"/market-data/instruments/{instrument_id}/history/candles/desc/{interval}/{count}"
        data = etoro_get_fn(path)
        candles_data = data.get("candles", [])
        if candles_data:
            return candles_data[0].get("candles", [])
    except Exception as e:
        log.warning(f"Candle fetch failed for instrument {instrument_id}: {e}")
    return []


def analyse_price_action(candles: list[dict]) -> dict:
    """
    Derive simple technical signals from candle data.
    """
    if not candles or len(candles) < 5:
        return {}

    closes = [c["close"] for c in candles if "close" in c]
    highs  = [c["high"]  for c in candles if "high"  in c]
    lows   = [c["low"]   for c in candles if "low"   in c]

    if len(closes) < 5:
        return {}

    latest = closes[0]
    prev   = closes[1]
    high5  = max(highs[:5])
    low5   = min(lows[:5])

    # Simple moving averages
    ma5  = sum(closes[:5])  / 5
    ma20 = sum(closes[:20]) / 20 if len(closes) >= 20 else None

    # RSI (simplified 14-period)
    gains  = [max(closes[i] - closes[i+1], 0) for i in range(min(14, len(closes)-1))]
    losses = [max(closes[i+1] - closes[i], 0) for i in range(min(14, len(closes)-1))]
    avg_gain = sum(gains)  / len(gains)  if gains  else 0
    avg_loss = sum(losses) / len(losses) if losses else 1
    rs  = avg_gain / avg_loss if avg_loss else 0
    rsi = 100 - (100 / (1 + rs))

    return {
        "latest_price":    round(latest, 4),
        "prev_price":      round(prev,   4),
        "change_pct":      round((latest - prev) / prev * 100, 2) if prev else 0,
        "high_5periods":   round(high5, 4),
        "low_5periods":    round(low5,  4),
        "ma5":             round(ma5,   4),
        "ma20":            round(ma20,  4) if ma20 else None,
        "rsi_14":          round(rsi,   1),
        "trend":           "up" if ma5 > (ma20 or ma5) else "down",
        "overbought":      rsi > 70,
        "oversold":        rsi < 30,
    }


def build_premarket_brief(etoro_get_fn, instrument_ids: dict[str, int]) -> dict:
    """
    Main entry point — builds the full pre-market intelligence brief.
    Combines news/sentiment + technical price action for each stock.
    """
    log.info("Building pre-market intelligence brief...")

    # 1. Fetch news and sentiment via web search
    news_data = fetch_market_news()

    # 2. Fetch overnight price action for each stock
    technicals = {}
    for ticker, iid in instrument_ids.items():
        candles = get_candle_data(etoro_get_fn, iid, interval="OneHour", count=30)
        if candles:
            technicals[ticker] = analyse_price_action(candles)

    # 3. Combine into a single brief
    brief = {
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "news_sentiment": news_data,
        "technicals":     technicals,
    }

    log.info(f"Pre-market brief ready. News: {'OK' if news_data else 'unavailable'}, "
             f"Technicals: {len(technicals)} stocks")

    return brief


def format_brief_for_claude(brief: dict) -> str:
    """
    Format the pre-market brief as a readable string for the Claude trading prompt.
    """
    import json
    lines = []

    news = brief.get("news_sentiment", {})
    overview = news.get("market_overview", {})
    if overview:
        lines.append("=== PRE-MARKET INTELLIGENCE ===")
        lines.append(f"Futures: S&P {overview.get('sp500_futures_change_pct', '?')}% | "
                     f"Nasdaq {overview.get('nasdaq_futures_change_pct', '?')}%")
        lines.append(f"Market Sentiment: {overview.get('futures_sentiment', 'unknown').upper()}")
        lines.append(f"Summary: {overview.get('overall_summary', '')}")
        macros = overview.get("key_macro_events", [])
        if macros:
            lines.append(f"Macro Events: {', '.join(macros)}")

    approach = news.get("trading_approach", {})
    if approach:
        lines.append("")
        lines.append("=== SUGGESTED APPROACH ===")
        lines.append(f"Recommended Strategy: {approach.get('recommended_strategy', '?').upper()}")
        lines.append(f"Confidence: {approach.get('confidence', '?').upper()}")
        lines.append(f"Daily Thesis: {approach.get('daily_thesis', '')}")
        risks = approach.get("key_risks", [])
        if risks:
            lines.append(f"Key Risks: {', '.join(risks)}")

    stocks = news.get("stocks", {})
    if stocks:
        lines.append("")
        lines.append("=== STOCK SENTIMENT ===")
        for ticker, data in stocks.items():
            sentiment  = data.get("sentiment", "?")
            change     = data.get("premarket_change_pct", 0)
            catalyst   = data.get("catalyst", "none")
            headlines  = data.get("key_headlines", [])
            change_str = f"+{change:.1f}%" if change > 0 else f"{change:.1f}%"
            lines.append(f"{ticker}: {sentiment.upper()} {change_str} | {catalyst} | {headlines[0] if headlines else 'No news'}")

    techs = brief.get("technicals", {})
    if techs:
        lines.append("")
        lines.append("=== TECHNICAL SIGNALS ===")
        for ticker, t in techs.items():
            rsi     = t.get("rsi_14", "?")
            trend   = t.get("trend", "?")
            chg     = t.get("change_pct", 0)
            ob_os   = "OVERBOUGHT" if t.get("overbought") else ("OVERSOLD" if t.get("oversold") else "neutral")
            chg_str = f"+{chg:.1f}%" if chg > 0 else f"{chg:.1f}%"
            lines.append(f"{ticker}: RSI {rsi} | {trend.upper()} trend | {chg_str} | {ob_os}")

    return "\n".join(lines)
