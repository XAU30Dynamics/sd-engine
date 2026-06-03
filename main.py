"""
AlgoTrader VPS Backtesting Server
----------------------------------
FastAPI + vectorbt + pandas-ta
Runs on port 8000. Start with:
    python -m uvicorn main:app --host 0.0.0.0 --port 8000

Windows: python -m uvicorn main:app --host 0.0.0.0 --port 8000
Linux:   uvicorn main:app --host 0.0.0.0 --port 8000

Endpoints:
  POST /backtest              — fixed indicator-based backtest (legacy)
  POST /backtest-custom       — AI-generated Python signal code backtest
  POST /walk-forward          — walk-forward analysis (rolling in/out of sample windows)
  POST /monte-carlo           — Monte Carlo simulation on trade results
  POST /optimize              — parameter optimisation (synchronous, kept for compat)
  POST /optimize-async        — submit async optimisation job, returns job_id immediately
  GET  /optimize-status/{id}  — poll async optimisation job status + progress
  DELETE /optimize/{id}       — cancel a running async optimisation job
  POST /update                — download latest engine from GitHub and restart
  GET  /health                — health check
"""

import os
import re
import time
import threading
import json
import urllib.request
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Literal, Any
import pandas as pd
import pandas_ta as ta
import numpy as np
import vectorbt as vbt
from datetime import datetime
import math
import traceback
import uuid
import types

app = FastAPI(title="AlgoTrader Backtest Engine", version="3.4.0")

# Read allowed origins from ALLOWED_ORIGINS env var (comma-separated) or use safe defaults.
# Set ALLOWED_ORIGINS="https://myapp.com,https://api.myapp.com" on the server to restrict access.
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "")
_allowed_origins = (
    [o.strip() for o in _raw_origins.split(",") if o.strip()]
    if _raw_origins else [
        "http://localhost:3000",
        "http://localhost:8081",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8081",
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# LRU-style bar cache — keyed by cache_key string
# Stores up to 20 datasets in memory (evict oldest when full)
_bar_cache: OrderedDict[str, list] = OrderedDict()
BAR_CACHE_MAX_SIZE = 20

# Parallel workers for optimisation — capped at 8 to avoid memory pressure
MAX_WORKERS = min(os.cpu_count() or 4, 8)


# ─── Async Job Store ────────────────────────────────────────────────────────────
# Each job is stored in memory AND on disk so results survive a VPS restart.

JOBS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "opt_jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

# job_id -> { status, tested, total, progress, result, error, cancel_event }
_opt_jobs: dict = {}


def _save_job_disk(job_id: str) -> None:
    """Persist job state to disk (excludes the threading Event)."""
    job = _opt_jobs.get(job_id)
    if not job:
        return
    try:
        payload = {
            "job_id": job_id,
            "status": job["status"],
            "testedCombinations": job["tested"],
            "totalCombinations": job["total"],
            "progressPct": job["progress"],
            "result": job["result"],
            "errorMessage": job["error"],
        }
        with open(os.path.join(JOBS_DIR, f"{job_id}.json"), "w") as f:
            json.dump(payload, f)
    except Exception as e:
        print(f"[jobs] Failed to save job {job_id}: {e}")


def _run_optimize_bg(job_id: str, req) -> None:
    """
    Background-thread entry point for async optimisation.
    Mirrors run_optimize() logic but:
      - checks cancel_event between every combination
      - writes progress to _opt_jobs and disk periodically
    """
    job = _opt_jobs[job_id]
    cancel_event: threading.Event = job["cancel_event"]
    started_at = datetime.utcnow()

    try:
        job["status"] = "running"
        _save_job_disk(job_id)

        import itertools
        keys = list(req.paramGrid.keys())
        values = list(req.paramGrid.values())
        all_combos = list(itertools.product(*values))

        total_grid_combinations = len(all_combos)
        was_capped = total_grid_combinations > req.maxCombinations
        sampling_method = "random_sample" if was_capped else "full_grid"

        if was_capped:
            rng_sample = np.random.default_rng(42)
            indices = sorted(rng_sample.choice(total_grid_combinations, size=req.maxCombinations, replace=False).tolist())
            all_combos = [all_combos[i] for i in indices]

        total = len(all_combos)
        job["total"] = total
        _save_job_disk(job_id)

        df = bars_to_df(req.bars)
        # Prefer the caller-supplied pip_size when provided (canonical source of truth).
        # Fall back to the engine's symbol-keyword lookup so older callers continue to work.
        pip_size = req.pip_size if (getattr(req, "pip_size", None) is not None) else (get_pip_size(req.symbol) if req.symbol else 0.0001)
        results = []
        tested_combinations = 0
        _start_time = time.time()

        ts_first = df['timestamp'].iloc[0] if len(df) > 0 else 'N/A'
        ts_last  = df['timestamp'].iloc[-1] if len(df) > 0 else 'N/A'
        print(f"[optimize-bg] job={job_id} bars={len(df)} date_range={ts_first} → {ts_last}")
        print(f"[optimize-bg] sequential combinations={total} pip_size={pip_size}")

        # Pre-compile strategy code once — compiled code objects are immutable and thread-safe.
        check_code_safety(req.pythonCode)
        compiled_code = compile(req.pythonCode, "<strategy>", "exec")
        print(f"[optimize-bg] Code pre-compiled successfully")

        # ── Walk-forward mode ─────────────────────────────────────────────────
        if req.walkForwardEnabled:
            n = len(df)
            window_size = n // req.nWindows
            if window_size < 30:
                job["status"] = "error"
                job["error"] = "Not enough bars per window. Reduce nWindows or provide more data."
                _save_job_disk(job_id)
                return

            wf_windows = []
            for w in range(req.nWindows):
                start_idx = w * window_size
                end_idx = start_idx + window_size if w < req.nWindows - 1 else n
                window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
                split_idx = int(len(window_df) * req.inSamplePct)
                oos_df = window_df.iloc[split_idx:].reset_index(drop=True)
                if len(oos_df) >= 10:
                    wf_windows.append(oos_df)

            if not wf_windows:
                job["status"] = "error"
                job["error"] = "No usable out-of-sample windows. Reduce nWindows or increase inSamplePct."
                _save_job_disk(job_id)
                return

            def _run_wf_combo(combo):
                if cancel_event.is_set():
                    return None
                params = dict(zip(keys, combo))
                oos_metrics_list = []
                for oos_df in wf_windows:
                    try:
                        module_ns = {
                            "__builtins__": _SAFE_BUILTINS,
                            "pd": pd, "ta": ta, "np": np, "vbt": vbt,
                        }
                        module_ns.update(params)
                        exec(compiled_code, module_ns)
                        if "get_signals" not in module_ns:
                            continue
                        signals = module_ns["get_signals"](oos_df)
                        el  = pd.Series(signals["entry_long"],  index=oos_df.index).fillna(False).astype(bool)
                        es  = pd.Series(signals["entry_short"], index=oos_df.index).fillna(False).astype(bool)
                        xl  = pd.Series(signals["exit_long"],   index=oos_df.index).fillna(False).astype(bool)
                        xs  = pd.Series(signals["exit_short"],  index=oos_df.index).fillna(False).astype(bool)
                        sl_s = pd.Series(signals["sl_price"], index=oos_df.index).astype(float) if signals.get("sl_price") is not None else None
                        tp_s = pd.Series(signals["tp_price"], index=oos_df.index).astype(float) if signals.get("tp_price") is not None else None
                        trades, equity_curve = simulate_trades(
                            df=oos_df, entry_long=el, entry_short=es, exit_long=xl, exit_short=xs,
                            stop_loss_pct=req.stopLossPercent, take_profit_pct=req.takeProfitPercent,
                            trade_direction=req.tradeDirection,
                            initial_capital=req.initialCapital,
                            risk_percent=req.riskPercent, risk_type=req.riskType, lot_size=req.lotSize,
                            stop_loss_pips=req.stopLossPips, take_profit_pips=req.takeProfitPips, pip_size=pip_size,
                            sl_price_series=sl_s, tp_price_series=tp_s,
                            execute_on_next_bar=req.executeOnNextBar,
                            spread_pips=req.spreadPips, slippage_pips=req.slippagePips,
                            contract_size=get_contract_size(req.symbol),
                        )
                        if trades:
                            oos_metrics_list.append(compute_metrics(trades, equity_curve, req.initialCapital))
                    except Exception:
                        continue
                if not oos_metrics_list:
                    return None
                def avg(key):
                    return sum(m[key] for m in oos_metrics_list) / len(oos_metrics_list)
                avg_metrics = {
                    "netProfitPercent": avg("netProfitPercent"),
                    "winRate": avg("winRate"),
                    "sharpeRatio": avg("sharpeRatio"),
                    "profitFactor": avg("profitFactor"),
                    "maxDrawdownPercent": avg("maxDrawdownPercent"),
                    "totalTrades": int(avg("totalTrades")),
                    "winningTrades": int(avg("winningTrades")),
                    "losingTrades": int(avg("losingTrades")),
                    "consecutiveWins": int(avg("consecutiveWins")),
                    "consecutiveLosses": int(avg("consecutiveLosses")),
                }
                score = compute_score(
                    avg_metrics["winRate"], avg_metrics["profitFactor"],
                    avg_metrics["maxDrawdownPercent"], avg_metrics["sharpeRatio"],
                    avg_metrics["totalTrades"], avg_metrics["netProfitPercent"]
                )
                return OptimizeResultRow(
                    params=params,
                    netProfitPercent=avg_metrics["netProfitPercent"],
                    winRate=avg_metrics["winRate"],
                    sharpeRatio=avg_metrics["sharpeRatio"],
                    profitFactor=avg_metrics["profitFactor"],
                    maxDrawdownPercent=avg_metrics["maxDrawdownPercent"],
                    totalTrades=avg_metrics["totalTrades"],
                    winningTrades=avg_metrics["winningTrades"],
                    losingTrades=avg_metrics["losingTrades"],
                    consecutiveWins=avg_metrics["consecutiveWins"],
                    consecutiveLosses=avg_metrics["consecutiveLosses"],
                    score=score,
                    grade=grade_from_score(score),
                )

            for combo in all_combos:
                if cancel_event.is_set():
                    job["status"] = "cancelled"
                    _save_job_disk(job_id)
                    return

                row = _run_wf_combo(combo)
                if row is not None:
                    results.append(row)
                tested_combinations += 1
                job["tested"] = tested_combinations
                job["progress"] = int(tested_combinations / total * 100)
                if tested_combinations % 10 == 0:
                    _save_job_disk(job_id)
                    elapsed = time.time() - _start_time
                    rate = tested_combinations / elapsed if elapsed > 0 else 0
                    eta = (total - tested_combinations) / rate if rate > 0 else 0
                    print(f"[optimize-bg] wf progress={tested_combinations}/{total} "
                          f"rate={rate:.1f} combos/sec elapsed={elapsed:.0f}s eta={eta:.0f}s")

        # ── Standard mode ─────────────────────────────────────────────────────
        else:
            for combo in all_combos:
                if cancel_event.is_set():
                    job["status"] = "cancelled"
                    _save_job_disk(job_id)
                    return

                params = dict(zip(keys, combo))
                try:
                    module_ns = {
                        "__builtins__": _SAFE_BUILTINS,
                        "pd": pd, "ta": ta, "np": np, "vbt": vbt,
                    }
                    module_ns.update(params)
                    exec(compiled_code, module_ns)
                    if "get_signals" not in module_ns:
                        tested_combinations += 1
                        job["tested"] = tested_combinations
                        job["progress"] = int(tested_combinations / total * 100)
                        continue
                    signals = module_ns["get_signals"](df)
                    el  = pd.Series(signals["entry_long"],  index=df.index).fillna(False).astype(bool)
                    es  = pd.Series(signals["entry_short"], index=df.index).fillna(False).astype(bool)
                    xl  = pd.Series(signals["exit_long"],   index=df.index).fillna(False).astype(bool)
                    xs  = pd.Series(signals["exit_short"],  index=df.index).fillna(False).astype(bool)
                    sl_s = pd.Series(signals["sl_price"], index=df.index).astype(float) if signals.get("sl_price") is not None else None
                    tp_s = pd.Series(signals["tp_price"], index=df.index).astype(float) if signals.get("tp_price") is not None else None
                    trades, equity_curve = simulate_trades(
                        df=df, entry_long=el, entry_short=es, exit_long=xl, exit_short=xs,
                        stop_loss_pct=req.stopLossPercent, take_profit_pct=req.takeProfitPercent,
                        trade_direction=req.tradeDirection,
                        initial_capital=req.initialCapital,
                        risk_percent=req.riskPercent, risk_type=req.riskType, lot_size=req.lotSize,
                        stop_loss_pips=req.stopLossPips, take_profit_pips=req.takeProfitPips, pip_size=pip_size,
                        sl_price_series=sl_s, tp_price_series=tp_s,
                        execute_on_next_bar=req.executeOnNextBar,
                        spread_pips=req.spreadPips, slippage_pips=req.slippagePips,
                        contract_size=get_contract_size(req.symbol),
                    )
                except Exception:
                    tested_combinations += 1
                    job["tested"] = tested_combinations
                    job["progress"] = int(tested_combinations / total * 100)
                    continue

                if not trades:
                    tested_combinations += 1
                    job["tested"] = tested_combinations
                    job["progress"] = int(tested_combinations / total * 100)
                    continue

                metrics = compute_metrics(trades, equity_curve, req.initialCapital)
                score = compute_score(
                    metrics["winRate"], metrics["profitFactor"],
                    metrics["maxDrawdownPercent"], metrics["sharpeRatio"],
                    metrics["totalTrades"], metrics["netProfitPercent"]
                )
                results.append(OptimizeResultRow(
                    params=params,
                    netProfitPercent=metrics["netProfitPercent"],
                    winRate=metrics["winRate"],
                    sharpeRatio=metrics["sharpeRatio"],
                    profitFactor=metrics["profitFactor"],
                    maxDrawdownPercent=metrics["maxDrawdownPercent"],
                    totalTrades=metrics["totalTrades"],
                    winningTrades=metrics["winningTrades"],
                    losingTrades=metrics["losingTrades"],
                    consecutiveWins=metrics["consecutiveWins"],
                    consecutiveLosses=metrics["consecutiveLosses"],
                    score=score,
                    grade=grade_from_score(score),
                ))

                tested_combinations += 1
                job["tested"] = tested_combinations
                job["progress"] = int(tested_combinations / total * 100)
                if tested_combinations % 10 == 0:
                    _save_job_disk(job_id)
                    elapsed = time.time() - _start_time
                    rate = tested_combinations / elapsed if elapsed > 0 else 0
                    eta = (total - tested_combinations) / rate if rate > 0 else 0
                    print(f"[optimize-bg] progress={tested_combinations}/{total} "
                          f"rate={rate:.1f} combos/sec elapsed={elapsed:.0f}s eta={eta:.0f}s")

        if not results:
            job["status"] = "error"
            job["error"] = "No valid parameter combinations produced trades."
            _save_job_disk(job_id)
            return

        # Sort by chosen metric
        def get_metric_bg(r, objective: str) -> float:
            return {
                "sharpe": r.sharpeRatio,
                "netProfit": r.netProfitPercent,
                "winRate": r.winRate,
                "profitFactor": r.profitFactor,
                "maxDrawdown": -r.maxDrawdownPercent,
                "returnDrawdown": (r.netProfitPercent / r.maxDrawdownPercent) if r.maxDrawdownPercent > 0 else (r.netProfitPercent if r.netProfitPercent > 0 else 0.0),
            }.get(objective, r.sharpeRatio)

        if req.secondaryObjective and req.secondaryObjective != req.optimizeFor:
            primary_vals = [get_metric_bg(r, req.optimizeFor) for r in results]
            secondary_vals = [get_metric_bg(r, req.secondaryObjective) for r in results]

            def normalise(vals):
                lo, hi = min(vals), max(vals)
                spread = hi - lo
                if spread == 0:
                    return [0.5] * len(vals)
                return [(v - lo) / spread for v in vals]

            p_norm = normalise(primary_vals)
            s_norm = normalise(secondary_vals)
            blended = [0.70 * p + 0.30 * s for p, s in zip(p_norm, s_norm)]
            results = [r for _, r in sorted(zip(blended, results), key=lambda x: x[0], reverse=True)]
        else:
            results.sort(key=lambda r: get_metric_bg(r, req.optimizeFor), reverse=True)

        best = results[0]
        completed_at = datetime.utcnow()
        result_obj = OptimizeResult(
            id=generate_id(),
            strategyId=req.strategyId,
            strategyName=req.strategyName,
            datasetName=req.datasetName,
            nCombinations=len(results),
            optimizeFor=req.optimizeFor,
            bestParams=best.params,
            bestResult=best,
            allResults=results,
            totalGridCombinations=total_grid_combinations,
            testedCombinations=tested_combinations,
            wasCapped=was_capped,
            samplingMethod=sampling_method,
            createdAt=datetime.utcnow().isoformat(),
            startedAt=started_at.isoformat(),
            completedAt=completed_at.isoformat(),
            durationMs=round((completed_at - started_at).total_seconds() * 1000, 1),
            executeOnNextBar=req.executeOnNextBar,
            spreadPips=req.spreadPips,
            slippagePips=req.slippagePips,
            maxOpenPositions=req.maxOpenPositions,
        )

        job["status"] = "done"
        job["result"] = result_obj.model_dump()
        job["progress"] = 100
        job["tested"] = tested_combinations
        _save_job_disk(job_id)
        print(f"[optimize-async] Job {job_id} completed: {tested_combinations} combos, best={best.params}")

    except Exception as e:
        print(f"[optimize-async] Job {job_id} failed: {traceback.format_exc()}")
        job["status"] = "error"
        job["error"] = str(e)
        _save_job_disk(job_id)


# ─── Shared Models ──────────────────────────────────────────────────────────────

class OHLCBar(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None


class TradeResult(BaseModel):
    entryDate: str
    exitDate: str
    entryPrice: float
    exitPrice: float
    type: Literal["long", "short"]
    pnl: float
    pnlPercent: float
    positionSize: float


class BacktestResult(BaseModel):
    id: str
    strategyId: str
    datasetId: str
    strategyName: str
    datasetName: str
    netProfit: float
    netProfitPercent: float
    profitFactor: float
    maxDrawdownPercent: float
    winRate: float
    totalTrades: int
    winningTrades: int
    losingTrades: int
    consecutiveWins: int
    consecutiveLosses: int
    sharpeRatio: float
    grade: str
    score: int
    trades: List[TradeResult]
    equityCurve: List[float]
    riskPercent: float
    riskType: str
    lotSize: float
    stopLossPercent: float
    createdAt: str
    engineVersion: str = "python-3.0"
    startedAt: str = ""
    completedAt: str = ""
    durationMs: float = 0.0
    # Execution settings (populated by backtest-custom, walk-forward, optimize)
    executeOnNextBar: Optional[bool] = None
    spreadPips: Optional[float] = None
    slippagePips: Optional[float] = None
    maxOpenPositions: Optional[int] = None


# ─── Pip size helpers ──────────────────────────────────────────────────────────

def get_pip_size(symbol: str) -> float:
    """
    Returns the pip size for a given symbol.
    - JPY pairs (e.g. USDJPY, EURJPY): 1 pip = 0.01
    - Most forex pairs (e.g. EURUSD, GBPUSD): 1 pip = 0.0001
    - Indices / commodities with larger price scales: 1 pip = 1.0
    - Gold (XAUUSD): 1 pip = 0.1
    - Crypto (BTCUSD, ETHUSD, etc.): 1 pip = 1.0
    """
    s = symbol.upper().replace("/", "").replace("-", "").replace("_", "").replace(".", "")

    # JPY pairs
    if s.endswith("JPY") or s.startswith("JPY"):
        return 0.01

    # Gold
    if "XAU" in s or "GOLD" in s:
        return 0.1

    # Silver
    if "XAG" in s or "SILVER" in s:
        return 0.001

    # Crypto (common tickers)
    crypto_bases = ["BTC", "ETH", "LTC", "XRP", "ADA", "SOL", "DOT", "MATIC", "BNB", "AVAX", "LINK", "DOGE"]
    for c in crypto_bases:
        if s.startswith(c):
            return 1.0

    # Indices / US30 / NAS / SPX etc.
    index_keywords = ["US30", "NAS", "SPX", "DAX", "FTSE", "NDX", "GER", "UK100", "AUS200", "JPN225", "CN50"]
    for idx in index_keywords:
        if idx in s:
            return 1.0

    # Default: standard 4-decimal forex pair (EURUSD, GBPUSD, AUDUSD, etc.)
    return 0.0001


def get_contract_size(symbol: str) -> float:
    """
    Returns the contract size for a given symbol.
    - Standard forex (EURUSD, GBPUSD, AUDUSD, etc.): 100,000 units
    - JPY pairs (USDJPY, EURJPY, etc.): 100,000 units
    - Gold (XAUUSD): 100 troy ounces per lot
    - Silver (XAGUSD): 5,000 oz per lot
    - Crypto (BTCUSD, ETHUSD, etc.): 1 unit per lot
    - Indices (US30, NAS100, SPX500, etc.): 1 unit per lot
    """
    s = symbol.upper().replace("/", "").replace("-", "").replace("_", "").replace(".", "")

    # Gold
    if "XAU" in s or "GOLD" in s:
        return 100.0

    # Silver
    if "XAG" in s or "SILVER" in s:
        return 5000.0

    # Crypto
    crypto_bases = ["BTC", "ETH", "LTC", "XRP", "ADA", "SOL", "DOT", "MATIC", "BNB", "AVAX", "LINK", "DOGE"]
    for c in crypto_bases:
        if s.startswith(c):
            return 1.0

    # Indices
    index_keywords = ["US30", "NAS", "SPX", "DAX", "FTSE", "NDX", "GER", "UK100", "AUS200", "JPN225", "CN50"]
    for idx in index_keywords:
        if idx in s:
            return 1.0

    # Standard forex (including JPY pairs): 100,000 units per lot
    return 100000.0


# ─── Custom Backtest Request (AI-generated Python code) ────────────────────────

class CustomBacktestRequest(BaseModel):
    strategyId: str
    strategyName: str
    datasetId: str
    datasetName: str
    symbol: str = ""
    bars: Optional[list] = None
    cache_key: Optional[str] = None
    pythonCode: str
    stopLossPercent: float = 1.0        # Default 1% stop distance (price movement, NOT account risk %)
    takeProfitPercent: float = 2.0      # Default 2% target distance (1:2 R:R)
    stopLossPips: Optional[float] = None
    takeProfitPips: Optional[float] = None
    tradeDirection: Literal["long", "short", "both"] = "both"  # Default both — Python strategies are bidirectional
    initialCapital: float = 100000.0   # Default account size 100,000
    riskPercent: float = 1.0           # Account risk per trade: 1% of equity (NOT the stop distance)
    riskType: Literal["percentBalance", "fixedLot"] = "percentBalance"
    lotSize: float = 0.1
    executeOnNextBar: bool = True       # True = entry fires at next bar's open (no look-ahead bias)
    spreadPips: float = 0.0             # Bid/ask spread in pips — applied half on entry, half on exit
    slippagePips: float = 0.0           # Execution slippage in pips — applied in full on entry and exit
    maxOpenPositions: int = 1           # Engine is single-position only. Values >1 are clamped to 1 with a warning.
    pip_size: Optional[float] = None    # Caller-supplied pip size. When None (default), engine falls back to get_pip_size(symbol). Set this to avoid relying on the engine's symbol-keyword lookup when the backend already knows the correct pip size.


# ─── Legacy Fixed-Indicator Request ────────────────────────────────────────────

class SignalCondition(BaseModel):
    indicator: str
    params: dict = {}


class SignalSpec(BaseModel):
    entryLong: Optional[SignalCondition] = None
    entryShort: Optional[SignalCondition] = None
    exitLong: Optional[SignalCondition] = None
    exitShort: Optional[SignalCondition] = None
    stopLossPercent: float = 2.0
    takeProfitPercent: float = 4.0
    tradeDirection: Literal["long", "short", "both"] = "both"


class BacktestRequest(BaseModel):
    strategyId: str
    strategyName: str
    datasetId: str
    datasetName: str
    symbol: str = ""
    bars: List[Any]
    signalSpec: SignalSpec
    initialCapital: float = 100000.0
    riskPercent: float = 1.0
    riskType: Literal["percentBalance", "fixedLot"] = "percentBalance"
    lotSize: float = 0.1


# ─── Walk-Forward Request ───────────────────────────────────────────────────────

class WalkForwardRequest(BaseModel):
    strategyId: str
    strategyName: str
    datasetId: str
    datasetName: str
    symbol: str = ""
    bars: List[Any]
    pythonCode: str
    stopLossPercent: float = 1.0        # Default 1% stop distance (NOT account risk %)
    takeProfitPercent: float = 2.0      # Default 2% target distance
    stopLossPips: Optional[float] = None
    takeProfitPips: Optional[float] = None
    tradeDirection: Literal["long", "short", "both"] = "both"  # Default both
    initialCapital: float = 100000.0   # Default account size 100,000
    riskPercent: float = 1.0           # Account risk per trade: 1% of equity (NOT stop distance)
    riskType: Literal["percentBalance", "fixedLot"] = "percentBalance"
    lotSize: float = 0.1
    executeOnNextBar: bool = True
    spreadPips: float = 0.0
    slippagePips: float = 0.0
    maxOpenPositions: int = 1           # Engine is single-position only. Values >1 are clamped to 1 with a warning.
    pip_size: Optional[float] = None    # Caller-supplied pip size. When None, engine falls back to get_pip_size(symbol).
    # Walk-forward settings
    nWindows: int = 5          # number of windows
    inSamplePct: float = 0.7   # fraction of each window used for in-sample (0.7 = 70%)


class WalkForwardWindow(BaseModel):
    windowIndex: int
    inSampleStart: str
    inSampleEnd: str
    outOfSampleStart: str
    outOfSampleEnd: str
    inSampleResult: BacktestResult
    outOfSampleResult: BacktestResult


class WalkForwardResult(BaseModel):
    id: str
    strategyId: str
    strategyName: str
    datasetName: str
    nWindows: int
    inSamplePct: float
    windows: List[WalkForwardWindow]
    # Aggregate out-of-sample stats
    totalOutOfSampleTrades: int
    combinedWinRate: float
    combinedNetProfitPercent: float
    combinedSharpeRatio: float
    consistencyScore: float   # % of windows where out-of-sample was profitable
    createdAt: str
    startedAt: str = ""
    completedAt: str = ""
    durationMs: float = 0.0
    # Execution settings
    executeOnNextBar: Optional[bool] = None
    spreadPips: Optional[float] = None
    slippagePips: Optional[float] = None
    maxOpenPositions: Optional[int] = None


# ─── Monte Carlo Request ────────────────────────────────────────────────────────

class MonteCarloRequest(BaseModel):
    strategyId: str
    strategyName: str
    trades: List[TradeResult]    # Pass in the trades from a completed backtest
    initialCapital: float = 100000.0
    nSimulations: int = 1000
    confidenceLevels: List[float] = [0.05, 0.25, 0.50, 0.75, 0.95]
    resampleMode: Literal["bootstrap", "shuffle", "block"] = "bootstrap"
    # bootstrap = resample single trades with replacement (standard Monte Carlo)
    # shuffle   = random permutation of the exact trade sequence (no replacement)
    # block     = resample CONTIGUOUS blocks of trades with replacement — preserves
    #             win/loss streak autocorrelation, giving realistic drawdown tails
    blockSize: int = 0           # 0 = auto (~sqrt(n_trades)); only used when resampleMode="block"
    # ── Risk-of-ruin (intra-path, measured from the starting balance) ──
    ruinThresholdPct: float = 50.0   # a path is "ruined" if equity EVER falls this % below start
    # ── Prop-firm challenge model ──
    # A path PASSES if it reaches +propProfitTargetPct before its drawdown breaches
    # propMaxDrawdownPct; it FAILS if the drawdown cap is breached first.
    propProfitTargetPct: float = 10.0
    propMaxDrawdownPct: float = 10.0
    propDrawdownMode: Literal["trailing", "initial"] = "trailing"
    # trailing = drawdown from the running peak (matches the app's maxDrawdownPercent)
    # initial  = loss from the starting balance (static max-loss rule some firms use)
    horizonTrades: int = 0       # 0 = use the backtest's trade count; else simulate this many
                                 # trades per path (e.g. the trades expected in a 6-week phase)


class MonteCarloResult(BaseModel):
    id: str
    strategyId: str
    strategyName: str
    nSimulations: int
    initialCapital: float
    resampleMode: str
    # Final equity distribution
    medianFinalEquity: float
    meanFinalEquity: float
    worstCaseFinalEquity: float    # 5th percentile
    bestCaseFinalEquity: float     # 95th percentile
    probabilityOfProfit: float     # % of sims that ended profitable
    probabilityOfRuin: float       # % of sims whose equity EVER fell ruinThresholdPct below start (INTRA-PATH)
    ruinThresholdPct: float = 50.0 # echo of the threshold used for probabilityOfRuin
    # Max drawdown distribution
    medianMaxDrawdown: float
    worstCaseMaxDrawdown: float    # 95th percentile drawdown
    # ── Prop-firm challenge outcomes (reach target before breaching the DD cap) ──
    propProfitTargetPct: float = 0.0
    propMaxDrawdownPct: float = 0.0
    propPassProbability: float = 0.0     # % that hit the profit target before the DD cap
    propFailDrawdown: float = 0.0        # % that breached the DD cap before the target
    propIncomplete: float = 0.0          # % that did neither within the trade horizon
    medianTradesToTarget: float = 0.0    # among passers, median trades taken to reach target
    # Percentile curves (sampled equity curves at key percentiles)
    percentileCurves: dict         # {"p5": [...], "p25": [...], "p50": [...], "p75": [...], "p95": [...]}
    createdAt: str
    startedAt: str = ""
    completedAt: str = ""
    durationMs: float = 0.0


# ─── Optimisation Request ───────────────────────────────────────────────────────

class OptimizeRequest(BaseModel):
    strategyId: str
    strategyName: str
    datasetId: str
    datasetName: str
    symbol: str = ""
    bars: Optional[list] = None
    cache_key: Optional[str] = None
    pythonCode: str              # Must use param_* variables (see docs)
    tradeDirection: Literal["long", "short", "both"] = "both"  # Default both
    initialCapital: float = 100000.0   # Default account size 100,000
    riskPercent: float = 1.0           # Account risk per trade: 1% of equity (NOT stop distance)
    riskType: Literal["percentBalance", "fixedLot"] = "percentBalance"
    lotSize: float = 0.1
    stopLossPercent: float = 1.0        # Default 1% stop distance (NOT account risk %)
    takeProfitPercent: float = 2.0      # Default 2% target distance
    stopLossPips: Optional[float] = None
    takeProfitPips: Optional[float] = None
    executeOnNextBar: bool = True
    spreadPips: float = 0.0
    slippagePips: float = 0.0
    maxOpenPositions: int = 1           # Engine is single-position only. Values >1 are clamped to 1 with a warning.
    # Parameter grid — key = variable name in pythonCode, value = list of values to try
    paramGrid: dict              # e.g. {"fast_period": [10, 20, 30], "slow_period": [50, 100, 200]}
    optimizeFor: Literal["sharpe", "netProfit", "winRate", "profitFactor", "returnDrawdown"] = "sharpe"
    secondaryObjective: Optional[Literal["sharpe", "netProfit", "winRate", "profitFactor", "maxDrawdown", "returnDrawdown"]] = None
    maxCombinations: int = 500   # cap to avoid runaway compute (raised from 200)
    pip_size: Optional[float] = None    # Caller-supplied pip size. When None, engine falls back to get_pip_size(symbol).
    # Walk-forward fields — when enabled, combos are scored by avg out-of-sample performance
    walkForwardEnabled: bool = False
    nWindows: int = 5
    inSamplePct: float = 0.7


class OptimizeResultRow(BaseModel):
    params: dict
    netProfitPercent: float
    winRate: float
    sharpeRatio: float
    profitFactor: float
    maxDrawdownPercent: float
    totalTrades: int
    winningTrades: int
    losingTrades: int
    consecutiveWins: int
    consecutiveLosses: int
    score: int
    grade: str


class OptimizeResult(BaseModel):
    id: str
    strategyId: str
    strategyName: str
    datasetName: str
    nCombinations: int
    optimizeFor: str
    bestParams: dict
    bestResult: OptimizeResultRow
    allResults: List[OptimizeResultRow]
    totalGridCombinations: int = 0   # full grid size before any capping
    testedCombinations: int = 0      # how many combos were actually attempted
    wasCapped: bool = False          # True if grid was larger than maxCombinations
    samplingMethod: str = "full_grid"  # "full_grid" or "random_sample"
    createdAt: str
    startedAt: str = ""
    completedAt: str = ""
    durationMs: float = 0.0
    # Execution settings
    executeOnNextBar: Optional[bool] = None
    spreadPips: Optional[float] = None
    slippagePips: Optional[float] = None
    maxOpenPositions: Optional[int] = None


# ─── Trade Simulation (shared by all endpoints) ────────────────────────────────

def simulate_trades(
    df: pd.DataFrame,
    entry_long: pd.Series,
    entry_short: pd.Series,
    exit_long: pd.Series,
    exit_short: pd.Series,
    stop_loss_pct: float,
    take_profit_pct: float,
    trade_direction: str,
    initial_capital: float,
    risk_percent: float,
    risk_type: str,
    lot_size: float,
    stop_loss_pips: Optional[float] = None,
    take_profit_pips: Optional[float] = None,
    pip_size: float = 0.0001,
    sl_price_series: Optional[pd.Series] = None,
    tp_price_series: Optional[pd.Series] = None,
    execute_on_next_bar: bool = True,
    spread_pips: float = 0.0,
    slippage_pips: float = 0.0,
    contract_size: float = 100000.0,
) -> tuple[list, list]:
    """
    Simulates trades on the given OHLC dataframe.

    SL/TP priority (highest to lowest):
    1. Dynamic price series (sl_price_series / tp_price_series) — set per entry bar
       by the signal code. Covers ATR stops, session lows, EMA levels, etc.
       Position is sized to risk exactly risk_percent% of equity over the SL distance.
    2. Fixed-pip mode (stop_loss_pips / take_profit_pips + pip_size) — places SL/TP
       at exactly N pips from entry. Position sized over pip distance.
    3. Percentage fallback — legacy mode, uses entry_price * (1 ± pct).

    exit_long / exit_short signals are manual-exit signals. When execute_on_next_bar=True
    they are also deferred and execute at the NEXT bar's open (same as entries).
    SL and TP always fire intrabar once a trade is active — they are not deferred.

    execute_on_next_bar: when True, BOTH entry signals AND manual exit signals fire at
    the NEXT bar's open price (eliminates bar-close look-ahead bias). SL/TP remain
    intrabar and are not affected.

    spread_pips / slippage_pips: deterministic execution costs applied as:
        cost_per_side = (spread_pips / 2 + slippage_pips) * pip_size
    Long entries pay more; long exits receive less. Short entries receive less;
    short exits pay more. Both SL and TP exits are also adjusted.
    """
    trades = []
    equity = initial_capital
    equity_curve = [initial_capital]

    in_position = False
    position_type = None
    entry_price = 0.0
    entry_date = ""
    active_sl_price = 0.0
    active_tp_price = 0.0

    use_dynamic_mode = (sl_price_series is not None and sl_price_series.notna().any())
    use_pip_mode     = (not use_dynamic_mode and stop_loss_pips is not None and stop_loss_pips > 0)

    sl_pct = stop_loss_pct / 100
    tp_pct = take_profit_pct / 100

    # Execution cost per side: half-spread + full slippage, converted to price units
    cost_per_side = (spread_pips / 2 + slippage_pips) * pip_size

    close_arr = df["close"].values
    open_arr  = df["open"].values
    high_arr  = df["high"].values
    low_arr   = df["low"].values
    date_arr  = df["timestamp"].values

    entry_long_arr  = entry_long.values
    entry_short_arr = entry_short.values
    exit_long_arr   = exit_long.values
    exit_short_arr  = exit_short.values

    sl_arr = sl_price_series.values if sl_price_series is not None else None
    tp_arr = tp_price_series.values if tp_price_series is not None else None

    # Pending entry state (used when execute_on_next_bar=True)
    pending_type: Optional[str] = None
    pending_use_dynamic: bool = False
    pending_dyn_sl: float = 0.0
    pending_dyn_tp: float = 0.0

    # Pending manual exit state (used when execute_on_next_bar=True)
    # SL/TP are NOT deferred — they fire intrabar. Only manual exit signals are deferred.
    pending_exit: bool = False

    warmup = 52

    n_bars = len(df)
    for i in range(warmup, n_bars):
        just_opened = False  # guard: skip exit checks on the bar a trade opens

        close = close_arr[i]
        open_ = open_arr[i]
        high  = high_arr[i]
        low   = low_arr[i]
        date  = str(date_arr[i])

        # ── Execute pending entry at this bar's open ───────────────────────────
        if not in_position and pending_type is not None:
            in_position   = True
            position_type = pending_type
            raw_price     = open_
            entry_price   = raw_price + cost_per_side if pending_type == "long" else raw_price - cost_per_side
            entry_date    = date

            if pending_use_dynamic:
                active_sl_price = pending_dyn_sl if pending_dyn_sl > 0 else (
                    entry_price * (1 - sl_pct) if pending_type == "long" else entry_price * (1 + sl_pct)
                )
                active_tp_price = pending_dyn_tp
            elif use_pip_mode:
                if pending_type == "long":
                    active_sl_price = entry_price - stop_loss_pips * pip_size
                    active_tp_price = entry_price + take_profit_pips * pip_size if (take_profit_pips and take_profit_pips > 0) else 0.0
                else:
                    active_sl_price = entry_price + stop_loss_pips * pip_size
                    active_tp_price = entry_price - take_profit_pips * pip_size if (take_profit_pips and take_profit_pips > 0) else 0.0
            else:
                active_sl_price = entry_price * (1 - sl_pct) if pending_type == "long" else entry_price * (1 + sl_pct)
                active_tp_price = entry_price * (1 + tp_pct) if pending_type == "long" else entry_price * (1 - tp_pct)

            pending_type = None

            # Minimum stop distance guard — prevents position sizing explosion
            # when generated Python code returns a near-zero SL distance.
            _sl_distance = abs(entry_price - active_sl_price)
            _min_stop = entry_price * 0.0005  # 0.05% minimum (e.g. $1.50 on $3000 gold)
            if _sl_distance < _min_stop:
                # Stop too tight to be meaningful — skip this trade entirely
                in_position = False
                position_type = None
                pending_type = None
                pending_exit = False
                continue

            just_opened = True  # trade opened this bar — skip exit checks below

        # ── Check for entry signal ─────────────────────────────────────────────
        if not in_position:
            if trade_direction in ("long", "both") and bool(entry_long_arr[i]):
                if execute_on_next_bar:
                    pending_type        = "long"
                    pending_use_dynamic = use_dynamic_mode
                    if use_dynamic_mode:
                        pending_dyn_sl = float(sl_arr[i]) if not (sl_arr is None or np.isnan(sl_arr[i])) else 0.0
                        pending_dyn_tp = float(tp_arr[i]) if (tp_arr is not None and not np.isnan(tp_arr[i])) else 0.0
                    continue
                else:
                    in_position   = True
                    position_type = "long"
                    entry_price   = close + cost_per_side
                    entry_date    = date
                    if use_dynamic_mode:
                        active_sl_price = float(sl_arr[i]) if not (sl_arr is None or np.isnan(sl_arr[i])) else entry_price * (1 - sl_pct)
                        active_tp_price = float(tp_arr[i]) if (tp_arr is not None and not np.isnan(tp_arr[i])) else 0.0
                    elif use_pip_mode:
                        active_sl_price = entry_price - stop_loss_pips * pip_size
                        active_tp_price = entry_price + take_profit_pips * pip_size if (take_profit_pips and take_profit_pips > 0) else 0.0
                    else:
                        active_sl_price = entry_price * (1 - sl_pct)
                        active_tp_price = entry_price * (1 + tp_pct)
                    continue

            if trade_direction in ("short", "both") and bool(entry_short_arr[i]):
                if execute_on_next_bar:
                    pending_type        = "short"
                    pending_use_dynamic = use_dynamic_mode
                    if use_dynamic_mode:
                        pending_dyn_sl = float(sl_arr[i]) if not (sl_arr is None or np.isnan(sl_arr[i])) else 0.0
                        pending_dyn_tp = float(tp_arr[i]) if (tp_arr is not None and not np.isnan(tp_arr[i])) else 0.0
                    continue
                else:
                    in_position   = True
                    position_type = "short"
                    entry_price   = close - cost_per_side
                    entry_date    = date
                    if use_dynamic_mode:
                        active_sl_price = float(sl_arr[i]) if not (sl_arr is None or np.isnan(sl_arr[i])) else entry_price * (1 + sl_pct)
                        active_tp_price = float(tp_arr[i]) if (tp_arr is not None and not np.isnan(tp_arr[i])) else 0.0
                    elif use_pip_mode:
                        active_sl_price = entry_price + stop_loss_pips * pip_size
                        active_tp_price = entry_price - take_profit_pips * pip_size if (take_profit_pips and take_profit_pips > 0) else 0.0
                    else:
                        active_sl_price = entry_price * (1 + sl_pct)
                        active_tp_price = entry_price * (1 - tp_pct)
                    continue

        # ── Check for exit ─────────────────────────────────────────────────────
        elif not just_opened:
            exit_triggered    = False
            actual_exit_price = close

            if position_type == "long":
                if low <= active_sl_price:
                    # SL fires intrabar — never deferred
                    actual_exit_price = active_sl_price - cost_per_side
                    exit_triggered    = True
                    pending_exit      = False   # cancel any pending manual exit
                elif active_tp_price > 0 and high >= active_tp_price:
                    # TP fires intrabar — never deferred
                    actual_exit_price = active_tp_price - cost_per_side
                    exit_triggered    = True
                    pending_exit      = False
                elif pending_exit:
                    # Deferred manual exit executes at this bar's open
                    actual_exit_price = open_ - cost_per_side
                    exit_triggered    = True
                    pending_exit      = False
                elif bool(exit_long_arr[i]):
                    if execute_on_next_bar:
                        # Defer manual exit to next bar's open
                        pending_exit = True
                    else:
                        actual_exit_price = close - cost_per_side
                        exit_triggered    = True
            else:  # short
                if high >= active_sl_price:
                    # SL fires intrabar — never deferred
                    actual_exit_price = active_sl_price + cost_per_side
                    exit_triggered    = True
                    pending_exit      = False
                elif active_tp_price > 0 and low <= active_tp_price:
                    # TP fires intrabar — never deferred
                    actual_exit_price = active_tp_price + cost_per_side
                    exit_triggered    = True
                    pending_exit      = False
                elif pending_exit:
                    # Deferred manual exit executes at this bar's open
                    actual_exit_price = open_ + cost_per_side
                    exit_triggered    = True
                    pending_exit      = False
                elif bool(exit_short_arr[i]):
                    if execute_on_next_bar:
                        # Defer manual exit to next bar's open
                        pending_exit = True
                    else:
                        actual_exit_price = close + cost_per_side
                        exit_triggered    = True

            if exit_triggered:
                # Position sizing — always based on the actual SL distance from entry
                sl_distance = abs(entry_price - active_sl_price)

                if risk_type == "fixedLot":
                    position_size = lot_size * contract_size * entry_price
                elif sl_distance > 0:
                    risk_amount   = equity * (risk_percent / 100)
                    # position_size * (sl_distance / entry_price) = risk_amount
                    position_size = risk_amount * entry_price / sl_distance
                else:
                    position_size = equity * (risk_percent / 100)

                # Cap position size to prevent runaway sizing
                _max_position = equity * 10  # never risk more than 10x account value
                position_size = min(position_size, _max_position)

                if position_type == "long":
                    pnl = position_size * ((actual_exit_price - entry_price) / entry_price)
                else:
                    pnl = position_size * ((entry_price - actual_exit_price) / entry_price)

                pnl_pct = (pnl / equity) * 100
                equity  += pnl

                trades.append({
                    "entryDate":    entry_date,
                    "exitDate":     date,
                    "entryPrice":   round(entry_price, 6),
                    "exitPrice":    round(actual_exit_price, 6),
                    "type":         position_type,
                    "pnl":          round(pnl, 2),
                    "pnlPercent":   round(pnl_pct, 2),
                    "positionSize": round(position_size, 2),
                })
                equity_curve.append(round(equity, 2))

                in_position   = False
                position_type = None
                entry_price   = 0.0
                pending_exit  = False

    # Close any open position (or pending entry) at end of data using the last bar's close.
    # A pending entry that was never filled is simply discarded (no data to fill it).
    if in_position:
        close_final  = close_arr[-1]
        actual_final = close_final - cost_per_side if position_type == "long" else close_final + cost_per_side
        sl_distance  = abs(entry_price - active_sl_price)

        if risk_type == "fixedLot":
            position_size = lot_size * contract_size * entry_price
        elif sl_distance > 0:
            risk_amount   = equity * (risk_percent / 100)
            position_size = risk_amount * entry_price / sl_distance
        else:
            position_size = equity * (risk_percent / 100)

        # Cap position size to prevent runaway sizing
        _max_position = equity * 10  # never risk more than 10x account value
        position_size = min(position_size, _max_position)

        if position_type == "long":
            pnl = position_size * ((actual_final - entry_price) / entry_price)
        else:
            pnl = position_size * ((entry_price - actual_final) / entry_price)

        pnl_pct = (pnl / equity) * 100
        equity  += pnl

        trades.append({
            "entryDate":    entry_date,
            "exitDate":     str(date_arr[-1]),
            "entryPrice":   round(entry_price, 6),
            "exitPrice":    round(actual_final, 6),
            "type":         position_type,
            "pnl":          round(pnl, 2),
            "pnlPercent":   round(pnl_pct, 2),
            "positionSize": round(position_size, 2),
        })
        equity_curve.append(round(equity, 2))

    return trades, equity_curve


# ─── AI Code Safety ────────────────────────────────────────────────────────────

_DANGEROUS_PATTERNS = [
    "import os", "import sys", "import subprocess", "import socket",
    "import shutil", "import pathlib", "import importlib",
    "__import__", "open(", "eval(", "exec(", "compile(",
    "globals(", "locals(", "getattr(", "setattr(", "delattr(",
    "__builtins__", "__class__", "__subclasses__",
]

_SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "filter": filter, "float": float, "int": int,
    "isinstance": isinstance, "len": len, "list": list, "map": map,
    "max": max, "min": min, "print": print, "range": range, "round": round,
    "set": set, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
    "zip": zip, "True": True, "False": False, "None": None,
}


def check_code_safety(code: str) -> None:
    """Reject code containing obviously dangerous patterns before execution."""
    code_lower = code.lower()
    for pattern in _DANGEROUS_PATTERNS:
        if pattern.lower() in code_lower:
            raise HTTPException(
                status_code=400,
                detail=f"Strategy code contains a disallowed pattern: '{pattern}'"
            )


# ─── Execute AI-generated Python code against a DataFrame ─────────────────────

def run_python_signals(code: str, df: pd.DataFrame, extra_params: dict = None) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, Optional[pd.Series], Optional[pd.Series]]:
    """
    Executes AI-generated Python code that defines get_signals(df).
    Returns (entry_long, entry_short, exit_long, exit_short, sl_price, tp_price).

    sl_price and tp_price are optional Series of floats — one value per bar,
    set to the exact SL/TP price at the bar the trade is entered, NaN otherwise.
    When present they override the fixed percentage/pip SL/TP entirely.
    This enables dynamic stops: ATR-based, session lows, EMA levels, etc.
    """
    check_code_safety(code)

    module = types.ModuleType("strategy_signals")
    module.__dict__["__builtins__"] = _SAFE_BUILTINS
    module.__dict__["pd"] = pd
    module.__dict__["ta"] = ta
    module.__dict__["np"] = np
    module.__dict__["vbt"] = vbt

    # Inject any extra parameter values (used during optimisation)
    if extra_params:
        for k, v in extra_params.items():
            module.__dict__[k] = v

    try:
        exec(compile(code, "<strategy>", "exec"), module.__dict__)
    except SyntaxError as e:
        raise HTTPException(status_code=400, detail=f"Python syntax error in strategy code: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading strategy code: {str(e)}")

    if "get_signals" not in module.__dict__:
        raise HTTPException(
            status_code=400,
            detail="Strategy code must define a get_signals(df) function"
        )

    try:
        signals = module.__dict__["get_signals"](df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error running get_signals(): {str(e)}")

    for key in ["entry_long", "entry_short", "exit_long", "exit_short"]:
        if key not in signals:
            raise HTTPException(status_code=400, detail=f"get_signals() must return a dict with key '{key}'")

    entry_long  = pd.Series(signals["entry_long"],  index=df.index).fillna(False).astype(bool)
    entry_short = pd.Series(signals["entry_short"], index=df.index).fillna(False).astype(bool)
    exit_long   = pd.Series(signals["exit_long"],   index=df.index).fillna(False).astype(bool)
    exit_short  = pd.Series(signals["exit_short"],  index=df.index).fillna(False).astype(bool)

    # Optional dynamic SL/TP price series
    sl_price: Optional[pd.Series] = None
    tp_price: Optional[pd.Series] = None
    if "sl_price" in signals and signals["sl_price"] is not None:
        sl_price = pd.Series(signals["sl_price"], index=df.index).astype(float)
    if "tp_price" in signals and signals["tp_price"] is not None:
        tp_price = pd.Series(signals["tp_price"], index=df.index).astype(float)

    return entry_long, entry_short, exit_long, exit_short, sl_price, tp_price


# ─── Metrics & Grading ─────────────────────────────────────────────────────────

def compute_metrics(trades: list, equity_curve: list, initial_capital: float):
    total = len(trades)
    winning = [t for t in trades if t["pnl"] > 0]
    losing = [t for t in trades if t["pnl"] <= 0]

    win_rate = (len(winning) / total * 100) if total > 0 else 0.0
    final_equity = equity_curve[-1] if equity_curve else initial_capital
    net_profit = final_equity - initial_capital
    net_profit_pct = (net_profit / initial_capital) * 100

    gross_profit = sum(t["pnl"] for t in winning)
    gross_loss = abs(sum(t["pnl"] for t in losing))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)

    peak = initial_capital
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = ((peak - eq) / peak) * 100
        if dd > max_dd:
            max_dd = dd

    max_cons_wins = 0
    max_cons_losses = 0
    cur_wins = 0
    cur_losses = 0
    for t in trades:
        if t["pnl"] > 0:
            cur_wins += 1
            cur_losses = 0
            max_cons_wins = max(max_cons_wins, cur_wins)
        else:
            cur_losses += 1
            cur_wins = 0
            max_cons_losses = max(max_cons_losses, cur_losses)

    returns = [t["pnlPercent"] / 100 for t in trades]
    if len(returns) > 1:
        avg_r = np.mean(returns)
        std_r = np.std(returns, ddof=1)
        sharpe = (avg_r / std_r) * math.sqrt(252) if std_r > 0 else 0.0
    else:
        sharpe = 0.0

    return {
        "winRate": round(win_rate, 2),
        "netProfit": round(net_profit, 2),
        "netProfitPercent": round(net_profit_pct, 2),
        "profitFactor": round(min(profit_factor, 999.0), 2),
        "maxDrawdownPercent": round(max_dd, 2),
        "totalTrades": total,
        "winningTrades": len(winning),
        "losingTrades": len(losing),
        "consecutiveWins": max_cons_wins,
        "consecutiveLosses": max_cons_losses,
        "sharpeRatio": round(sharpe, 2),
    }


def grade_from_score(score: int) -> str:
    if score >= 90: return "A+"
    if score >= 80: return "A"
    if score >= 70: return "B+"
    if score >= 60: return "B"
    if score >= 50: return "C"
    if score >= 40: return "D"
    return "F"


def compute_score(win_rate, profit_factor, max_dd, sharpe, total_trades, net_pct) -> int:
    score = 0
    if win_rate >= 60: score += 25
    elif win_rate >= 50: score += 20
    elif win_rate >= 40: score += 15
    elif win_rate >= 30: score += 10
    else: score += 5

    if profit_factor >= 3: score += 25
    elif profit_factor >= 2: score += 20
    elif profit_factor >= 1.5: score += 15
    elif profit_factor >= 1: score += 10

    if max_dd <= 5: score += 20
    elif max_dd <= 10: score += 16
    elif max_dd <= 20: score += 12
    elif max_dd <= 30: score += 8
    else: score += 4

    if sharpe >= 2: score += 15
    elif sharpe >= 1.5: score += 12
    elif sharpe >= 1: score += 9
    elif sharpe >= 0.5: score += 6
    elif sharpe > 0: score += 3

    if total_trades >= 30: score += 5
    elif total_trades >= 20: score += 4
    elif total_trades >= 10: score += 3
    elif total_trades >= 5: score += 2
    else: score += 1

    if net_pct >= 50: score += 10
    elif net_pct >= 25: score += 8
    elif net_pct >= 10: score += 6
    elif net_pct >= 0: score += 3

    return min(100, max(0, score))


def generate_id() -> str:
    return str(uuid.uuid4())[:8]


def build_result(
    req_id, strategy_id, dataset_id,
    strategy_name, dataset_name,
    trades, equity_curve,
    initial_capital, risk_percent,
    risk_type, lot_size, stop_loss_pct,
    engine_version,
    started_at: str = "",
    completed_at: str = "",
    duration_ms: float = 0.0,
    execute_on_next_bar: Optional[bool] = None,
    spread_pips: Optional[float] = None,
    slippage_pips: Optional[float] = None,
    max_open_positions: Optional[int] = None,
) -> BacktestResult:
    metrics = compute_metrics(trades, equity_curve, initial_capital)
    score = compute_score(
        metrics["winRate"], metrics["profitFactor"],
        metrics["maxDrawdownPercent"], metrics["sharpeRatio"],
        metrics["totalTrades"], metrics["netProfitPercent"]
    )
    return BacktestResult(
        id=req_id,
        strategyId=strategy_id,
        datasetId=dataset_id,
        strategyName=strategy_name,
        datasetName=dataset_name,
        netProfit=metrics["netProfit"],
        netProfitPercent=metrics["netProfitPercent"],
        profitFactor=metrics["profitFactor"],
        maxDrawdownPercent=metrics["maxDrawdownPercent"],
        winRate=metrics["winRate"],
        totalTrades=metrics["totalTrades"],
        winningTrades=metrics["winningTrades"],
        losingTrades=metrics["losingTrades"],
        consecutiveWins=metrics["consecutiveWins"],
        consecutiveLosses=metrics["consecutiveLosses"],
        sharpeRatio=metrics["sharpeRatio"],
        grade=grade_from_score(score),
        score=score,
        trades=trades,
        equityCurve=equity_curve,
        riskPercent=risk_percent,
        riskType=risk_type,
        lotSize=lot_size,
        stopLossPercent=stop_loss_pct,
        createdAt=datetime.utcnow().isoformat(),
        engineVersion=engine_version,
        startedAt=started_at,
        completedAt=completed_at,
        durationMs=duration_ms,
        executeOnNextBar=execute_on_next_bar,
        spreadPips=spread_pips,
        slippagePips=slippage_pips,
        maxOpenPositions=max_open_positions,
    )


def bars_to_df(bars) -> pd.DataFrame:
    rows = []
    for b in bars:
        if isinstance(b, dict):
            rows.append({
                "timestamp": b["timestamp"],
                "open": b["open"],
                "high": b["high"],
                "low": b["low"],
                "close": b["close"],
                "volume": b.get("volume") or 0.0,
            })
        else:
            rows.append({
                "timestamp": b.timestamp,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume if b.volume else 0.0,
            })
    return pd.DataFrame(rows)


# ─── POST /backtest-custom ─────────────────────────────────────────────────────

@app.post("/backtest-custom", response_model=BacktestResult)
async def run_custom_backtest(req: CustomBacktestRequest):
    """Run a backtest using AI-generated Python signal code."""
    started_at = datetime.utcnow()
    try:
        # Resolve bars from cache or inline
        if not req.bars and req.cache_key:
            if req.cache_key not in _bar_cache:
                return JSONResponse({"error": "Bar cache miss — upload bars first",
                                     "code": "CACHE_MISS"}, status_code=400)
            bars = _bar_cache[req.cache_key]
            _bar_cache.move_to_end(req.cache_key)  # refresh LRU
        elif req.bars:
            bars = req.bars
        else:
            return JSONResponse({"error": "bars or cache_key required"}, status_code=400)

        # Convert raw dicts (from cache) or OHLCBar objects to OHLCBar list
        ohlc_bars = [OHLCBar(**b) if isinstance(b, dict) else b for b in bars]

        if len(ohlc_bars) < 60:
            raise HTTPException(status_code=400, detail="Need at least 60 bars to run a backtest")

        # Engine is single-position only. Clamp maxOpenPositions to 1 and warn if the
        # caller passed a higher value so they are aware the extra positions won't open.
        if req.maxOpenPositions != 1:
            print(f"[backtest-custom] maxOpenPositions={req.maxOpenPositions} requested but engine supports only 1 open position at a time. Clamping to 1.")
            req.maxOpenPositions = 1

        # Prefer the caller-supplied pip_size when provided (canonical source of truth).
        # Fall back to the engine's symbol-keyword lookup so older callers continue to work.
        pip_size = req.pip_size if (getattr(req, "pip_size", None) is not None) else (get_pip_size(req.symbol) if req.symbol else 0.0001)

        # ── Determine active SL/TP mode ──────────────────────────────────────────
        # Priority: 1) sl_price/tp_price from Python signals  2) pips  3) percent
        # We detect mode after running signals, so log what the caller sent first.
        use_pip_mode_expected = (req.stopLossPips is not None and req.stopLossPips > 0)
        has_sl_pct = (req.stopLossPercent > 0)
        print(f"\n[backtest-custom] ===== EXECUTION SETTINGS =====")
        print(f"  strategy:          {req.strategyName!r}")
        print(f"  symbol:            {req.symbol!r}  pip_size={pip_size}")
        print(f"  tradeDirection:    {req.tradeDirection}")
        print(f"  executeOnNextBar:  {req.executeOnNextBar}")
        print(f"  initialCapital:    {req.initialCapital:,.0f}")
        print(f"  riskPercent:       {req.riskPercent}%  (account risk per trade, NOT stop distance)")
        print(f"  riskType:          {req.riskType}")
        print(f"  lotSize:           {req.lotSize}")
        print(f"  --- SL/TP source (caller-provided) ---")
        print(f"  stopLossPercent:   {req.stopLossPercent}%  ({'price distance fallback' if has_sl_pct else 'NOT SET'})")
        print(f"  takeProfitPercent: {req.takeProfitPercent}%")
        print(f"  stopLossPips:      {req.stopLossPips}  ({'pip mode expected' if use_pip_mode_expected else 'not set'})")
        print(f"  takeProfitPips:    {req.takeProfitPips}")
        print(f"  spreadPips:        {req.spreadPips}")
        print(f"  slippagePips:      {req.slippagePips}")
        print(f"  bars:              {len(ohlc_bars)}")
        print(f"  NOTE: Dynamic sl_price/tp_price mode is detected AFTER running Python signals.")
        print(f"==============================================\n")

        print(f"[backtest-custom] symbol={req.symbol!r} pip_size={pip_size} sl_pips={req.stopLossPips} tp_pips={req.takeProfitPips}")

        df = bars_to_df(ohlc_bars)
        entry_long, entry_short, exit_long, exit_short, sl_price_series, tp_price_series = run_python_signals(req.pythonCode, df)

        # ── Log the active SL/TP mode (determined after running signals) ─────────
        _has_dynamic_sl = (sl_price_series is not None and sl_price_series.notna().any())
        active_mode = (
            "DYNAMIC (sl_price/tp_price from Python signals)" if _has_dynamic_sl
            else f"PIP-BASED ({req.stopLossPips} pips SL / {req.takeProfitPips} pips TP, pip_size={pip_size})" if (req.stopLossPips is not None and req.stopLossPips > 0)
            else f"PERCENT ({req.stopLossPercent}% SL / {req.takeProfitPercent}% TP)"
        )
        print(f"[backtest-custom] Active SL/TP mode: {active_mode}")
        print(f"[backtest-custom] Position sizing: risk_type={req.riskType}, risk_percent={req.riskPercent}%")
        print(f"[backtest-custom]   → A full stopout ≈ {req.riskPercent}% of current equity (percentBalance mode)")
        if sl_price_series is not None:
            valid_sl = sl_price_series.dropna()
            print(f"[backtest-custom]   → sl_price series: {len(valid_sl)} non-null values out of {len(sl_price_series)} bars")

        trades, equity_curve = simulate_trades(
            df=df,
            entry_long=entry_long, entry_short=entry_short,
            exit_long=exit_long, exit_short=exit_short,
            stop_loss_pct=req.stopLossPercent,
            take_profit_pct=req.takeProfitPercent,
            trade_direction=req.tradeDirection,
            initial_capital=req.initialCapital,
            risk_percent=req.riskPercent,
            risk_type=req.riskType,
            lot_size=req.lotSize,
            stop_loss_pips=req.stopLossPips,
            take_profit_pips=req.takeProfitPips,
            pip_size=pip_size,
            sl_price_series=sl_price_series,
            tp_price_series=tp_price_series,
            execute_on_next_bar=req.executeOnNextBar,
            spread_pips=req.spreadPips,
            slippage_pips=req.slippagePips,
            contract_size=get_contract_size(req.symbol),
        )

        # ── Post-simulation summary ───────────────────────────────────────────────
        if trades:
            first = trades[0]
            print(f"[backtest-custom] Simulation complete: {len(trades)} trades")
            print(f"[backtest-custom]   First trade: entry={first['entryPrice']} exit={first['exitPrice']} type={first['type']} pnlPct={first['pnlPercent']}%")
            print(f"[backtest-custom]   First trade position size: {first['positionSize']:,.2f}")
        else:
            print(f"[backtest-custom] Simulation complete: 0 trades — no signals triggered")

        completed_at = datetime.utcnow()
        return build_result(
            req_id=generate_id(),
            strategy_id=req.strategyId, dataset_id=req.datasetId,
            strategy_name=req.strategyName, dataset_name=req.datasetName,
            trades=trades, equity_curve=equity_curve,
            initial_capital=req.initialCapital, risk_percent=req.riskPercent,
            risk_type=req.riskType, lot_size=req.lotSize,
            stop_loss_pct=req.stopLossPercent, engine_version="custom-python-3.2",
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            duration_ms=round((completed_at - started_at).total_seconds() * 1000, 1),
            execute_on_next_bar=req.executeOnNextBar,
            spread_pips=req.spreadPips,
            slippage_pips=req.slippagePips,
            max_open_positions=req.maxOpenPositions,
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Custom backtest error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")


# ─── POST /walk-forward ────────────────────────────────────────────────────────

@app.post("/walk-forward", response_model=WalkForwardResult)
async def run_walk_forward(req: WalkForwardRequest):
    """
    Walk-forward analysis.
    Splits the dataset into nWindows sequential windows.
    Each window is further split into in-sample (training) and out-of-sample (testing) periods.
    The strategy is run independently on each period — in-sample shows how it would look if
    optimised on that data, out-of-sample shows real-world performance.
    """
    started_at = datetime.utcnow()
    try:
        if len(req.bars) < 120:
            raise HTTPException(status_code=400, detail="Need at least 120 bars for walk-forward analysis")

        # Engine is single-position only. Clamp maxOpenPositions to 1.
        if req.maxOpenPositions != 1:
            print(f"[walk-forward] maxOpenPositions={req.maxOpenPositions} requested but engine supports only 1 open position at a time. Clamping to 1.")
            req.maxOpenPositions = 1

        df = bars_to_df(req.bars)
        n = len(df)
        window_size = n // req.nWindows

        if window_size < 60:
            raise HTTPException(status_code=400, detail="Not enough bars per window. Reduce nWindows or provide more data.")

        # Prefer the caller-supplied pip_size when provided (canonical source of truth).
        # Fall back to the engine's symbol-keyword lookup so older callers continue to work.
        pip_size = req.pip_size if (getattr(req, "pip_size", None) is not None) else (get_pip_size(req.symbol) if req.symbol else 0.0001)
        windows = []

        for w in range(req.nWindows):
            start_idx = w * window_size
            end_idx = start_idx + window_size if w < req.nWindows - 1 else n

            window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
            split_idx = int(len(window_df) * req.inSamplePct)

            in_sample_df = window_df.iloc[:split_idx].reset_index(drop=True)
            out_of_sample_df = window_df.iloc[split_idx:].reset_index(drop=True)

            if len(in_sample_df) < 60 or len(out_of_sample_df) < 10:
                continue

            # Run in-sample
            try:
                el, es, xl, xs, sl_s, tp_s = run_python_signals(req.pythonCode, in_sample_df)
                in_trades, in_equity = simulate_trades(
                    df=in_sample_df, entry_long=el, entry_short=es, exit_long=xl, exit_short=xs,
                    stop_loss_pct=req.stopLossPercent, take_profit_pct=req.takeProfitPercent,
                    trade_direction=req.tradeDirection, initial_capital=req.initialCapital,
                    risk_percent=req.riskPercent, risk_type=req.riskType, lot_size=req.lotSize,
                    stop_loss_pips=req.stopLossPips, take_profit_pips=req.takeProfitPips, pip_size=pip_size,
                    sl_price_series=sl_s, tp_price_series=tp_s,
                    execute_on_next_bar=req.executeOnNextBar,
                    spread_pips=req.spreadPips, slippage_pips=req.slippagePips,
                    contract_size=get_contract_size(req.symbol),
                )
            except Exception:
                in_trades, in_equity = [], [req.initialCapital]

            # Run out-of-sample
            try:
                el, es, xl, xs, sl_s, tp_s = run_python_signals(req.pythonCode, out_of_sample_df)
                out_trades, out_equity = simulate_trades(
                    df=out_of_sample_df, entry_long=el, entry_short=es, exit_long=xl, exit_short=xs,
                    stop_loss_pct=req.stopLossPercent, take_profit_pct=req.takeProfitPercent,
                    trade_direction=req.tradeDirection, initial_capital=req.initialCapital,
                    risk_percent=req.riskPercent, risk_type=req.riskType, lot_size=req.lotSize,
                    stop_loss_pips=req.stopLossPips, take_profit_pips=req.takeProfitPips, pip_size=pip_size,
                    sl_price_series=sl_s, tp_price_series=tp_s,
                    execute_on_next_bar=req.executeOnNextBar,
                    spread_pips=req.spreadPips, slippage_pips=req.slippagePips,
                    contract_size=get_contract_size(req.symbol),
                )
            except Exception:
                out_trades, out_equity = [], [req.initialCapital]

            in_result = build_result(
                req_id=generate_id(), strategy_id=req.strategyId, dataset_id=req.datasetId,
                strategy_name=req.strategyName, dataset_name=f"{req.datasetName} W{w+1} In-Sample",
                trades=in_trades, equity_curve=in_equity,
                initial_capital=req.initialCapital, risk_percent=req.riskPercent,
                risk_type=req.riskType, lot_size=req.lotSize,
                stop_loss_pct=req.stopLossPercent, engine_version="walk-forward-3.2",
            )
            out_result = build_result(
                req_id=generate_id(), strategy_id=req.strategyId, dataset_id=req.datasetId,
                strategy_name=req.strategyName, dataset_name=f"{req.datasetName} W{w+1} Out-of-Sample",
                trades=out_trades, equity_curve=out_equity,
                initial_capital=req.initialCapital, risk_percent=req.riskPercent,
                risk_type=req.riskType, lot_size=req.lotSize,
                stop_loss_pct=req.stopLossPercent, engine_version="walk-forward-3.2",
            )

            windows.append(WalkForwardWindow(
                windowIndex=w + 1,
                inSampleStart=str(in_sample_df["timestamp"].iloc[0]),
                inSampleEnd=str(in_sample_df["timestamp"].iloc[-1]),
                outOfSampleStart=str(out_of_sample_df["timestamp"].iloc[0]),
                outOfSampleEnd=str(out_of_sample_df["timestamp"].iloc[-1]),
                inSampleResult=in_result,
                outOfSampleResult=out_result,
            ))

        if not windows:
            raise HTTPException(status_code=400, detail="No valid windows could be computed.")

        # Aggregate out-of-sample stats
        all_oos_trades = []
        for w in windows:
            all_oos_trades.extend([t.dict() for t in w.outOfSampleResult.trades])

        total_oos_trades = len(all_oos_trades)
        winning_oos = [t for t in all_oos_trades if t["pnl"] > 0]
        combined_win_rate = (len(winning_oos) / total_oos_trades * 100) if total_oos_trades > 0 else 0.0

        oos_net_pcts = [w.outOfSampleResult.netProfitPercent for w in windows]
        combined_net_pct = float(np.mean(oos_net_pcts))

        oos_sharpes = [w.outOfSampleResult.sharpeRatio for w in windows]
        combined_sharpe = float(np.mean(oos_sharpes))

        profitable_windows = sum(1 for w in windows if w.outOfSampleResult.netProfitPercent > 0)
        consistency_score = (profitable_windows / len(windows)) * 100

        completed_at = datetime.utcnow()
        return WalkForwardResult(
            id=generate_id(),
            strategyId=req.strategyId,
            strategyName=req.strategyName,
            datasetName=req.datasetName,
            nWindows=len(windows),
            inSamplePct=req.inSamplePct,
            windows=windows,
            totalOutOfSampleTrades=total_oos_trades,
            combinedWinRate=round(combined_win_rate, 2),
            combinedNetProfitPercent=round(combined_net_pct, 2),
            combinedSharpeRatio=round(combined_sharpe, 2),
            consistencyScore=round(consistency_score, 2),
            createdAt=datetime.utcnow().isoformat(),
            startedAt=started_at.isoformat(),
            completedAt=completed_at.isoformat(),
            durationMs=round((completed_at - started_at).total_seconds() * 1000, 1),
            executeOnNextBar=req.executeOnNextBar,
            spreadPips=req.spreadPips,
            slippagePips=req.slippagePips,
            maxOpenPositions=req.maxOpenPositions,
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Walk-forward error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Walk-forward failed: {str(e)}")


# ─── POST /monte-carlo ─────────────────────────────────────────────────────────

@app.post("/monte-carlo", response_model=MonteCarloResult)
async def run_monte_carlo(req: MonteCarloRequest):
    """
    Monte Carlo simulation.
    Takes the trade list from a completed backtest and resamples trade returns
    to generate thousands of hypothetical equity paths.

    resampleMode:
      "bootstrap" — resample with replacement (standard Monte Carlo, default)
      "shuffle"   — random permutation of the exact trade sequence (no replacement)
    """
    started_at = datetime.utcnow()
    try:
        if len(req.trades) < 5:
            raise HTTPException(status_code=400, detail="Need at least 5 trades to run Monte Carlo simulation")

        # Per-trade fractional returns (pnlPercent is pnl / equity-at-the-time, so
        # compounding equity *= (1 + r) is the correct fractional-return model).
        pnl_pcts = np.array([t.pnlPercent / 100.0 for t in req.trades], dtype=float)
        n_trades = len(pnl_pcts)
        horizon = req.horizonTrades if (req.horizonTrades and req.horizonTrades > 0) else n_trades
        block = req.blockSize if (req.blockSize and req.blockSize > 0) else max(2, int(round(n_trades ** 0.5)))

        rng = np.random.default_rng(42)  # fixed seed for reproducibility

        target_mult   = 1.0 + req.propProfitTargetPct / 100.0
        cap_frac      = req.propMaxDrawdownPct / 100.0
        ruin_floor    = req.ruinThresholdPct / 100.0   # ruined if equity <= start*(1 - this), intra-path

        def resample(gen):
            """Draw `horizon` trade returns under the chosen resample mode."""
            if req.resampleMode == "shuffle":
                # Permutation(s) of the exact sequence — no replacement.
                if horizon <= n_trades:
                    return gen.permutation(pnl_pcts)[:horizon]
                reps = horizon // n_trades + 1
                return np.concatenate([gen.permutation(pnl_pcts) for _ in range(reps)])[:horizon]
            if req.resampleMode == "block":
                # Contiguous blocks (circular) — preserves win/loss streak autocorrelation.
                out = np.empty(horizon)
                filled = 0
                while filled < horizon:
                    start = int(gen.integers(0, n_trades))
                    take = min(block, horizon - filled)
                    for k in range(take):
                        out[filled + k] = pnl_pcts[(start + k) % n_trades]
                    filled += take
                return out
            return gen.choice(pnl_pcts, size=horizon, replace=True)  # bootstrap (default)

        final_equities = np.empty(req.nSimulations)
        max_drawdowns  = np.empty(req.nSimulations)
        ruin_flags     = np.zeros(req.nSimulations, dtype=bool)
        outcomes       = []                    # 'pass' | 'fail' | 'incomplete'
        trades_to_target = []
        all_curves = []

        for s in range(req.nSimulations):
            sampled = resample(rng)
            equity = req.initialCapital
            peak   = equity
            max_dd = 0.0
            ruined = False
            outcome = "incomplete"
            curve = [equity]

            for i, r in enumerate(sampled):
                equity *= (1.0 + r)
                curve.append(round(equity, 2))
                if equity > peak:
                    peak = equity
                dd_trailing = (peak - equity) / peak if peak > 0 else 0.0
                dd_static   = (req.initialCapital - equity) / req.initialCapital
                if dd_trailing * 100 > max_dd:
                    max_dd = dd_trailing * 100
                # Intra-path ruin: equity ever this far below the starting balance.
                if not ruined and dd_static >= ruin_floor:
                    ruined = True
                # Prop challenge: first event wins (breach the DD cap, or hit the target).
                if outcome == "incomplete":
                    breach_dd = dd_trailing if req.propDrawdownMode == "trailing" else dd_static
                    if breach_dd >= cap_frac:
                        outcome = "fail"
                    elif equity >= req.initialCapital * target_mult:
                        outcome = "pass"
                        trades_to_target.append(i + 1)

            final_equities[s] = equity
            max_drawdowns[s]  = max_dd
            ruin_flags[s]     = ruined
            outcomes.append(outcome)
            all_curves.append(curve)

        # Downsample curves to 100 points each for response efficiency
        def downsample(curve, points=100):
            if len(curve) <= points:
                return curve
            indices = np.linspace(0, len(curve) - 1, points, dtype=int)
            return [curve[i] for i in indices]

        # Compute percentile curves
        all_curves_arr = np.array([downsample(c) for c in all_curves])
        percentile_curves = {}
        for level in req.confidenceLevels:
            pct_label = f"p{int(level * 100)}"
            percentile_curves[pct_label] = np.percentile(all_curves_arr, level * 100, axis=0).tolist()

        prob_of_profit = float(np.mean(final_equities > req.initialCapital) * 100)
        prob_of_ruin   = float(np.mean(ruin_flags) * 100)   # INTRA-PATH (not final-equity)
        n = max(1, req.nSimulations)
        pass_pct = outcomes.count("pass") / n * 100
        fail_pct = outcomes.count("fail") / n * 100
        inc_pct  = outcomes.count("incomplete") / n * 100
        median_tt = float(np.median(trades_to_target)) if trades_to_target else 0.0

        completed_at = datetime.utcnow()
        return MonteCarloResult(
            id=generate_id(),
            strategyId=req.strategyId,
            strategyName=req.strategyName,
            nSimulations=req.nSimulations,
            initialCapital=req.initialCapital,
            resampleMode=req.resampleMode,
            medianFinalEquity=round(float(np.median(final_equities)), 2),
            meanFinalEquity=round(float(np.mean(final_equities)), 2),
            worstCaseFinalEquity=round(float(np.percentile(final_equities, 5)), 2),
            bestCaseFinalEquity=round(float(np.percentile(final_equities, 95)), 2),
            probabilityOfProfit=round(prob_of_profit, 2),
            probabilityOfRuin=round(prob_of_ruin, 2),
            ruinThresholdPct=req.ruinThresholdPct,
            medianMaxDrawdown=round(float(np.median(max_drawdowns)), 2),
            worstCaseMaxDrawdown=round(float(np.percentile(max_drawdowns, 95)), 2),
            propProfitTargetPct=req.propProfitTargetPct,
            propMaxDrawdownPct=req.propMaxDrawdownPct,
            propPassProbability=round(pass_pct, 2),
            propFailDrawdown=round(fail_pct, 2),
            propIncomplete=round(inc_pct, 2),
            medianTradesToTarget=round(median_tt, 1),
            percentileCurves=percentile_curves,
            createdAt=datetime.utcnow().isoformat(),
            startedAt=started_at.isoformat(),
            completedAt=completed_at.isoformat(),
            durationMs=round((completed_at - started_at).total_seconds() * 1000, 1),
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Monte Carlo error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Monte Carlo failed: {str(e)}")


# ─── POST /optimize ────────────────────────────────────────────────────────────

@app.post("/optimize", response_model=OptimizeResult)
async def run_optimize(req: OptimizeRequest):
    """
    Parameter optimisation via grid search.
    The pythonCode must use variables named exactly as the keys in paramGrid.
    Example: if paramGrid = {"fast_period": [10, 20], "slow_period": [50, 100]}
    then the Python code should reference fast_period and slow_period directly.
    These values are injected into the code's namespace before get_signals() is called.
    """
    started_at = datetime.utcnow()
    try:
        # Resolve bars from cache or inline
        if not req.bars and req.cache_key:
            if req.cache_key not in _bar_cache:
                raise HTTPException(status_code=400, detail="Bar cache miss — upload bars first (code: CACHE_MISS)")
            bars = _bar_cache[req.cache_key]
            _bar_cache.move_to_end(req.cache_key)  # refresh LRU
        elif req.bars:
            bars = req.bars
        else:
            raise HTTPException(status_code=400, detail="bars or cache_key required")

        ohlc_bars = [OHLCBar(**b) if isinstance(b, dict) else b for b in bars]

        if len(ohlc_bars) < 60:
            raise HTTPException(status_code=400, detail="Need at least 60 bars to optimise")

        # Engine is single-position only. Clamp maxOpenPositions to 1.
        if req.maxOpenPositions != 1:
            print(f"[optimize] maxOpenPositions={req.maxOpenPositions} requested but engine supports only 1 open position at a time. Clamping to 1.")
            req.maxOpenPositions = 1

        # Build all parameter combinations
        import itertools
        keys = list(req.paramGrid.keys())
        values = list(req.paramGrid.values())
        all_combos = list(itertools.product(*values))

        # Track capping metadata
        total_grid_combinations = len(all_combos)
        was_capped = total_grid_combinations > req.maxCombinations
        sampling_method = "random_sample" if was_capped else "full_grid"

        if was_capped:
            # Deterministic random subset — same inputs always produce the same subset
            rng_sample = np.random.default_rng(42)
            indices = sorted(rng_sample.choice(total_grid_combinations, size=req.maxCombinations, replace=False).tolist())
            all_combos = [all_combos[i] for i in indices]

        df = bars_to_df(ohlc_bars)
        # Prefer the caller-supplied pip_size when provided (canonical source of truth).
        # Fall back to the engine's symbol-keyword lookup so older callers continue to work.
        pip_size = req.pip_size if (getattr(req, "pip_size", None) is not None) else (get_pip_size(req.symbol) if req.symbol else 0.0001)
        results = []
        tested_combinations = 0   # counts every combo attempted, regardless of outcome

        # ── Walk-forward mode: score each combo by avg out-of-sample performance ──
        if req.walkForwardEnabled:
            n = len(df)
            window_size = n // req.nWindows
            if window_size < 30:
                raise HTTPException(status_code=400, detail="Not enough bars per window. Reduce nWindows or provide more data.")

            # Pre-build window slices
            wf_windows = []
            for w in range(req.nWindows):
                start_idx = w * window_size
                end_idx = start_idx + window_size if w < req.nWindows - 1 else n
                window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
                split_idx = int(len(window_df) * req.inSamplePct)
                oos_df = window_df.iloc[split_idx:].reset_index(drop=True)
                if len(oos_df) >= 10:
                    wf_windows.append(oos_df)

            if not wf_windows:
                raise HTTPException(status_code=400, detail="No usable out-of-sample windows. Reduce nWindows or increase inSamplePct.")

            for combo in all_combos:
                params = dict(zip(keys, combo))
                tested_combinations += 1
                oos_metrics_list = []

                for oos_df in wf_windows:
                    try:
                        el, es, xl, xs, sl_s, tp_s = run_python_signals(req.pythonCode, oos_df, extra_params=params)
                        trades, equity_curve = simulate_trades(
                            df=oos_df, entry_long=el, entry_short=es, exit_long=xl, exit_short=xs,
                            stop_loss_pct=req.stopLossPercent, take_profit_pct=req.takeProfitPercent,
                            trade_direction=req.tradeDirection,
                            initial_capital=req.initialCapital,
                            risk_percent=req.riskPercent, risk_type=req.riskType, lot_size=req.lotSize,
                            stop_loss_pips=req.stopLossPips, take_profit_pips=req.takeProfitPips, pip_size=pip_size,
                            sl_price_series=sl_s, tp_price_series=tp_s,
                            execute_on_next_bar=req.executeOnNextBar,
                            spread_pips=req.spreadPips, slippage_pips=req.slippagePips,
                            contract_size=get_contract_size(req.symbol),
                        )
                        if trades:
                            oos_metrics_list.append(compute_metrics(trades, equity_curve, req.initialCapital))
                    except Exception:
                        continue

                if not oos_metrics_list:
                    continue

                # Average the OOS metrics across all windows
                def avg(key):
                    return sum(m[key] for m in oos_metrics_list) / len(oos_metrics_list)

                avg_metrics = {
                    "netProfitPercent": avg("netProfitPercent"),
                    "winRate": avg("winRate"),
                    "sharpeRatio": avg("sharpeRatio"),
                    "profitFactor": avg("profitFactor"),
                    "maxDrawdownPercent": avg("maxDrawdownPercent"),
                    "totalTrades": int(avg("totalTrades")),
                    "winningTrades": int(avg("winningTrades")),
                    "losingTrades": int(avg("losingTrades")),
                    "consecutiveWins": int(avg("consecutiveWins")),
                    "consecutiveLosses": int(avg("consecutiveLosses")),
                }
                score = compute_score(
                    avg_metrics["winRate"], avg_metrics["profitFactor"],
                    avg_metrics["maxDrawdownPercent"], avg_metrics["sharpeRatio"],
                    avg_metrics["totalTrades"], avg_metrics["netProfitPercent"]
                )
                results.append(OptimizeResultRow(
                    params=params,
                    netProfitPercent=avg_metrics["netProfitPercent"],
                    winRate=avg_metrics["winRate"],
                    sharpeRatio=avg_metrics["sharpeRatio"],
                    profitFactor=avg_metrics["profitFactor"],
                    maxDrawdownPercent=avg_metrics["maxDrawdownPercent"],
                    totalTrades=avg_metrics["totalTrades"],
                    winningTrades=avg_metrics["winningTrades"],
                    losingTrades=avg_metrics["losingTrades"],
                    consecutiveWins=avg_metrics["consecutiveWins"],
                    consecutiveLosses=avg_metrics["consecutiveLosses"],
                    score=score,
                    grade=grade_from_score(score),
                ))

        # ── Standard mode: score each combo on the full dataset ──────────────────
        else:
            for combo in all_combos:
                params = dict(zip(keys, combo))
                tested_combinations += 1

                try:
                    el, es, xl, xs, sl_s, tp_s = run_python_signals(req.pythonCode, df, extra_params=params)
                    trades, equity_curve = simulate_trades(
                        df=df, entry_long=el, entry_short=es, exit_long=xl, exit_short=xs,
                        stop_loss_pct=req.stopLossPercent, take_profit_pct=req.takeProfitPercent,
                        trade_direction=req.tradeDirection,
                        initial_capital=req.initialCapital,
                        risk_percent=req.riskPercent, risk_type=req.riskType, lot_size=req.lotSize,
                        stop_loss_pips=req.stopLossPips, take_profit_pips=req.takeProfitPips, pip_size=pip_size,
                        sl_price_series=sl_s, tp_price_series=tp_s,
                        execute_on_next_bar=req.executeOnNextBar,
                        spread_pips=req.spreadPips, slippage_pips=req.slippagePips,
                        contract_size=get_contract_size(req.symbol),
                    )
                except Exception:
                    continue

                if not trades:
                    continue

                metrics = compute_metrics(trades, equity_curve, req.initialCapital)
                score = compute_score(
                    metrics["winRate"], metrics["profitFactor"],
                    metrics["maxDrawdownPercent"], metrics["sharpeRatio"],
                    metrics["totalTrades"], metrics["netProfitPercent"]
                )

                results.append(OptimizeResultRow(
                    params=params,
                    netProfitPercent=metrics["netProfitPercent"],
                    winRate=metrics["winRate"],
                    sharpeRatio=metrics["sharpeRatio"],
                    profitFactor=metrics["profitFactor"],
                    maxDrawdownPercent=metrics["maxDrawdownPercent"],
                    totalTrades=metrics["totalTrades"],
                    winningTrades=metrics["winningTrades"],
                    losingTrades=metrics["losingTrades"],
                    consecutiveWins=metrics["consecutiveWins"],
                    consecutiveLosses=metrics["consecutiveLosses"],
                    score=score,
                    grade=grade_from_score(score),
                ))

        if not results:
            raise HTTPException(status_code=400, detail="No valid parameter combinations produced trades.")

        # Sort by chosen metric, with optional secondary objective as weighted tiebreaker
        def get_metric(r, objective: str) -> float:
            return {
                "sharpe": r.sharpeRatio,
                "netProfit": r.netProfitPercent,
                "winRate": r.winRate,
                "profitFactor": r.profitFactor,
                "maxDrawdown": -r.maxDrawdownPercent,  # lower DD is better, negate so higher = better
                "returnDrawdown": (r.netProfitPercent / r.maxDrawdownPercent) if r.maxDrawdownPercent > 0 else (r.netProfitPercent if r.netProfitPercent > 0 else 0.0),
            }.get(objective, r.sharpeRatio)

        if req.secondaryObjective and req.secondaryObjective != req.optimizeFor:
            # Normalise both metrics across the result set (min-max to 0-1),
            # then blend 70% primary / 30% secondary so ranking reflects both objectives
            primary_vals = [get_metric(r, req.optimizeFor) for r in results]
            secondary_vals = [get_metric(r, req.secondaryObjective) for r in results]

            def normalise(vals):
                lo, hi = min(vals), max(vals)
                spread = hi - lo
                if spread == 0:
                    return [0.5] * len(vals)
                return [(v - lo) / spread for v in vals]

            p_norm = normalise(primary_vals)
            s_norm = normalise(secondary_vals)
            blended = [0.70 * p + 0.30 * s for p, s in zip(p_norm, s_norm)]
            results = [r for _, r in sorted(zip(blended, results), key=lambda x: x[0], reverse=True)]
        else:
            results.sort(key=lambda r: get_metric(r, req.optimizeFor), reverse=True)
        best = results[0]

        completed_at = datetime.utcnow()
        return OptimizeResult(
            id=generate_id(),
            strategyId=req.strategyId,
            strategyName=req.strategyName,
            datasetName=req.datasetName,
            nCombinations=len(results),
            optimizeFor=req.optimizeFor,
            bestParams=best.params,
            bestResult=best,
            allResults=results,
            totalGridCombinations=total_grid_combinations,
            testedCombinations=tested_combinations,      # combos actually attempted
            wasCapped=was_capped,
            samplingMethod=sampling_method,
            createdAt=datetime.utcnow().isoformat(),
            startedAt=started_at.isoformat(),
            completedAt=completed_at.isoformat(),
            durationMs=round((completed_at - started_at).total_seconds() * 1000, 1),
            executeOnNextBar=req.executeOnNextBar,
            spreadPips=req.spreadPips,
            slippagePips=req.slippagePips,
            maxOpenPositions=req.maxOpenPositions,
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Optimise error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Optimisation failed: {str(e)}")


# ─── POST /optimize-async — Submit async optimisation job ──────────────────────

class OptimizeAsyncJobResponse(BaseModel):
    job_id: str


class OptimizeStatusResponse(BaseModel):
    job_id: str
    status: str  # "pending" | "running" | "done" | "error" | "cancelled"
    testedCombinations: int = 0
    totalCombinations: int = 0
    progressPct: int = 0
    result: Optional[Any] = None
    errorMessage: Optional[str] = None


@app.post("/optimize-async", response_model=OptimizeAsyncJobResponse)
async def run_optimize_async(req: OptimizeRequest):
    """
    Submit an optimisation job that runs in the background.
    Returns a job_id immediately. Poll /optimize-status/{job_id} to track progress.
    The result is stored on disk so it survives a server restart.
    """
    # Resolve bars from cache or inline
    if not req.bars and req.cache_key:
        if req.cache_key not in _bar_cache:
            raise HTTPException(status_code=400, detail="Bar cache miss — upload bars first (code: CACHE_MISS)")
        bars = _bar_cache[req.cache_key]
        _bar_cache.move_to_end(req.cache_key)  # refresh LRU
    elif req.bars:
        bars = req.bars
    else:
        raise HTTPException(status_code=400, detail="bars or cache_key required")

    ohlc_bars = [OHLCBar(**b) if isinstance(b, dict) else b for b in bars]

    if len(ohlc_bars) < 60:
        raise HTTPException(status_code=400, detail="Need at least 60 bars to optimise")

    if req.maxOpenPositions != 1:
        req.maxOpenPositions = 1

    # Store the resolved bars back onto the request so _run_optimize_bg can use req.bars
    req.bars = ohlc_bars

    job_id = str(uuid.uuid4())
    cancel_event = threading.Event()
    _opt_jobs[job_id] = {
        "status": "pending",
        "tested": 0,
        "total": 0,
        "progress": 0,
        "result": None,
        "error": None,
        "cancel_event": cancel_event,
    }
    _save_job_disk(job_id)

    thread = threading.Thread(target=_run_optimize_bg, args=(job_id, req), daemon=True)
    thread.start()
    print(f"[optimize-async] Submitted job {job_id}")
    return OptimizeAsyncJobResponse(job_id=job_id)


@app.get("/optimize-status/{job_id}", response_model=OptimizeStatusResponse)
async def get_optimize_status(job_id: str):
    """
    Poll the status of an async optimisation job.
    Returns progress and, when done, the full result.
    Falls back to disk if the job is no longer in memory (e.g. after a restart).
    """
    job = _opt_jobs.get(job_id)
    if job:
        return OptimizeStatusResponse(
            job_id=job_id,
            status=job["status"],
            testedCombinations=job["tested"],
            totalCombinations=job["total"],
            progressPct=job["progress"],
            result=job["result"],
            errorMessage=job["error"],
        )

    # Fallback: read from disk (job completed or VPS restarted)
    disk_path = os.path.join(JOBS_DIR, f"{job_id}.json")
    if os.path.exists(disk_path):
        try:
            with open(disk_path) as f:
                data = json.load(f)
            return OptimizeStatusResponse(**data)
        except Exception:
            pass

    raise HTTPException(status_code=404, detail=f"Job {job_id} not found")


@app.delete("/optimize/{job_id}")
async def cancel_optimize(job_id: str):
    """
    Cancel a running optimisation job.
    Sets a cancel flag that the background thread checks between combinations.
    The job will stop at the next combination boundary and be marked 'cancelled'.
    """
    job = _opt_jobs.get(job_id)
    if not job:
        disk_path = os.path.join(JOBS_DIR, f"{job_id}.json")
        if not os.path.exists(disk_path):
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        return {"cancelled": False, "message": "Job already completed or not in memory"}

    job["cancel_event"].set()
    print(f"[optimize-async] Cancel requested for job {job_id}")
    return {"cancelled": True, "message": "Cancellation requested — job will stop at next combination"}


# ─── LEGACY: POST /backtest (fixed indicator-based) ────────────────────────────

def compute_indicators(df: pd.DataFrame, spec: SignalSpec) -> dict:
    ind = {}
    close = df["close"]
    high = df["high"]
    low = df["low"]

    conditions = [spec.entryLong, spec.entryShort, spec.exitLong, spec.exitShort]
    conditions = [c for c in conditions if c is not None]

    for cond in conditions:
        p = cond.params
        name = cond.indicator

        if name in ("ema_cross", "price_above_ema", "price_below_ema"):
            fast = int(p.get("fast", p.get("period", 20)))
            slow = int(p.get("slow", p.get("period", 50)))
            for period in set([fast, slow]):
                key = f"ema_{period}"
                if key not in ind:
                    ind[key] = ta.ema(close, length=period)
        elif name == "sma_cross":
            fast = int(p.get("fast", 20))
            slow = int(p.get("slow", 50))
            for period in set([fast, slow]):
                key = f"sma_{period}"
                if key not in ind:
                    ind[key] = ta.sma(close, length=period)
        elif name in ("rsi_oversold", "rsi_overbought"):
            period = int(p.get("period", 14))
            key = f"rsi_{period}"
            if key not in ind:
                ind[key] = ta.rsi(close, length=period)
        elif name == "macd_cross":
            fast = int(p.get("fast", 12))
            slow = int(p.get("slow", 26))
            signal = int(p.get("signal", 9))
            if "macd" not in ind:
                macd_result = ta.macd(close, fast=fast, slow=slow, signal=signal)
                if macd_result is not None and not macd_result.empty:
                    ind["macd_line"] = macd_result.iloc[:, 0]
                    ind["macd_signal"] = macd_result.iloc[:, 1]
                    ind["macd_hist"] = macd_result.iloc[:, 2]
        elif name in ("bollinger_breakout", "bollinger_mean_revert"):
            period = int(p.get("period", 20))
            std = float(p.get("std", 2.0))
            key = f"bb_{period}_{std}"
            if key not in ind:
                bb = ta.bbands(close, length=period, std=std)
                if bb is not None and not bb.empty:
                    ind[f"{key}_upper"] = bb.iloc[:, 0]
                    ind[f"{key}_mid"] = bb.iloc[:, 1]
                    ind[f"{key}_lower"] = bb.iloc[:, 2]
        elif name == "stoch_cross":
            k = int(p.get("k", 14))
            d = int(p.get("d", 3))
            key = f"stoch_{k}_{d}"
            if key not in ind:
                stoch = ta.stoch(high, low, close, k=k, d=d)
                if stoch is not None and not stoch.empty:
                    ind[f"{key}_k"] = stoch.iloc[:, 0]
                    ind[f"{key}_d"] = stoch.iloc[:, 1]
        elif name == "breakout_high":
            period = int(p.get("period", 20))
            key = f"highest_{period}"
            if key not in ind:
                ind[key] = high.rolling(window=period).max().shift(1)
        elif name == "breakout_low":
            period = int(p.get("period", 20))
            key = f"lowest_{period}"
            if key not in ind:
                ind[key] = low.rolling(window=period).min().shift(1)

    if "ema_20" not in ind:
        ind["ema_20"] = ta.ema(close, length=20)
    if "ema_50" not in ind:
        ind["ema_50"] = ta.ema(close, length=50)

    return ind


def generate_fixed_signals(df, ind, cond, direction):
    p = cond.params
    name = cond.indicator
    close = df["close"]
    n = len(df)
    result = pd.Series([False] * n, index=df.index)

    try:
        if name == "ema_cross":
            fast_p = int(p.get("fast", 20)); slow_p = int(p.get("slow", 50))
            fast = ind.get(f"ema_{fast_p}"); slow = ind.get(f"ema_{slow_p}")
            if fast is None or slow is None: return result
            result = (fast.shift(1) <= slow.shift(1)) & (fast > slow) if direction == "long" else (fast.shift(1) >= slow.shift(1)) & (fast < slow)
        elif name == "sma_cross":
            fast_p = int(p.get("fast", 20)); slow_p = int(p.get("slow", 50))
            fast = ind.get(f"sma_{fast_p}"); slow = ind.get(f"sma_{slow_p}")
            if fast is None or slow is None: return result
            result = (fast.shift(1) <= slow.shift(1)) & (fast > slow) if direction == "long" else (fast.shift(1) >= slow.shift(1)) & (fast < slow)
        elif name == "rsi_oversold":
            period = int(p.get("period", 14)); threshold = float(p.get("threshold", 30))
            rsi = ind.get(f"rsi_{period}")
            if rsi is None: return result
            result = (rsi.shift(1) <= threshold) & (rsi > threshold) if direction == "long" else (rsi.shift(1) >= threshold) & (rsi < threshold)
        elif name == "rsi_overbought":
            period = int(p.get("period", 14)); threshold = float(p.get("threshold", 70))
            rsi = ind.get(f"rsi_{period}")
            if rsi is None: return result
            result = (rsi.shift(1) >= threshold) & (rsi > threshold) if direction == "long" else (rsi.shift(1) >= threshold) & (rsi < threshold)
        elif name == "macd_cross":
            macd = ind.get("macd_line"); sig = ind.get("macd_signal")
            if macd is None or sig is None: return result
            result = (macd.shift(1) <= sig.shift(1)) & (macd > sig) if direction == "long" else (macd.shift(1) >= sig.shift(1)) & (macd < sig)
        elif name == "bollinger_breakout":
            period = int(p.get("period", 20)); std = float(p.get("std", 2.0)); key = f"bb_{period}_{std}"
            upper = ind.get(f"{key}_upper"); lower = ind.get(f"{key}_lower")
            if upper is None or lower is None: return result
            result = (close.shift(1) <= upper.shift(1)) & (close > upper) if direction == "long" else (close.shift(1) >= lower.shift(1)) & (close < lower)
        elif name == "bollinger_mean_revert":
            period = int(p.get("period", 20)); std = float(p.get("std", 2.0)); key = f"bb_{period}_{std}"
            upper = ind.get(f"{key}_upper"); lower = ind.get(f"{key}_lower")
            if upper is None or lower is None: return result
            result = (close.shift(1) <= lower.shift(1)) & (close > lower) if direction == "long" else (close.shift(1) >= upper.shift(1)) & (close < upper)
        elif name == "price_above_ema":
            period = int(p.get("period", 20)); ema = ind.get(f"ema_{period}")
            if ema is None: return result
            result = (close.shift(1) <= ema.shift(1)) & (close > ema) if direction == "long" else (close.shift(1) >= ema.shift(1)) & (close < ema)
        elif name == "price_below_ema":
            period = int(p.get("period", 20)); ema = ind.get(f"ema_{period}")
            if ema is None: return result
            result = (close.shift(1) >= ema.shift(1)) & (close > ema) if direction == "long" else (close.shift(1) <= ema.shift(1)) & (close < ema)
        elif name == "stoch_cross":
            k_p = int(p.get("k", 14)); d_p = int(p.get("d", 3)); key = f"stoch_{k_p}_{d_p}"
            k = ind.get(f"{key}_k"); d = ind.get(f"{key}_d")
            if k is None or d is None: return result
            result = (k.shift(1) <= d.shift(1)) & (k > d) if direction == "long" else (k.shift(1) >= d.shift(1)) & (k < d)
        elif name == "breakout_high":
            period = int(p.get("period", 20)); highest = ind.get(f"highest_{period}")
            if highest is None: return result
            result = (close.shift(1) <= highest.shift(1)) & (close > highest) if direction == "long" else close < highest
        elif name == "breakout_low":
            period = int(p.get("period", 20)); lowest = ind.get(f"lowest_{period}")
            if lowest is None: return result
            result = close > lowest if direction == "long" else (close.shift(1) >= lowest.shift(1)) & (close < lowest)
    except Exception as e:
        print(f"Signal generation error for {name}: {e}")
        return pd.Series([False] * n, index=df.index)

    return result.fillna(False)


@app.post("/backtest", response_model=BacktestResult)
async def run_backtest(req: BacktestRequest):
    started_at = datetime.utcnow()
    try:
        if len(req.bars) < 60:
            raise HTTPException(status_code=400, detail="Need at least 60 bars to run a backtest")

        df = bars_to_df(req.bars)
        spec = req.signalSpec
        ind = compute_indicators(df, spec)
        false_series = pd.Series([False] * len(df), index=df.index)

        entry_long = generate_fixed_signals(df, ind, spec.entryLong, "long") if spec.entryLong else false_series.copy()
        entry_short = generate_fixed_signals(df, ind, spec.entryShort, "short") if spec.entryShort else false_series.copy()
        exit_long = generate_fixed_signals(df, ind, spec.exitLong, "short") if spec.exitLong else false_series.copy()
        exit_short = generate_fixed_signals(df, ind, spec.exitShort, "long") if spec.exitShort else false_series.copy()

        # Legacy fixed-indicator path. Intentionally simpler than /backtest-custom:
        # does not pass executeOnNextBar, spread/slippage, or dynamic sl_price/tp_price.
        # All three default to their safe values (execute_on_next_bar=True, costs=0,
        # dynamic mode off) inside simulate_trades. Do not change this without also
        # updating the legacy BacktestRequest model and the TypeScript caller.
        trades, equity_curve = simulate_trades(
            df=df, entry_long=entry_long, entry_short=entry_short,
            exit_long=exit_long, exit_short=exit_short,
            stop_loss_pct=spec.stopLossPercent, take_profit_pct=spec.takeProfitPercent,
            trade_direction=spec.tradeDirection, initial_capital=req.initialCapital,
            risk_percent=req.riskPercent, risk_type=req.riskType, lot_size=req.lotSize,
            contract_size=get_contract_size(req.symbol),
        )
        trades.sort(key=lambda t: t["entryDate"])

        completed_at = datetime.utcnow()
        return build_result(
            req_id=generate_id(), strategy_id=req.strategyId, dataset_id=req.datasetId,
            strategy_name=req.strategyName, dataset_name=req.datasetName,
            trades=trades, equity_curve=equity_curve,
            initial_capital=req.initialCapital, risk_percent=req.riskPercent,
            risk_type=req.riskType, lot_size=req.lotSize,
            stop_loss_pct=spec.stopLossPercent, engine_version="fixed-indicators-3.0",
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            duration_ms=round((completed_at - started_at).total_seconds() * 1000, 1),
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Backtest error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")


# ─── Health Check ──────────────────────────────────────────────────────────────

@app.post("/cache-bars")
async def cache_bars(request: Request):
    body = await request.json()
    cache_key = body.get("cacheKey")
    bars = body.get("bars")

    if not cache_key or not bars:
        return JSONResponse({"error": "cacheKey and bars required"}, status_code=400)

    # If already cached, just confirm (no-op)
    if cache_key in _bar_cache:
        _bar_cache.move_to_end(cache_key)  # refresh LRU order
        return JSONResponse({"cached": True, "cacheKey": cache_key,
                             "bars": len(_bar_cache[cache_key])})

    # Evict oldest if at capacity
    if len(_bar_cache) >= BAR_CACHE_MAX_SIZE:
        _bar_cache.popitem(last=False)

    _bar_cache[cache_key] = bars
    return JSONResponse({"cached": True, "cacheKey": cache_key,
                         "bars": len(bars)})


@app.get("/cache-bars/{cache_key}")
async def check_cache(cache_key: str):
    if cache_key in _bar_cache:
        return JSONResponse({"exists": True, "bars": len(_bar_cache[cache_key])})
    return JSONResponse({"exists": False})


@app.post("/update")
async def update_engine(request: Request):
    """Receive new engine content from backend and restart, or pull from GitHub as fallback."""
    current_file = os.path.abspath(__file__)

    try:
        body = await request.json()
        pushed_content: str | None = body.get("content") if isinstance(body, dict) else None
    except Exception:
        pushed_content = None

    if pushed_content:
        new_content = pushed_content
    else:
        GITHUB_URL = "https://raw.githubusercontent.com/XAU30Dynamics/sd-engine/main/main.py"
        try:
            with urllib.request.urlopen(GITHUB_URL, timeout=30) as resp:
                new_content = resp.read().decode("utf-8")
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Failed to download update: {str(e)}")

    match = re.search(r'version="([^"]+)"', new_content)
    new_version = match.group(1) if match else "unknown"

    try:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".py", dir=os.path.dirname(current_file))
        os.close(tmp_fd)
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        shutil.move(tmp_path, current_file)
    except Exception as e:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Failed to write update: {str(e)}")

    def _exit_soon():
        time.sleep(1)
        os._exit(0)

    threading.Thread(target=_exit_soon, daemon=True).start()
    return {"status": "ok", "version": new_version, "message": "Engine updated. Restarting…"}


@app.get("/health")
async def health():
    # Read version from the FastAPI app object so there's only one source of truth.
    # Previously this was a hardcoded literal that drifted from the FastAPI title
    # version, silently breaking the in-app Update Engine flow (Settings would keep
    # showing the stale literal even after a successful auto-update).
    return {
        "status": "ok",
        "version": app.version,
        "endpoints": ["/backtest", "/backtest-custom", "/walk-forward", "/monte-carlo", "/optimize", "/optimize-async", "/optimize-status/{id}", "DELETE /optimize/{id}"],
        "libraries": {
            "pandas": pd.__version__,
            "pandas_ta": ta.version if hasattr(ta, "version") else "installed",
            "numpy": np.__version__,
            "vectorbt": vbt.__version__ if hasattr(vbt, "__version__") else "installed",
        }
    }


@app.get("/")
async def root():
    return {
        "message": "AlgoTrader VPS Backtest Engine v3.3 is running.",
        "endpoints": {
            "POST /backtest": "Fixed indicator-based backtest (legacy)",
            "POST /backtest-custom": "AI-generated Python signal code backtest",
            "POST /walk-forward": "Walk-forward analysis (rolling in/out of sample windows)",
            "POST /monte-carlo": "Monte Carlo simulation on trade results",
            "POST /optimize": "Parameter optimisation via grid search (synchronous)",
            "POST /optimize-async": "Submit async optimisation job — returns job_id immediately",
            "GET /optimize-status/{id}": "Poll async optimisation job status and progress",
            "DELETE /optimize/{id}": "Cancel a running async optimisation job",
            "GET /health": "Health check",
        }
    }
