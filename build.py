"""Swing Scout data builder.

Runs on GitHub Actions after each US close. Pulls free end-of-day data
(yfinance, with Wikipedia for index membership), computes indicators and
setup flags, and writes files that the swing-scout skill reads.

Usage: python build.py --out out --prev prev
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone, date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# --------------------------------------------------------------------------
# Indicators (pure functions on a single OHLCV frame; index = dates ascending)
# --------------------------------------------------------------------------

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(dn != 0, 100.0)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def pct(a, b):
    if b is None or a is None or b == 0 or pd.isna(a) or pd.isna(b):
        return np.nan
    return (a / b - 1) * 100


def ret(close: pd.Series, n: int):
    return pct(close.iloc[-1], close.iloc[-1 - n]) if len(close) > n else np.nan


def features(df: pd.DataFrame, spy: pd.Series | None) -> dict:
    """All per-ticker numbers. Percentages are in % units."""
    df = df.dropna(subset=["Close"])
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    n = len(df)
    f: dict = {"date": df.index[-1].strftime("%Y-%m-%d"), "rows": n, "close": round(float(c.iloc[-1]), 4)}
    if n < 60:
        f["note"] = "short_history"
    for k in (1, 5, 21, 63, 126, 252):
        f[f"ret_{k}d"] = ret(c, k)
    f["mom_12_1"] = pct(c.iloc[-22], c.iloc[-253]) if n > 252 else np.nan

    win = min(n, 252)
    hi52, lo52 = h.iloc[-win:].max(), l.iloc[-win:].min()
    f["hi_52w"], f["lo_52w"] = hi52, lo52
    f["pct_from_hi"] = pct(c.iloc[-1], hi52)
    f["pct_from_lo"] = pct(c.iloc[-1], lo52)
    f["days_since_hi"] = int(n - 1 - int(np.argmax(h.iloc[-win:].values)) - (n - win))

    for m in (10, 20, 50, 150, 200):
        f[f"sma{m}"] = c.rolling(m).mean().iloc[-1] if n >= m else np.nan
        f[f"dist_sma{m}"] = pct(c.iloc[-1], f[f"sma{m}"])
    s50, s200 = c.rolling(50).mean(), c.rolling(200).mean()
    f["sma50_slope_10d"] = pct(s50.iloc[-1], s50.iloc[-11]) if n >= 61 else np.nan
    f["sma200_slope_20d"] = pct(s200.iloc[-1], s200.iloc[-21]) if n >= 221 else np.nan

    f["rsi14"] = rsi(c).iloc[-1]
    a14 = atr(df, 14)
    f["atr14"] = a14.iloc[-1]
    f["atr_pct"] = f["atr14"] / c.iloc[-1] * 100 if f["atr14"] == f["atr14"] else np.nan
    a10, a50 = atr(df, 10).iloc[-1], atr(df, 50).iloc[-1] if n >= 60 else np.nan
    f["atr_contraction"] = a10 / a50 if a50 and a50 == a50 else np.nan   # <0.8 = volatility drying up

    avg50v = v.iloc[-51:-1].mean() if n > 51 else v.mean()
    f["vol_ratio"] = v.iloc[-1] / avg50v if avg50v else np.nan
    f["dollar_vol_50d_m"] = (c * v).iloc[-50:].mean() / 1e6

    # Base geometry over the last 8 weeks (excluding today)
    if n > 41:
        hh, ll = h.iloc[-41:-1].max(), l.iloc[-41:-1].min()
        f["base_depth_40d"] = (hh - ll) / hh * 100
        f["pivot_50d"] = h.iloc[-51:-1].max() if n > 51 else hh
        f["dist_pivot"] = pct(c.iloc[-1], f["pivot_50d"])
        f["tight_10d"] = c.pct_change().iloc[-10:].std() * 100
    # Stabilization after a drop: higher low and reclaim of the 20-day
    if n > 25:
        recent_low, prior_low = l.iloc[-5:].min(), l.iloc[-25:-5].min()
        f["higher_low"] = bool(recent_low > prior_low)
        f["new_low_5d"] = bool(l.iloc[-5:].min() <= l.iloc[-win:].min() + 1e-9)
    # Weekly trend
    wk = c.resample("W-FRI").last().dropna()
    if len(wk) >= 31:
        f["wk_above_10w"] = bool(wk.iloc[-1] > wk.rolling(10).mean().iloc[-1])
        f["wk_above_30w"] = bool(wk.iloc[-1] > wk.rolling(30).mean().iloc[-1])
        f["wk_30w_rising"] = bool(wk.rolling(30).mean().iloc[-1] > wk.rolling(30).mean().iloc[-5])
    # Gap-and-hold in the last 30 sessions (usually earnings; skill confirms)
    if n > 60:
        gaps = (df["Open"] / c.shift() - 1) * 100
        volr = v / v.rolling(50).mean().shift()
        idx = [i for i in range(n - 30, n) if gaps.iloc[i] >= 5 and volr.iloc[i] >= 2.5]
        if idx:
            i = idx[-1]
            f["gap_date"] = df.index[i].strftime("%Y-%m-%d")
            f["gap_pct"] = gaps.iloc[i]
            f["gap_holding"] = bool(l.iloc[i:].min() >= l.iloc[i] * 0.99)
        dn = [i for i in range(n - 30, n) if gaps.iloc[i] <= -5 and volr.iloc[i] >= 2.5]
        if dn:
            f["gapdown_date"] = df.index[dn[-1]].strftime("%Y-%m-%d")
            f["gapdown_pct"] = gaps.iloc[dn[-1]]
    if spy is not None:
        s = spy.reindex(c.index).ffill()
        for k in (21, 63, 126):
            f[f"rs_vs_spy_{k}d"] = ret(c, k) - ret(s, k) if n > k else np.nan
        f["rs_line_new_high"] = bool(n > 252 and (c / s).iloc[-1] >= (c / s).iloc[-252:].max() * 0.995)
    return f


def flags(f: dict) -> list[str]:
    """Setup flags. Definitions are mirrored in the skill's Setups rulebook."""
    g = lambda k: f.get(k, np.nan)
    ok = lambda x: x is not None and x == x
    yes = lambda k: ok(f.get(k)) and bool(f.get(k))   # NaN/None -> False (NaN is truthy in Python!)
    if f.get("stale"):
        return []
    out = []
    trend_ok = ok(g("sma50")) and ok(g("sma200")) and g("sma50") > g("sma200") and (g("sma200_slope_20d") or 0) > 0
    uptrend = trend_ok and g("close") > g("sma50")
    # 1. Breakout from a base on volume
    if uptrend and ok(g("pivot_50d")) and g("close") > g("pivot_50d") and g("vol_ratio") >= 1.5 and g("base_depth_40d") <= 30:
        out.append("BREAKOUT")
    # 2. Coiling just under a pivot (watch for breakout)
    elif uptrend and ok(g("dist_pivot")) and -5 <= g("dist_pivot") <= 0 and g("base_depth_40d") <= 25 and g("atr_contraction") < 0.85:
        out.append("PIVOT_WATCH")
    # 3. Pullback to rising 50-day in an uptrend
    if trend_ok and ok(g("dist_sma50")) and -3 <= g("dist_sma50") <= 3 and -20 <= g("pct_from_hi") <= -5 and g("rsi14") <= 55:
        out.append("PULLBACK_50")
    # 4. Leader near highs (momentum)
    if ok(g("rs_rank")) and g("rs_rank") >= 85 and g("pct_from_hi") >= -7 and uptrend:
        out.append("LEADER")
    # 5. Post-gap drift candidate
    if yes("gap_holding") and g("close") > g("sma50"):
        out.append("GAP_HOLD")
    # 6. Discount (feeds the fundamental dip process)
    if (ok(g("pct_from_hi")) and g("pct_from_hi") <= -20) or (ok(g("ret_21d")) and g("ret_21d") <= -12) or (ok(g("rsi14")) and g("rsi14") < 35):
        if yes("new_low_5d") and g("vol_ratio") >= 1.5:
            out.append("DISCOUNT_FALLING")      # knife: wait
        elif yes("higher_low") and ok(g("dist_sma20")) and g("dist_sma20") > 0:
            out.append("DISCOUNT_STABILIZING")  # best dip state
        else:
            out.append("DISCOUNT")
    return out


# --------------------------------------------------------------------------
# Data access (network). Kept separate so the logic above is testable offline.
# --------------------------------------------------------------------------

WIKI = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "ndx": "https://en.wikipedia.org/wiki/Nasdaq-100",
}


EXTRA: list[str] = []


def load_universe(prev: Path | None) -> pd.DataFrame:
    import io
    import urllib.request
    rows = []
    try:
        req = urllib.request.Request(WIKI["sp500"], headers={"User-Agent": "Mozilla/5.0 swing-scout"})
        t = pd.read_html(io.StringIO(urllib.request.urlopen(req, timeout=30).read().decode()))[0]
        for _, r in t.iterrows():
            rows.append({"ticker": str(r["Symbol"]).replace(".", "-"), "name": r["Security"], "sector": r["GICS Sector"],
                         "industry": r["GICS Sub-Industry"], "index": "SP500"})
        ndx_n = 0
        for url in (WIKI["ndx"], "https://www.slickcharts.com/nasdaq100"):   # Wikipedia dropped its table; slickcharts as backup
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 swing-scout"})
                tables = pd.read_html(io.StringIO(urllib.request.urlopen(req, timeout=30).read().decode()))
            except Exception as e:
                print("ndx source failed", url, e, file=sys.stderr)
                continue
            for t in tables:
                if len(t) < 90:
                    continue
                t.columns = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
                tc = next((c for c in t.columns if "ticker" in c.lower() or "symbol" in c.lower()), None)
                if not tc:
                    continue
                nc = next((c for c in t.columns if "company" in c.lower() or "security" in c.lower()), None)
                for _, r in t.iterrows():
                    rows.append({"ticker": str(r[tc]).strip().replace(".", "-"), "name": r[nc] if nc else "",
                                 "sector": "", "industry": "", "index": "NDX"})
                    ndx_n += 1
                break
            if ndx_n:
                break
        print("nasdaq-100 rows parsed:", ndx_n, file=sys.stderr)
        for t in EXTRA:
            rows.append({"ticker": t, "name": "", "sector": "", "industry": "", "index": "EXTRA"})
        u = pd.DataFrame(rows).drop_duplicates("ticker", keep="first")
        if len(u) < 450:
            raise ValueError(f"universe too small: {len(u)}")
        return u
    except Exception as e:  # fall back to the previous run's universe
        print("universe fetch failed:", e, file=sys.stderr)
        if prev and (prev / "universe.csv").exists():
            return pd.read_csv(prev / "universe.csv")
        return pd.DataFrame(columns=["ticker", "name", "sector", "industry", "index"])


def download(tickers: list[str]) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    out = {}
    for i in range(0, len(tickers), 100):
        chunk = tickers[i:i + 100]
        for attempt in range(3):
            try:
                d = yf.download(chunk, period="2y", interval="1d", auto_adjust=True, group_by="ticker",
                                threads=True, progress=False)
                break
            except Exception as e:
                print("download retry", attempt, e, file=sys.stderr)
                time.sleep(10 * (attempt + 1))
        else:
            continue
        for t in chunk:
            try:
                df = d[t] if isinstance(d.columns, pd.MultiIndex) else d
                df = df.dropna(how="all")
                if len(df) >= 5:
                    out[t] = df[["Open", "High", "Low", "Close", "Volume"]]
            except KeyError:
                pass
        time.sleep(2)
    for t in [t for t in tickers if t not in out]:   # one retry each, individually
        try:
            df = yf.download(t, period="2y", interval="1d", auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna(how="all")
            if len(df) >= 5:
                out[t] = df[["Open", "High", "Low", "Close", "Volume"]]
        except Exception as e:
            print("retry failed", t, e, file=sys.stderr)
        time.sleep(1)
    return out


def option_chains(t: str, spot: float, cfg: dict) -> tuple[pd.DataFrame, float | None]:
    import yfinance as yf
    tk = yf.Ticker(t)
    today = date.today()
    frames, atm_iv = [], None
    exps = []
    for e in tk.options or []:
        dte = (date.fromisoformat(e) - today).days
        for lo, hi in cfg["option_dte_windows"]:
            if lo <= dte <= hi:
                exps.append((e, dte, (lo, hi)))
    picked, seen = [], set()
    for e, dte, w in exps:  # first expiration in each window, plus monthly-ish second in swing window
        if w not in seen or (w == tuple(cfg["option_dte_windows"][0]) and len([p for p in picked if p[2] == w]) < 2):
            picked.append((e, dte, w)); seen.add(w)
    band = cfg["option_strike_band"]
    for e, dte, _ in picked:
        ch = tk.option_chain(e)
        for side, df in (("put", ch.puts), ("call", ch.calls)):
            df = df[(df["strike"] >= spot * (1 - band)) & (df["strike"] <= spot * (1 + band))].copy()
            df["type"], df["expiration"], df["dte"] = side, e, dte
            frames.append(df[["type", "expiration", "dte", "strike", "bid", "ask", "lastPrice", "impliedVolatility",
                              "openInterest", "volume"]])
        if atm_iv is None and 21 <= dte <= 50:
            p = ch.puts.iloc[(ch.puts["strike"] - spot).abs().argsort()[:2]]
            c = ch.calls.iloc[(ch.calls["strike"] - spot).abs().argsort()[:2]]
            vals = pd.concat([p, c])["impliedVolatility"]
            vals = vals[(vals > 0.01) & (vals < 5)]
            atm_iv = float(vals.mean()) if len(vals) else None
        time.sleep(0.5)
    return (pd.concat(frames) if frames else pd.DataFrame()), atm_iv


FUND_KEYS = {"marketCap": "mcap_b", "forwardPE": "fwd_pe", "trailingPE": "ttm_pe", "trailingPegRatio": "peg",
             "enterpriseToEbitda": "ev_ebitda", "priceToSalesTrailing12Months": "ps", "profitMargins": "net_margin",
             "operatingMargins": "op_margin", "revenueGrowth": "rev_growth", "earningsGrowth": "eps_growth",
             "freeCashflow": "fcf", "debtToEquity": "debt_eq", "totalCash": "cash", "totalDebt": "debt",
             "targetMeanPrice": "target_mean", "targetMedianPrice": "target_median", "recommendationMean": "rec_mean",
             "numberOfAnalystOpinions": "n_analysts", "shortPercentOfFloat": "short_float", "heldPercentInsiders": "insider_pct",
             "sharesOutstanding": "shares", "currentPrice": "price"}


def fundamentals(tickers: list[str]) -> pd.DataFrame:
    """Yahoo quoteSummary fields for a limited set (about 1s per ticker)."""
    import yfinance as yf
    rows = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info or {}
        except Exception as e:
            print("info failed", t, e, file=sys.stderr)
            continue
        r = {"ticker": t, "sector": info.get("sector", ""), "industry": info.get("industry", "")}
        for k, v in FUND_KEYS.items():
            r[v] = info.get(k)
        if r["mcap_b"]:
            r["mcap_b"] = r["mcap_b"] / 1e9
            if r["fcf"]:
                r["fcf_yield"] = r["fcf"] / (r["mcap_b"] * 1e9) * 100
        for k in ("net_margin", "op_margin", "rev_growth", "eps_growth", "short_float", "insider_pct"):
            if r.get(k) is not None:
                r[k] = r[k] * 100
        if r.get("target_mean") and r.get("price"):
            r["target_upside"] = (r["target_mean"] / r["price"] - 1) * 100
        rows.append(r)
        time.sleep(0.6)
    return pd.DataFrame(rows)


def _n(x, fmt="{:.0f}", suf=""):
    return "–" if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else fmt.format(x) + suf


def digest(snap: pd.DataFrame, fund: pd.DataFrame, market: dict, earn: pd.DataFrame, ivr: pd.DataFrame, cfg: dict,
           meta: dict) -> str:
    """Compact markdown the skill reads instead of the raw CSVs (~3-4k tokens)."""
    fund = fund.reindex(columns=list(dict.fromkeys(["ticker", "sector", "mcap_b", "fcf_yield", "op_margin"] + list(fund.columns))))
    s = snap.set_index("ticker")
    f = fund.set_index("ticker") if len(fund) else pd.DataFrame()
    e = earn.set_index("ticker")["next_earnings"].to_dict() if len(earn) else {}
    iv = ivr.set_index("ticker")["iv_rank"].to_dict() if len(ivr) else {}
    fv = lambda t, k: f.at[t, k] if t in f.index and k in f.columns else None

    def line(t):
        r = s.loc[t]
        sec = r.get("sector") if isinstance(r.get("sector"), str) and r.get("sector") else (fv(t, "sector") or "")
        sec = sec if isinstance(sec, str) else ""
        parts = [f"**{t}**", sec[:14], f"${_n(r['close'], '{:.2f}')}", f"hi {_n(r['pct_from_hi'], '{:+.0f}', '%')}",
                 f"50d {_n(r['dist_sma50'], '{:+.0f}', '%')}", f"200d {_n(r['dist_sma200'], '{:+.0f}', '%')}",
                 f"RSI {_n(r['rsi14'])}", f"RS {_n(r.get('rs_rank'))}", f"ATR {_n(r['atr_pct'], '{:.1f}', '%')}"]
        if fv(t, "mcap_b"):
            parts += [f"cap {_n(fv(t, 'mcap_b'), '{:.0f}', 'B')}", f"fPE {_n(fv(t, 'fwd_pe'), '{:.1f}')}",
                      f"FCFy {_n(fv(t, 'fcf_yield'), '{:.1f}', '%')}", f"opM {_n(fv(t, 'op_margin'), '{:.0f}', '%')}",
                      f"revG {_n(fv(t, 'rev_growth'), '{:+.0f}', '%')}", f"tgt {_n(fv(t, 'target_upside'), '{:+.0f}', '%')}",
                      f"SI {_n(fv(t, 'short_float'), '{:.1f}', '%')}"]
        if e.get(t) and str(e[t])[:10] >= meta["data_asof"]:   # Yahoo repeats the last date until a new one is set
            parts.append(f"earn {str(e[t])[5:10]}")
        if t in iv and iv[t] == iv[t] and iv[t] is not None:
            parts.append(f"IVR {_n(iv[t])}")
        if r.get("flags"):
            parts.append(f"[{r['flags']}]")
        return " | ".join(p for p in parts if p)

    m = market
    out = [f"# Swing Scout digest — data as of {meta['data_asof']} (built {meta['built_utc'][:16]}Z)", ""]
    ix = m["indexes"]
    reg = [f"{k} {_n((v or {}).get('close'), '{:.2f}')} (50d {_n((v or {}).get('dist_sma50'), '{:+.1f}', '%')}, "
           f"200d {_n((v or {}).get('dist_sma200'), '{:+.1f}', '%')}, 200d slope {_n((v or {}).get('sma200_slope_20d'), '{:+.1f}', '%')})"
           for k, v in ix.items() if k != "^VIX"]
    b = m["breadth_sp500"]
    out += ["## Regime", "; ".join(reg) + f"; VIX {_n((ix.get('^VIX') or {}).get('close'), '{:.1f}')}",
            f"S&P 500 breadth: {_n(b['pct_above_200d'])}% >200d, {_n(b['pct_above_50d'])}% >50d, 52w highs/lows {b['new_52w_highs']}/{b['new_52w_lows']}", "",
            "## Sectors & industries (1m / 3m %, weakest first)",
            ", ".join(f"{x['ticker']} {_n(x['ret_21d'], '{:+.0f}')}/{_n(x['ret_63d'], '{:+.0f}')}" for x in m["sectors"]), ""]
    special = set(cfg["watchlist"]) | set(cfg.get("holdings", [])) | set(cfg["frontier"])
    stocks = snap[(~snap["etf"]) & (~snap["stale"]) & (~snap["ticker"].isin(special))].copy()
    stocks = stocks.merge(fund[["ticker", "mcap_b", "fcf_yield", "op_margin"]], on="ticker", how="left")
    big = stocks[(stocks["mcap_b"].isna()) | (stocks["mcap_b"] >= cfg.get("min_market_cap_b", 10))]
    has = lambda fl: big[big["flags"].fillna("").str.split(",").apply(lambda xs: fl in xs)]
    sections = [
        ("DISCOUNT_STABILIZING", "Dips that stabilized (quality first: FCF+ and op margin+, deepest discount first)",
         lambda d: d.assign(q=((d["fcf_yield"] > 0) & (d["op_margin"] > 0)).astype(int)).sort_values(["q", "pct_from_hi"], ascending=[False, True]), 12),
        ("PULLBACK_50", "Pullbacks to the 50-day in uptrends (by RS)", lambda d: d.sort_values("rs_rank", ascending=False), 8),
        ("BREAKOUT", "Breakouts", lambda d: d.sort_values("rs_rank", ascending=False), 6),
        ("PIVOT_WATCH", "Coiling under a pivot", lambda d: d.sort_values("rs_rank", ascending=False), 6),
        ("GAP_HOLD", "Gap-and-hold (likely earnings)", lambda d: d.sort_values("rs_rank", ascending=False), 6),
        ("LEADER", "Leaders near highs (enter only via pullback/breakout)", lambda d: d.sort_values("rs_rank", ascending=False), 8),
    ]
    out.append("## Candidates outside your lists (market cap ≥ $10B where known)")
    shown = set()
    for fl, title, sort, k in sections:
        d = has(fl)
        d = d[~d["ticker"].isin(shown)]
        if len(d):
            top = sort(d)["ticker"].head(k).tolist()
            shown |= set(top)
            out.append(f"### {title} — {len(d)} more, top {len(top)}")
            out += ["- " + line(t) for t in top]
    out.append(f"(Plain DISCOUNT, not yet stabilizing: {len(has('DISCOUNT'))}; still falling: {len(has('DISCOUNT_FALLING'))}. See setups.csv.)")
    out.append("")
    for title, names in (("Holdings", cfg.get("holdings", [])), ("Watchlist", [t for t in cfg["watchlist"] if t not in cfg.get("holdings", [])]),
                         ("Frontier", cfg["frontier"])):
        if not names:
            continue
        out.append(f"## {title}")
        out += ["- " + (line(t) if t in s.index else f"**{t}** — no data") for t in names]
        out.append("")
    soon = [f"{t} {str(d)[:10]}" for t, d in e.items() if d and str(d)[:10] <= (pd.Timestamp(meta["data_asof"]) + pd.Timedelta(days=21)).strftime("%Y-%m-%d")
            and str(d)[:10] >= meta["data_asof"]]
    out += ["## Earnings in the next 3 weeks (your lists)", ", ".join(soon) or "none", ""]
    notes = []
    if meta.get("missing"):
        notes.append("missing: " + ", ".join(meta["missing"][:20]))
    st = snap.loc[snap["stale"], "ticker"].tolist()
    if st:
        notes.append("stale (ignored): " + ", ".join(st))
    out += ["## Data notes", "; ".join(notes) or "none",
            "Legend: hi = % from 52w high; 50d/200d = % vs SMA; RS = rank 0-100; fPE = forward P/E; FCFy = FCF yield; "
            "opM = operating margin; revG = revenue growth y/y; tgt = analyst mean target vs price; SI = short % of float; IVR = IV rank."]
    return "\n".join(out)


def earnings_date(t: str):
    import yfinance as yf
    try:
        cal = yf.Ticker(t).calendar
        d = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if d:
            return str(d[0])
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

def clean(x):
    if isinstance(x, (float, np.floating)):
        return None if math.isnan(x) or math.isinf(x) else round(float(x), 3)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def build(prices: dict[str, pd.DataFrame], universe: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    spy = prices["SPY"]["Close"] if "SPY" in prices else None
    meta_u = universe.set_index("ticker").to_dict("index") if len(universe) else {}
    special = set(cfg["watchlist"]) | set(cfg.get("holdings", [])) | set(cfg["frontier"])
    rows = []
    for t, df in prices.items():
        try:
            f = features(df, spy)
        except Exception as e:
            print("features failed", t, e, file=sys.stderr)
            continue
        m = meta_u.get(t, {})
        f.update({"ticker": t, "name": m.get("name", ""), "sector": m.get("sector", ""), "industry": m.get("industry", ""),
                  "index": m.get("index", ""), "watchlist": t in cfg["watchlist"], "holding": t in cfg.get("holdings", []),
                  "frontier": t in cfg["frontier"],
                  "etf": t in set(cfg["sector_etfs"]) | set(cfg["industry_etfs"]) | set(cfg["market"])})
        rows.append(f)
    snap = pd.DataFrame(rows)
    asof = snap.loc[~snap["ticker"].str.startswith("^"), "date"].mode().iloc[0]
    snap["stale"] = snap["date"] < asof
    # IBD-style relative strength rank across stocks (not ETFs)
    w = 0.4 * snap["ret_63d"].fillna(0) + 0.2 * snap["ret_126d"].fillna(0) + 0.2 * snap["ret_252d"].fillna(snap["ret_126d"]).fillna(0) + 0.2 * snap["ret_21d"].fillna(0)
    stocks = ~snap["etf"]
    snap.loc[stocks, "rs_rank"] = w[stocks].rank(pct=True) * 100
    snap["flags"] = [",".join(flags(r)) for r in snap.to_dict("records")]
    lead = ["ticker", "name", "sector", "industry", "flags", "watchlist", "holding", "frontier", "etf", "index"]
    snap = snap[lead + [c for c in snap.columns if c not in lead]].sort_values("ticker")

    # Market regime and breadth
    def row(t):
        r = snap[snap["ticker"] == t]
        return {k: clean(v) for k, v in r.iloc[0].items()} if len(r) else None
    sp = snap[snap["index"] == "SP500"]
    market = {
        "asof": asof,
        "breadth_sp500": {
            "n": int(len(sp)),
            "pct_above_50d": clean((sp["close"] > sp["sma50"]).mean() * 100),
            "pct_above_200d": clean((sp["close"] > sp["sma200"]).mean() * 100),
            "new_52w_highs": int((sp["pct_from_hi"] >= -0.5).sum()),
            "new_52w_lows": int((sp["pct_from_lo"] <= 0.5).sum()),
        },
        "indexes": {t: row(t) for t in cfg["market"]},
        "sectors": sorted([{k: row(t).get(k) for k in ("ticker", "ret_21d", "ret_63d", "dist_sma50", "dist_sma200", "pct_from_hi")}
                           for t in cfg["sector_etfs"] + cfg["industry_etfs"] if row(t)], key=lambda r: r["ret_21d"] or 0),
        "flag_counts": snap["flags"].str.split(",").explode().value_counts().drop("", errors="ignore").to_dict(),
    }
    return snap, market


def run_options(cfg: dict, out: Path):
    """Intraday job: refresh option chains + ATM IV history in place on the data branch."""
    import yfinance as yf
    (out / "options").mkdir(exist_ok=True)
    special = list(dict.fromkeys(cfg["watchlist"] + cfg.get("holdings", []) + cfg["frontier"]))
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ivh_path = out / "iv_history.csv"
    ivh = pd.read_csv(ivh_path) if ivh_path.exists() else pd.DataFrame(columns=["date", "ticker", "atm_iv"])
    ok, empty, fail = [], [], []
    for t in special:
        try:
            spot = float(yf.Ticker(t).fast_info["last_price"])
            ch, iv = option_chains(t, spot, cfg)
            if not len(ch):
                continue
            if (ch["bid"].fillna(0) > 0).mean() < 0.2:      # quotes blanked (market closed / bad feed)
                empty.append(t)
                continue
            ch.insert(0, "quoted_utc", datetime.now(timezone.utc).isoformat(timespec="minutes"))
            ch.insert(1, "spot", round(spot, 4))
            ch.round(4).to_csv(out / "options" / f"{t}.csv", index=False)
            ok.append(t)
            if iv:
                ivh = pd.concat([ivh[~((ivh["date"] == today) & (ivh["ticker"] == t))],
                                 pd.DataFrame([{"date": today, "ticker": t, "atm_iv": round(iv, 4)}])])
        except Exception as e:
            fail.append(f"{t}: {e}")
    ivh.to_csv(ivh_path, index=False)
    ivr = []
    for t, g in ivh.groupby("ticker"):
        g = g.sort_values("date")
        cur, lo, hi = g["atm_iv"].iloc[-1], g["atm_iv"].min(), g["atm_iv"].max()
        ivr.append({"ticker": t, "atm_iv": cur, "days": len(g),
                    "iv_rank": round((cur - lo) / (hi - lo) * 100, 1) if hi > lo and len(g) >= 60 else None})
    pd.DataFrame(ivr).to_csv(out / "iv_rank.csv", index=False)
    meta = {"quoted_utc": datetime.now(timezone.utc).isoformat(timespec="minutes"), "ok": ok,
            "blank_quotes": empty, "failures": fail}
    (out / "options_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({"ok": len(ok), "blank": len(empty), "fail": len(fail)}))
    if not ok:
        sys.exit("no usable option quotes; not publishing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["prices", "options"], default="prices")
    ap.add_argument("--out", default="out")
    ap.add_argument("--prev", default="prev")
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    out, prev = Path(a.out), Path(a.prev)
    out.mkdir(parents=True, exist_ok=True)
    if a.mode == "prices":
        import shutil
        shutil.copy(a.config, out / "config.yaml")   # so readers of the data branch see the lists
    if a.mode == "options":
        return run_options(cfg, out)
    (out / "ohlcv").mkdir(parents=True, exist_ok=True)
    (out / "options").mkdir(exist_ok=True)
    started = datetime.now(timezone.utc)

    EXTRA.extend(cfg.get("extra_universe", []))
    universe = load_universe(prev)
    universe.to_csv(out / "universe.csv", index=False)
    special = list(dict.fromkeys(cfg["watchlist"] + cfg.get("holdings", []) + cfg["frontier"]))
    tickers = list(dict.fromkeys(cfg["market"] + cfg["sector_etfs"] + cfg["industry_etfs"] + special + universe["ticker"].tolist()))
    prices = download(tickers)
    missing = [t for t in tickers if t not in prices]

    snap, market = build(prices, universe, cfg)
    snap.round(3).to_csv(out / "snapshot.csv", index=False)
    setups = snap[(snap["flags"] != "") & (~snap["etf"])].copy()
    setups.round(3).to_csv(out / "setups.csv", index=False)
    (out / "market.json").write_text(json.dumps(market, indent=1, default=clean))

    # Charts for special tickers and everything flagged (except plain DISCOUNT noise outside the list)
    chart_set = set(special) | set(setups.loc[~setups["flags"].isin(["DISCOUNT"]), "ticker"]) | set(cfg["sector_etfs"]) | {"SPY", "QQQ"}
    for t in chart_set:
        if t in prices:
            prices[t].tail(cfg["ohlcv_keep_rows"]).round(4).to_csv(out / "ohlcv" / f"{t.replace('^', '')}.csv")

    # Earnings dates (nightly). Option chains come from the intraday job (--mode options); carry them forward.
    earn = []
    for t in special:
        if t in prices and t not in cfg["sector_etfs"] + cfg["industry_etfs"]:
            earn.append({"ticker": t, "next_earnings": earnings_date(t)})
    earn_df = pd.DataFrame(earn)
    earn_df.to_csv(out / "earnings.csv", index=False)
    # Fundamentals for the names the brief can actually discuss
    fl = snap["flags"].fillna("")
    pool = snap.loc[(~snap["etf"]) & (fl.str.contains("STABILIZING|PULLBACK|BREAKOUT|PIVOT|GAP|LEADER")), "ticker"].tolist()
    fund = fundamentals([t for t in dict.fromkeys(special + pool) if t in prices and t not in cfg["sector_etfs"] + cfg["industry_etfs"]])
    fund.round(3).to_csv(out / "fundamentals.csv", index=False)
    import shutil
    for name in ("iv_history.csv", "iv_rank.csv", "options_meta.json"):
        if (prev / name).exists():
            shutil.copy(prev / name, out / name)
    if (prev / "options").exists() and (prev / "options_meta.json").exists():   # only chains from the intraday job
        shutil.copytree(prev / "options", out / "options", dirs_exist_ok=True)
    asof = market["asof"]
    opt_fail = []

    meta = {"built_utc": started.isoformat(timespec="seconds"), "data_asof": asof, "tickers_requested": len(tickers),
            "tickers_ok": len(prices), "missing": missing[:200], "option_failures": opt_fail[:50],
            "files": ["snapshot.csv", "setups.csv", "market.json", "earnings.csv", "universe.csv", "ohlcv/<T>.csv",
                      "options/<T>.csv + options_meta.json + iv_history.csv + iv_rank.csv (from the intraday options job)"]}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    ivr = pd.read_csv(out / "iv_rank.csv") if (out / "iv_rank.csv").exists() else pd.DataFrame()
    (out / "digest.md").write_text(digest(snap, fund, market, earn_df, ivr, cfg, meta))
    print(json.dumps({k: meta[k] for k in ("data_asof", "tickers_requested", "tickers_ok")}))
    if len(prices) < 0.8 * len(tickers):
        sys.exit("too many tickers missing; refusing to publish partial data")


if __name__ == "__main__":
    main()
