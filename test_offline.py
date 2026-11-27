"""Offline test: synthetic prices with known shapes -> check indicators and flags."""
import numpy as np
import pandas as pd
import yaml

import build

rng = np.random.default_rng(7)
idx = pd.bdate_range("2024-11-01", periods=520)


def frame(close, vol=None):
    close = pd.Series(close, index=idx[: len(close)])
    o = close.shift().fillna(close.iloc[0])
    h = np.maximum(o, close) * 1.005
    l = np.minimum(o, close) * 0.995
    v = pd.Series(vol if vol is not None else np.full(len(close), 1e6), index=close.index)
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": close, "Volume": v})


n = 520
spy = frame(100 * np.cumprod(1 + rng.normal(0.0004, 0.008, n)))

# 1) Uptrend, tight base, breakout on volume today
up = np.concatenate([np.linspace(50, 100, n - 41), 100 + rng.normal(0, 0.6, 40), [104]])
vol = np.full(n, 1e6); vol[-1] = 2.5e6
brk = frame(up, vol)

# 2) Steady uptrend then pullback to the 50-day
pb = np.concatenate([np.linspace(50, 80, n - 95), np.linspace(80, 110, 80), np.linspace(110, 102, 15)])
pull = frame(pb)

# 3) Crash still making new lows on heavy volume
cr = np.concatenate([np.linspace(100, 100, n - 30), np.linspace(100, 60, 30)])
v3 = np.full(n, 1e6); v3[-3:] = 3e6
crash = frame(cr, v3)

# 4) Dropped, then a higher low and back above the 20-day
st = np.concatenate([np.full(n - 40, 100.0), np.linspace(100, 70, 20), np.linspace(70, 66, 8), np.linspace(66, 76, 12)])
stab = frame(st)

# 5) Uptrend with a 8% gap-up on 4x volume 10 sessions ago, holding above the gap-day low
gp = np.concatenate([np.linspace(50, 90, n - 10), np.linspace(97, 100, 10)])
gv = np.full(n, 1e6); gv[-10] = 4e6
gapper = frame(gp, gv)
gapper.iloc[-10, gapper.columns.get_loc("Open")] = 97.0
gapper.iloc[-10, gapper.columns.get_loc("Low")] = 96.5
# 6) Brand-new IPO with 10 rows; 7) a stale ticker whose data stopped a week early
ipo = frame(np.linspace(83, 96, 10)).set_axis(idx[n - 10:n])
stale = frame(np.linspace(50, 60, n - 5))

prices = {"SPY": spy, "BRK": brk, "PULL": pull, "CRASH": crash, "STAB": stab, "GAPR": gapper, "IPO": ipo, "OLD": stale}
uni = pd.DataFrame([{"ticker": t, "name": t, "sector": "Test", "industry": "", "index": "SP500"} for t in prices if t != "SPY"])
cfg = yaml.safe_load(open("config.yaml"))
snap, market = build.build(prices, uni, cfg)
s = snap.set_index("ticker")
print(s[["close", "rsi14", "dist_sma50", "pct_from_hi", "vol_ratio", "base_depth_40d", "dist_pivot", "rs_rank", "flags"]].round(2).to_string())
print(market["breadth_sp500"], market["flag_counts"])

assert "BREAKOUT" in s.loc["BRK", "flags"], s.loc["BRK", "flags"]
assert "PULLBACK_50" in s.loc["PULL", "flags"], s.loc["PULL", "flags"]
assert "DISCOUNT_FALLING" in s.loc["CRASH", "flags"], s.loc["CRASH", "flags"]
assert "DISCOUNT_STABILIZING" in s.loc["STAB", "flags"], s.loc["STAB", "flags"]

# RSI sanity vs a hand-checked series: monotonic rise -> 100; alternating -> ~50
assert build.rsi(pd.Series(np.arange(1, 40.0))).iloc[-1] == 100
alt = build.rsi(pd.Series(100 + np.tile([1, -1], 50).cumsum())).iloc[-1]
assert 40 < alt < 60, alt
assert "GAP_HOLD" in s.loc["GAPR", "flags"], s.loc["GAPR", "flags"]
for t in ("BRK", "PULL", "CRASH", "STAB"):
    assert "GAP_HOLD" not in s.loc[t, "flags"], (t, s.loc[t, "flags"])   # NaN must not count as True
assert s.loc["IPO", "rows"] == 10 and s.loc["IPO", "flags"] == ""
assert bool(s.loc["OLD", "stale"]) and s.loc["OLD", "flags"] == ""
assert market["asof"] == s.loc["BRK", "date"]
print("ALL TESTS PASSED")
