# swing-scout-data

Free end-of-day market data for the Swing Scout skill. A GitHub Actions job runs after each US close (23:15 UTC on weekdays). It pulls about 650 tickers through yfinance: the S&P 500, the Nasdaq-100, `config.yaml` lists, and sector ETFs. It computes indicators and setup flags, then force-pushes the results to the `data` branch as a single commit, so the repo stays small.

**Edit `config.yaml`** to change the watchlist, holdings or frontier names. Run it on demand from the Actions tab (`daily-data` → Run workflow).

## Files on the `data` branch

| File | What |
|---|---|
| `digest.md` | **Start here.** ~4k-token summary: regime, breadth, sectors, ranked candidates per setup with fundamentals, and status lines for holdings, watchlist and frontier |
| `fundamentals.csv` | Yahoo fundamentals (market cap, P/E, EV/EBITDA, margins, growth, FCF yield, debt, analyst targets, short interest) for your lists plus every non-plain-DISCOUNT flagged stock |
| `meta.json` | Build time, data date, tickers ok/missing, option failures. **Check `data_asof` first.** |
| `market.json` | SPY/QQQ/IWM/VIX, S&P 500 breadth (% above the 50/200-day, new highs and lows), sector and industry ETF table sorted by 1-month return, flag counts |
| `snapshot.csv` | One row per ticker: returns (1d–12m, 12-1 momentum), 52-week high/low distance, SMA 10/20/50/150/200 with distances and slopes, RSI14, ATR14 and ATR%, ATR contraction, volume ratio, base depth and pivot, higher-low and new-low flags, weekly trend, gap-and-hold, RS vs SPY, RS rank, `flags` |
| `setups.csv` | Only stock rows with flags |
| `ohlcv/<T>.csv` | 300 days of daily OHLCV for watchlist, holdings, frontier, sector ETFs and every flagged setup |
| `options/<T>.csv` | Chains snapped **intraday** (2:30pm Central in winter, 3:30pm in summer; `intraday-options` workflow), with `quoted_utc` and `spot`. Bid/ask/IV/OI for the first 21–50 DTE expirations and one LEAPS, strikes within ±25%. Chains with blanked quotes are skipped and listed in `options_meta.json` |
| `iv_rank.csv` | ATM IV and IV rank from this repo's own history (valid after 60 trading days) |
| `earnings.csv` | Next earnings date for watchlist, holdings and frontier |

## Flags

`BREAKOUT`, `PIVOT_WATCH`, `PULLBACK_50`, `LEADER`, `GAP_HOLD`, `DISCOUNT`, `DISCOUNT_STABILIZING`, `DISCOUNT_FALLING`. Definitions are in `flags()` in `build.py`. They are screens, not signals.

## Notes

- yfinance is an unofficial Yahoo interface. It can break or get rate-limited. The job refuses to publish if more than 20% of tickers fail, so stale-but-honest data beats partial data. Check `meta.json`.
- Option quotes are end-of-day. Verify the live chain before trading.
