# OTC Signal Agent

A separate project from Premamulu. It shows **CALL / PUT / NO TRADE** signals for IQ Option OTC assets at **1, 2, 3 and 5 minute** expiries. It does **not** place trades. You decide and click.

## How it decides

Each closed 1-minute candle is passed through three layers:

1. **Indicators (about 30 of them):** EMA 5/9/21/50, MACD, RSI 7/14, Stochastic, Williams %R, CCI, ADX/DI, ROC, SuperTrend, Heikin-Ashi, Bollinger %B and width, ATR, Donchian and z-score.
2. **Candlestick patterns (19 of them):** doji (plain, dragonfly and gravestone), hammer, hanging man, inverted hammer, shooting star, marubozu, engulfing, harami, piercing line, dark cloud cover, morning star and evening star, three white soldiers and three black crows, tweezer top/bottom, three inside up/down, three outside up/down, kicker and belt hold.
3. **AI ensemble, one per expiry:**
   - gradient boosting on every feature
   - logistic regression on the same features
   - **history matching**: a k-nearest-neighbours model that finds the past moments whose last 10 candles looked most like now, and checks what happened next

## The safety gate (the most important part)

The model is tested **walk-forward**: it is always scored on candles it never trained on. An expiry is marked TRADABLE only if, at some confidence level, the out-of-sample win rate stays above break-even with 95% statistical confidence. At an 85% payout, break-even is a 54.05% win rate. A live signal must also meet all of these:

- confidence ≥ the validated threshold
- all 3 models agree on the direction

If any check fails, the agent shows **NO TRADE**. Seeing NO TRADE often is normal. It means the gate is working.

## Setup (Termux on Android)

Type the commands **one line at a time** and press Enter after each one. Do not paste the whole block at once.

```bash
pkg update
pkg install python git tur-repo
pkg install python-numpy python-pandas
pip install -r requirements-iq.txt
```

The repo is private, so `git clone` asks for a username and password:
- **Username:** `premkumarsmart5-hub`
- **Password:** a GitHub *token*, not your GitHub password. Create one in GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token. Set Repository access to *Only select repositories* → `premtheking`, and set Permissions → Contents to *Read-only*.

```bash
git clone https://github.com/premkumarsmart5-hub/premtheking
cd premtheking
cp .env.example .env
nano .env        # your IQ Option email + password, stays on your phone
```

Only numpy and pandas are needed, and the AI models are written in plain numpy, so scikit-learn is not required.
To update later, run `git pull` inside the `premtheking` folder.

On a PC or VPS: `pip install -r requirements.txt -r requirements-iq.txt`

## Use

```bash
python main.py demo                                   # offline test, no login
python main.py collect --asset EURUSD-OTC --days 7    # download history (more is better)
python main.py train   --asset EURUSD-OTC --payout 0.85
python main.py live    --asset EURUSD-OTC --dashboard # open http://127.0.0.1:8080
python main.py scan    --days 10                      # test 12 OTC pairs, list any with an edge
python main.py scan    --market real --days 14        # test 12 REAL forex pairs (EURUSD, GBPUSD…)
python main.py scan    --tf 5                         # 5-minute candles: 5 / 10 / 15 min expiries
```

**5 / 10 / 15 minute expiries:** add `--tf 5` to `collect`, `train`, `scan` and `live`. Signals then come at each 5-minute candle close. The expiry is 1, 2 or 3 candles later, which lines up with IQ Option's clock-based expiries. By default this downloads 60 days of history, because there are 5× fewer candles per day. Pick your own expiries with `--expiries 5,10,15,30`; each one must be a multiple of 5. Files for 1-min and 5-min candles are kept apart (`data/<asset>_1m.csv` and `data/<asset>_5m.csv`).

**Real markets vs OTC:** real pairs (names without `-OTC`) come from actual interbank trading and are open Monday to Friday only. History can still be downloaded at the weekend. Trades are never scored across a weekend or holiday gap. In live mode, a closed market shows "market closed" instead of signals. IQ Option's payout on real pairs is often lower than on OTC, so pass the real number with `--payout`.

`scan` downloads and tests each asset. Only an asset/expiry that passes **both** checks gets a TRADE verdict:
1. the walk-forward test on the older 80% of candles
2. a final exam on the newest 20% of candles, which are never used to choose any setting

Testing many assets makes lucky passes more likely, and the final exam guards against that. `train` uses the same final exam.
Results go to `scan_results.csv`. Use `--assets GBPUSD-OTC,AUDCAD-OTC` to pick your own list, or `--verbose` to see full reports.

Use the payout IQ Option shows for that asset (for example 0.82 or 0.87). Every signal is also appended to `signals_log.jsonl`, so you can check real results later. Retrain every few days, because OTC behaviour changes.

## Test your own strategy idea

```bash
python main.py strategy --list                 # the built-in strategies
python main.py strategy --name all             # test all of them on every 1-min file in data/
python main.py strategy --name all --tf 5      # same on 5-min candles (5/10/15 min expiries)
python main.py strategy --name ema_cross --assets EURUSD-OTC,GBPUSD-OTC
```

It uses candles already downloaded by `collect`/`scan`, so no internet is needed and results take seconds.
A rule PASSES only if its win rate beats break-even with 95% confidence and also on the newest 20% of candles.
New rules go in `otcagent/strategies.py`: write a function that returns +1 (CALL), -1 (PUT) or 0 for each candle, then add it to `STRATEGIES`.

## Honest limits

- OTC prices are generated by the broker. They are not a real market, and the broker can change how they behave at any time.
- No indicator or AI can promise a fixed win rate. The backtest is an estimate, and live results are usually somewhat worse.
- `iqoptionapi` is unofficial and may break or violate the broker's terms. Use the PRACTICE account.
- Forward-test on the demo account for at least 1–2 weeks and compare `signals_log.jsonl` with real outcomes before you risk money.

## Tests

`python -m pytest -q`: checks that there is no look-ahead in features, that the agent finds a planted pattern, and that it refuses to trade on pure noise.
