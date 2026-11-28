"""OTC Signal Agent - command line.

  python main.py collect  --asset EURUSD-OTC --days 7     # download candle history
  python main.py train    --asset EURUSD-OTC --payout 0.85 # backtest + train AI models
  python main.py live     --asset EURUSD-OTC --dashboard   # live CALL/PUT signals
  python main.py scan     --days 10                       # test 12 OTC pairs, list winners
  python main.py scan     --market real --days 14         # same for real forex pairs
  python main.py scan     --tf 5                          # 5-min candles: 5/10/15 min expiries
  python main.py strategy --name all                      # test classic rules on saved data
  python main.py forex    --tf 1                           # BIG forex agent: train + backtest on all real pairs
  python main.py forex-live --asset EURUSD --dashboard    # live analysis + CALL/PUT
  python main.py demo                                      # offline demo, no login needed
"""
import argparse
import json
from pathlib import Path

from otcagent import data
from otcagent.backtest import train_and_validate
from otcagent.signals import format_signal, generate, load_bundle, run_live, save_bundle


def paths(asset, tf=1):
    safe = asset.replace("/", "_")
    return Path("data") / f"{safe}_{tf}m.csv", Path("models") / f"{safe}_{tf}m.pkl"


def expiries_to_candles(a):
    """--expiries is in minutes; turn it into a number of candles of size --tf."""
    exp = a.expiries or ("1,2,3,5" if a.tf == 1 else "5,10,15")
    out = []
    for m in (int(x) for x in exp.split(",")):
        if m % a.tf:
            raise SystemExit(f"expiry {m} min is not a multiple of the {a.tf}-min candle")
        out.append(m // a.tf)
    return tuple(out)


def default_days(a):
    return a.days if a.days else (10 if a.tf == 1 else 60)


REAL_SCAN = [  # real interbank forex: open Mon-Fri only
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "EURJPY",
    "EURGBP", "GBPJPY", "AUDCAD", "NZDUSD", "USDCHF", "EURAUD",
]

DEFAULT_SCAN = [
    "EURUSD-OTC", "GBPUSD-OTC", "USDJPY-OTC", "AUDCAD-OTC", "EURGBP-OTC", "EURJPY-OTC",
    "NZDUSD-OTC", "USDCHF-OTC", "GBPJPY-OTC", "AUDUSD-OTC", "USDCAD-OTC", "EURAUD-OTC",
]


def collect_asset(src, asset, days, tf=1):
    csv, _ = paths(asset, tf)
    print(f"Downloading {days} days of {tf}-minute {asset} candles…")
    new = src.history(asset, days, size=60 * tf)
    if csv.exists():  # merge with what we already have, history keeps growing
        old = data.load_csv(csv)
        new = old.combine_first(new) if len(old) else new
    data.save_csv(new, csv)
    print(f"Saved {len(new)} candles to {csv}")
    return new


def cmd_collect(a):
    src = data.IQOptionSource(account=a.account).connect()
    collect_asset(src, a.asset, default_days(a), a.tf)


def cmd_scan(a):
    assets = ([x.strip() for x in a.assets.split(",")] if a.assets
              else REAL_SCAN if a.market == "real" else DEFAULT_SCAN)
    src = None if a.no_download else data.IQOptionSource(account=a.account).connect()
    horizons = expiries_to_candles(a)
    days = default_days(a)
    tf = a.tf
    import pandas as pd
    out = Path(f"scan_results_{tf}m.csv")
    rows = []
    if a.resume and out.exists():
        rows = [r for r in pd.read_csv(out, dtype=str).fillna("-").to_dict("records")
                if not str(r["verdict"]).startswith("ERROR")]
        done = {r["asset"] for r in rows}
        skip = [x for x in assets if x in done]
        assets = [x for x in assets if x not in done]
        if skip:
            print(f"Resuming: already done {', '.join(skip)}")
    elif out.exists():
        out.unlink()
    for i, asset in enumerate(assets, 1):
        print(f"\n##### [{i}/{len(assets)}] {asset} #####")
        try:
            if src:
                df = collect_asset(src, asset, days, tf)
            else:
                df = data.load_csv(paths(asset, tf)[0])
            if len(df) < 2000:
                raise ValueError(f"only {len(df)} candles - asset closed or unknown?")
            bundle = train_and_validate(df, payout=a.payout, horizons=horizons,
                                        n_folds=a.folds, tf=tf, log=print if a.verbose else (lambda *_: None))
            bundle["asset"] = asset
            Path("models").mkdir(exist_ok=True)
            save_bundle(bundle, paths(asset, tf)[1])
            for h, e in bundle["horizons"].items():
                ho = e.get("holdout") or {}
                rows.append({"asset": asset, "expiry": f"{h * tf}m",
                             "accuracy": f"{e['oos_accuracy']:.1%}",
                             "exam_win": f"{ho['win_rate']:.1%}" if ho else "-",
                             "exam_trades": ho.get("trades", 0),
                             "verdict": f"TRADE >= {e['threshold']:.2f}" if e["threshold"] else "NO TRADE"})
                print(f"  {h * tf}m: {rows[-1]['verdict']} (accuracy {rows[-1]['accuracy']})", flush=True)
        except (Exception, SystemExit) as e:
            print(f"  skipped: {e}")
            rows.append({"asset": asset, "expiry": "-", "accuracy": "-", "exam_win": "-",
                         "exam_trades": 0, "verdict": f"ERROR: {str(e)[:60]}"})
        pd.DataFrame(rows).to_csv(out, index=False)  # save progress after every asset
    res = pd.DataFrame(rows)
    print("\n================ SCAN RESULTS ================")
    print(f"{'asset':<12}{'exp':>4}{'acc':>7}{'exam':>7}  verdict")
    for r in rows:
        print(f"{r['asset']:<12}{r['expiry']:>4}{r['accuracy']:>7}{r['exam_win']:>7}  {r['verdict']}")
    good = res[res["verdict"].str.startswith("TRADE")]
    print(f"\n{len(good)} asset/expiry combination(s) passed both the walk-forward test "
          f"and the final exam (break-even {1 / (1 + a.payout):.2%}).")
    if len(good):
        print(f"Next: forward-test one on PRACTICE:  python main.py live --tf {tf} --asset <ASSET> --dashboard")
    print(f"Saved to {out}")


def cmd_train(a):
    csv, model_path = paths(a.asset, a.tf)
    src_csv = Path(a.csv or csv)
    if not src_csv.exists():
        raise SystemExit(f"No data file {src_csv}. Run first:  python main.py collect --tf {a.tf} --asset {a.asset}")
    df = data.load_csv(src_csv)
    print(f"Training on {len(df)} candles ({df.index[0]} → {df.index[-1]})")
    bundle = train_and_validate(df, payout=a.payout, horizons=expiries_to_candles(a),
                                n_folds=a.folds, min_trades=a.min_trades, tf=a.tf)
    bundle["asset"] = a.asset
    model_path.parent.mkdir(exist_ok=True)
    save_bundle(bundle, model_path)
    summary = {f"{h * a.tf}m": (f"TRADE at conf >= {e['threshold']:.2f}" if e["threshold"] else "NO TRADE")
               for h, e in bundle["horizons"].items()}
    print(f"\nSaved model to {model_path}\nSummary: {json.dumps(summary)}")


def cmd_live(a):
    if a.strategy:
        return cmd_live_rule(a)
    _, model_path = paths(a.asset, a.tf)
    if not model_path.exists():
        raise SystemExit(f"No model {model_path}. Run first:  python main.py train --tf {a.tf} --asset {a.asset}")
    bundle = load_bundle(model_path)
    src = data.IQOptionSource(account=a.account).connect()
    publish = None
    if a.dashboard:
        from otcagent import dashboard
        dashboard.serve(a.host, a.port)
        publish = dashboard.publish
    log = Path("signals_log.jsonl")

    def on_signal(sig):
        print(format_signal(sig), flush=True)
        with log.open("a") as fh:
            fh.write(json.dumps(sig) + "\n")
        if publish:
            publish(sig)

    print(f"Live on {a.asset}. Signals appear 1 s after each {a.tf}-min candle closes. Ctrl+C to stop.")
    run_live(src, a.asset, bundle, on_signal)


def cmd_live_rule(a):
    from otcagent.signals import run_rule_live
    from otcagent.strategies import STRATEGIES
    if a.strategy not in STRATEGIES:
        raise SystemExit(f"unknown strategy {a.strategy}. See: python main.py strategy --list")
    fn, desc = STRATEGIES[a.strategy]
    expiry = a.expiry or a.tf
    from otcagent.tracker import Tracker
    src = data.IQOptionSource(account=a.account).connect()
    log = Path("rule_signals_log.jsonl")
    tracker = Tracker(a.tf, "results_log.jsonl", a.payout)
    publish = None
    if a.dashboard:
        from otcagent import dashboard
        dashboard.serve(a.host, a.port)
        publish = dashboard.publish

    def on_signal(sig):
        with log.open("a") as fh:
            fh.write(json.dumps(sig) + "\n")

    print(f"Live rule '{a.strategy}' on {a.asset}, {a.tf}-min candles, expiry {expiry} min.\n"
          f"  {desc}\n  PRACTICE account only. Ctrl+C to stop.\n"
          f"  Each signal is scored automatically at expiry (WIN/LOSS + running score); "
          f"results go to results_log.jsonl")
    run_rule_live(src, a.asset, fn, a.tf, expiry, on_signal, tracker=tracker,
                  tag=f"{a.strategy} {expiry}m", publish=publish)


def cmd_strategy(a):
    import pandas as pd
    from otcagent.strategies import STRATEGIES, test_strategy
    if a.list:
        for k, (_, desc) in STRATEGIES.items():
            print(f"{k:<22} {desc}")
        return
    names = list(STRATEGIES) if a.name == "all" else [x.strip() for x in a.name.split(",")]
    bad = [n for n in names if n not in STRATEGIES]
    if bad:
        raise SystemExit(f"unknown strategy {bad}. See: python main.py strategy --list")
    if a.assets:
        assets = [x.strip() for x in a.assets.split(",")]
    else:
        assets = sorted(p.name[: -len(f"_{a.tf}m.csv")] for p in Path("data").glob(f"*_{a.tf}m.csv"))
    if not assets:
        raise SystemExit(f"No {a.tf}-min data in data/. Run collect or scan with --tf {a.tf} first.")
    horizons = expiries_to_candles(a)
    frames = {}
    for asset in assets:
        try:
            frames[asset] = data.load_csv(paths(asset, a.tf)[0])
        except (Exception, SystemExit) as e:
            print(f"skip {asset}: {e}")
    print(f"Testing {len(names)} strateg{'y' if len(names) == 1 else 'ies'} on {len(frames)} assets, "
          f"{a.tf}-min candles, break-even {1 / (1 + a.payout):.2%}")
    import pandas as pd_
    since = pd_.Timestamp(a.since, tz="UTC") if a.since else None
    if since is not None:
        print(f"Only trades from {since.date()} on are counted (signals still use earlier "
              f"candles for context). This whole period is the exam.")
    rows = []
    for name in names:
        fn, desc = STRATEGIES[name]
        for asset, df in frames.items():
            sig = fn(df)
            if since is not None:
                keep = df.index >= since
                if keep.sum() < 50:
                    continue
                df_t, sig_t, exam = df[keep], sig[keep], 1.0
            else:
                df_t, sig_t, exam = df, sig, 0.2
            for r_ in test_strategy(df_t, sig_t, horizons, a.tf, a.payout, exam_frac=exam):
                rows.append({"strategy": name, "asset": asset, **r_})
    res = pd.DataFrame(rows)
    res.to_csv(f"strategy_results_{a.tf}m.csv", index=False)
    print(f"\n{'strategy':<22}{'exp':>4}{'trades':>8}{'win%':>7}{'pass':>5}")
    for (name, exp), g in res.groupby(["strategy", "expiry"], sort=False):
        n, w = g["trades"].sum(), g["wins"].sum()
        print(f"{name:<22}{exp:>4}{n:>8}{(w / n if n else 0):>7.1%}{int(g['pass'].sum()):>3}/{len(g)}")
    good = res[res["pass"]]
    print(f"\n(win% = all assets together; pass = assets where it beat break-even with "
          f"95% confidence AND on the newest 20%)")
    if len(good):
        print("\nPASSED:")
        for _, g in good.iterrows():
            print(f"  {g['strategy']} {g['asset']} {g['expiry']}: {g['win_rate']:.1%} over "
                  f"{g['trades']} trades, newest 20%: {g['exam_win_rate']:.1%}")
        print("Several strategies x assets were tested - re-check any pass on NEW data before trusting it.")
    else:
        print("No strategy/asset/expiry beat break-even.")
    print(f"Full details: strategy_results_{a.tf}m.csv")


def forex_model_path(tf):
    return Path("models") / f"FOREX_AGENT_{tf}m.pkl"


def real_pairs_with_data(tf):
    return sorted(p.name[: -len(f"_{tf}m.csv")] for p in Path("data").glob(f"*_{tf}m.csv")
                  if "-OTC" not in p.name and p.name[: -len(f"_{tf}m.csv")] in REAL_SCAN)


def cmd_forex(a):
    from otcagent.forex_agent import train_forex
    pairs = [x.strip() for x in a.assets.split(",")] if a.assets else REAL_SCAN
    if a.download:
        src = data.IQOptionSource(account=a.account).connect()
        for p_ in pairs:
            try:
                collect_asset(src, p_, default_days(a), a.tf)
            except (Exception, SystemExit) as e:
                print(f"  skipped {p_}: {e}")
    have = set(real_pairs_with_data(a.tf))
    pairs = [p_ for p_ in pairs if p_ in have]
    if len(pairs) < 2:
        raise SystemExit(f"Need data for at least 2 real pairs ({a.tf}-min). Run with --download, "
                         f"or: python main.py scan --market real --tf {a.tf}")
    frames = {p_: data.load_csv(paths(p_, a.tf)[0]) for p_ in pairs}
    bundle = train_forex(frames, tf=a.tf, horizons=expiries_to_candles(a), payout=a.payout,
                         n_folds=a.folds, max_rows=a.max_rows)
    Path("models").mkdir(exist_ok=True)
    save_bundle(bundle, forex_model_path(a.tf))
    print("\n================ FOREX AGENT SUMMARY ================")
    for h, e in bundle["horizons"].items():
        ex = e.get("exam") or {}
        verdict = f"TRADE all pairs at conf >= {e['threshold']:.2f}" if e["threshold"] else "NO TRADE (all pairs)"
        if e.get("pair_thresholds"):
            verdict += " | pairs that passed: " + ", ".join(
                f"{k} >= {v['threshold']:.2f} ({v['win_rate']:.0%}/{v['trades']})" for k, v in e["pair_thresholds"].items())
        extra = f", exam {ex['win_rate']:.1%} over {ex['trades']}" if ex else ""
        print(f"  {h * a.tf:>3}m: {verdict}  (walk-forward accuracy {e['oos_accuracy']:.1%}{extra})")
    print(f"Break-even at {a.payout:.0%} payout: {1 / (1 + a.payout):.2%}. Model: {forex_model_path(a.tf)}")


def cmd_forex_live(a):
    import time as _t
    from otcagent.forex_agent import analyse, format_analysis
    mp = forex_model_path(a.tf)
    if not mp.exists():
        raise SystemExit(f"No model {mp}. Run first:  python main.py forex --tf {a.tf}")
    bundle = load_bundle(mp)
    if a.asset not in bundle["pairs"]:
        print(f"note: {a.asset} was not in training ({', '.join(bundle['pairs'])}); "
              "analysis still works but is less reliable.")
    src = data.IQOptionSource(account=a.account).connect()
    publish = None
    if a.dashboard:
        from otcagent import dashboard
        dashboard.serve(a.host, a.port)
        publish = dashboard.publish
    from otcagent.tracker import Tracker
    tracker = Tracker(a.tf, "results_log.jsonl", bundle["payout"])
    sec = 60 * a.tf
    lookback = 3000 if a.tf == 1 else 2000
    log = Path("forex_signals_log.jsonl")
    print(f"Forex agent live on {a.asset}, {a.tf}-min candles. Ctrl+C to stop.")
    if a.show_all:
        print(f"  --show-all: the AI's lean is also shown as a PAPER signal whenever its confidence "
              f">= {a.show_all:.2f}, even if that expiry did not pass the backtest. These are marked "
              "'(unverified)' and scored separately.")
    while True:
        _t.sleep(sec - (_t.time() % sec) + 1.0)
        try:
            df = src.candles(a.asset, lookback, size=sec)
            if len(df) and df.index[-1].timestamp() > _t.time() - sec:
                df = df.iloc[:-1]
            if not len(df) or df.index[-1].timestamp() < _t.time() - 3 * sec:
                print(f"{a.asset}: market closed / no fresh candles - no signals", flush=True)
                if publish:
                    from otcagent.signals import market_closed_text
                    publish({"status": market_closed_text(a.asset)})
                continue
            for line in tracker.update(df, a.asset):
                print(line, flush=True)
            res = analyse(df, bundle, a.asset)
            stamp, price = df.index[-1], float(df["close"].iloc[-1])
            extra = []
            for k, sg in res["signals"].items():
                mins = int(k[:-1])
                if sg["action"] in ("CALL", "PUT"):
                    tracker.add(f"AI {k}", a.asset, stamp, price, sg["action"], mins)
                    extra.append(f"\a  >>> {sg['action']} {k} on your DEMO now")
                elif a.show_all and sg["confidence"] >= a.show_all:
                    tracker.add(f"AI {k} (unverified)", a.asset, stamp, price, sg["lean"], mins)
                    extra.append(f"  paper: {sg['lean']} {k} (unverified, conf {sg['confidence']:.0%})")
            res["scoreboard"] = tracker.board()
            print(format_analysis(res), flush=True)
            for line in extra:
                print(line, flush=True)
            with log.open("a") as fh:
                fh.write(json.dumps(res) + "\n")
            if publish:
                publish(res)
        except Exception as e:  # keep running through network hiccups
            print(f"live error: {e}", flush=True)


def cmd_demo(a):
    print("Demo on synthetic candles with a hidden pattern (no login needed).")
    df = data.synthetic(n=a.rows, edge=0.65, seed=3)
    bundle = train_and_validate(df, payout=a.payout, n_folds=4)
    print("\nSignal for the last candle:\n" + format_signal(generate(df, bundle, "SYNTH-OTC")))


def main():
    p = argparse.ArgumentParser(description="OTC CALL/PUT signal agent")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("collect", "train", "live"):
        s = sub.add_parser(name)
        s.add_argument("--asset", default="EURUSD-OTC")
        s.add_argument("--account", default="PRACTICE", choices=["PRACTICE", "REAL"])
        s.add_argument("--tf", type=int, choices=[1, 5], default=1,
                       help="candle size in minutes: 1 (expiries 1/2/3/5) or 5 (expiries 5/10/15)")
    sub.choices["collect"].add_argument("--days", type=float,
                                        help="default 10 (1-min) or 60 (5-min candles)")
    t = sub.choices["train"]
    t.add_argument("--csv", help="train from a CSV instead of data/<asset>_1m.csv")
    t.add_argument("--payout", type=float, default=0.85, help="0.85 = 85%% payout")
    t.add_argument("--folds", type=int, default=5)
    t.add_argument("--min-trades", type=int, default=100)
    t.add_argument("--expiries", help="minutes, default 1,2,3,5 (--tf 1) or 5,10,15 (--tf 5)")
    lv = sub.choices["live"]
    lv.add_argument("--dashboard", action="store_true")
    lv.add_argument("--host", default="127.0.0.1")
    lv.add_argument("--port", type=int, default=8080)
    lv.add_argument("--strategy", help="run a rule from 'strategy --list' instead of the AI model")
    lv.add_argument("--expiry", type=int, help="expiry in minutes to show with rule signals")
    lv.add_argument("--payout", type=float, default=0.80, help="for the live score in stakes")
    sc = sub.add_parser("scan", help="collect + test many OTC assets, list the ones with an edge")
    sc.add_argument("--market", choices=["otc", "real"], default="otc",
                    help="otc = broker OTC pairs (default), real = real forex pairs (Mon-Fri)")
    sc.add_argument("--assets", help="comma list, default: 12 pairs of the chosen --market")
    sc.add_argument("--tf", type=int, choices=[1, 5], default=1,
                    help="candle size in minutes: 1 (expiries 1/2/3/5) or 5 (expiries 5/10/15)")
    sc.add_argument("--days", type=float, help="default 10 (1-min) or 60 (5-min candles)")
    sc.add_argument("--payout", type=float, default=0.85)
    sc.add_argument("--folds", type=int, default=3)
    sc.add_argument("--expiries", help="minutes, default 1,2,3,5 (--tf 1) or 5,10,15 (--tf 5)")
    sc.add_argument("--account", default="PRACTICE", choices=["PRACTICE", "REAL"])
    sc.add_argument("--no-download", action="store_true", help="use CSVs already in data/")
    sc.add_argument("--resume", action="store_true", help="skip assets already in scan_results.csv")
    sc.add_argument("--verbose", action="store_true", help="print full reports")
    st = sub.add_parser("strategy", help="test rule-based strategies on downloaded candles")
    st.add_argument("--name", default="all", help="strategy name(s), comma list, or 'all'")
    st.add_argument("--list", action="store_true", help="show available strategies")
    st.add_argument("--assets", help="comma list, default: every asset with data for --tf")
    st.add_argument("--tf", type=int, choices=[1, 5], default=1)
    st.add_argument("--expiries", help="minutes, default 1,2,3,5 (--tf 1) or 5,10,15 (--tf 5)")
    st.add_argument("--payout", type=float, default=0.85)
    st.add_argument("--since", help="only count trades from this date (YYYY-MM-DD), e.g. new data")
    fx = sub.add_parser("forex", help="train + backtest the big forex agent on all real pairs")
    fx.add_argument("--tf", type=int, choices=[1, 5], default=1)
    fx.add_argument("--expiries", help="minutes, default 1,2,3,5 (--tf 1) or 5,10,15 (--tf 5)")
    fx.add_argument("--assets", help="comma list, default: the 12 real pairs")
    fx.add_argument("--payout", type=float, default=0.80)
    fx.add_argument("--folds", type=int, default=3)
    fx.add_argument("--max-rows", type=int, default=60000, help="most recent rows used to train")
    fx.add_argument("--download", action="store_true", help="download fresh candles first")
    fx.add_argument("--days", type=float, help="history to download, default 10 (1m) / 60 (5m)")
    fx.add_argument("--account", default="PRACTICE", choices=["PRACTICE", "REAL"])
    fl = sub.add_parser("forex-live", help="live analysis + CALL/PUT from the forex agent")
    fl.add_argument("--asset", default="EURUSD")
    fl.add_argument("--tf", type=int, choices=[1, 5], default=1)
    fl.add_argument("--account", default="PRACTICE", choices=["PRACTICE", "REAL"])
    fl.add_argument("--dashboard", action="store_true")
    fl.add_argument("--host", default="127.0.0.1")
    fl.add_argument("--port", type=int, default=8080)
    fl.add_argument("--show-all", type=float, nargs="?", const=0.55, default=None,
                    help="also paper-track the AI lean when confidence >= this (default 0.55)")
    d = sub.add_parser("demo")
    d.add_argument("--rows", type=int, default=15000)
    d.add_argument("--payout", type=float, default=0.85)
    a = p.parse_args()
    {"collect": cmd_collect, "train": cmd_train, "live": cmd_live, "demo": cmd_demo,
     "scan": cmd_scan, "strategy": cmd_strategy, "forex": cmd_forex, "forex-live": cmd_forex_live}[a.cmd](a)


if __name__ == "__main__":
    main()
