"""Tiny phone-friendly web dashboard (standard library only - works in Termux).

Open http://127.0.0.1:8080 in your phone browser while `main.py live --dashboard` runs.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = {"latest": None, "history": []}
LOCK = threading.Lock()

PAGE = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Forex Signal Agent</title><style>
body{font-family:system-ui,sans-serif;background:#0f1115;color:#e8e8e8;margin:0;padding:16px}
h1{font-size:18px;margin:0 0 4px}.sub{color:#9aa;font-size:13px;margin-bottom:14px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px}
.card{background:#1a1d24;border-radius:12px;padding:14px;text-align:center}
.exp{color:#9aa;font-size:13px}.act{font-size:24px;font-weight:700;margin:6px 0}
.CALL{color:#22c55e}.PUT{color:#ef4444}.NO{color:#888}.small{font-size:12px;color:#9aa}
.box{background:#1a1d24;border-radius:12px;padding:12px;margin-top:12px;font-size:13px}
.warn{color:#f59e0b;font-size:12px;margin-top:14px}
</style></head><body>
<h1 id=t>Forex Signal Agent</h1><div class=sub id=s>waiting for the next candle…</div>
<div class=grid id=g></div><div class=box id=sb></div><div class=box id=p></div><div class=box id=i></div>
<div class=warn>Signals are probabilities, not guarantees. Practice account first.</div>
<script>
async function tick(){try{const r=await fetch('/api/latest');const d=await r.json();if(!d)return;
if(d.status){document.getElementById('s').textContent=d.status;if(!d.signals)return;}
document.getElementById('t').textContent=d.asset+' @ '+d.price.toFixed(5);
document.getElementById('s').textContent='candle '+d.candle_time+' · break-even '+(d.breakeven*100).toFixed(1)+'%';
let h='';for(const[k,v]of Object.entries(d.signals)){const c=v.action==='NO TRADE'?'NO':v.action;
h+=`<div class=card><div class=exp>${k} expiry</div><div class="act ${c}">${v.action}</div>
<div>${(v.confidence*100).toFixed(0)}% ${v.lean}</div><div class=small>${v.reason}</div>
<div class=small>backtest: ${v.backtest_win_rate?(v.backtest_win_rate*100).toFixed(1)+'% / '+v.backtest_trades+' trades':'n/a'}</div></div>`}
document.getElementById('g').innerHTML=h;
document.getElementById('p').textContent='Patterns: '+(d.patterns.join(', ')||'none');
document.getElementById('i').textContent=Object.entries(d.indicators).map(([k,v])=>k+': '+v).join(' · ');
const sb=d.scoreboard||{};document.getElementById('sb').textContent=Object.keys(sb).length?
'Live score: '+Object.entries(sb).map(([k,v])=>k+' '+v).join(' | '):'Live score: no finished signals yet';
}catch(e){}}
tick();setInterval(tick,3000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        b = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        with LOCK:
            if self.path == "/api/latest":
                return self._send(200, json.dumps(STATE["latest"]), "application/json")
            if self.path == "/api/history":
                return self._send(200, json.dumps(STATE["history"][-100:]), "application/json")
        if self.path in ("/", "/index.html"):
            return self._send(200, PAGE, "text/html; charset=utf-8")
        self._send(404, "not found", "text/plain")


def publish(signal: dict):
    with LOCK:
        STATE["latest"] = signal
        STATE["history"].append(signal)
        STATE["history"] = STATE["history"][-500:]


def serve(host="127.0.0.1", port=8080):
    srv = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"Dashboard: http://{host}:{port}")
    return srv
