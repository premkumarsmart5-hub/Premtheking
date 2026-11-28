"""Phone-friendly animated web dashboard (standard library only - works in Termux).

Open http://127.0.0.1:8080 in your phone browser while a live command runs with --dashboard.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = {"latest": None, "history": []}
LOCK = threading.Lock()

PAGE = r"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name=theme-color content="#0b0e14">
<title>Forex Signal Agent</title>
<style>
:root{--bg:#0b0e14;--card:#141925;--card2:#1a2030;--line:#252c3d;--txt:#e9edf5;--mut:#8b95a8;
--call:#22c55e;--put:#ef4444;--warn:#f59e0b;--acc:#60a5fa}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--txt);font:15px/1.4 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
min-height:100vh;padding:16px 16px 40px;overflow-x:hidden}
body:before{content:"";position:fixed;inset:-40%;z-index:-1;pointer-events:none;
background:radial-gradient(circle at 30% 20%,#1d4ed833,transparent 40%),radial-gradient(circle at 80% 70%,#7c3aed22,transparent 45%);
animation:drift 18s ease-in-out infinite alternate}
@keyframes drift{to{transform:translate(6%,4%) rotate(8deg)}}
.top{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:14px}
.brand{display:flex;align-items:center;gap:10px}
.logo{width:36px;height:36px;border-radius:11px;display:grid;place-items:center;font-weight:800;
background:linear-gradient(135deg,#3b82f6,#8b5cf6);box-shadow:0 6px 20px #3b82f655}
h1{font-size:17px;font-weight:700;letter-spacing:.2px}
.sub{color:var(--mut);font-size:12.5px}
.live{display:flex;align-items:center;gap:7px;font-size:12px;color:var(--mut);background:var(--card);
padding:6px 11px;border-radius:99px;border:1px solid var(--line)}
.dot{width:8px;height:8px;border-radius:50%;background:var(--call);box-shadow:0 0 0 0 #22c55e99;animation:ping 1.6s infinite}
.dot.off{background:var(--warn);box-shadow:none;animation:none}
@keyframes ping{0%{box-shadow:0 0 0 0 #22c55e88}70%{box-shadow:0 0 0 10px #22c55e00}100%{box-shadow:0 0 0 0 #22c55e00}}
.card{background:linear-gradient(180deg,var(--card),#121723);border:1px solid var(--line);border-radius:20px;
padding:16px;margin-bottom:12px;animation:rise .5s ease both}
@keyframes rise{from{opacity:0;transform:translateY(10px)}}
.hero{display:flex;align-items:center;justify-content:space-between;gap:12px}
.pair{font-size:22px;font-weight:800;letter-spacing:.5px}
.price{font-size:28px;font-weight:700;font-variant-numeric:tabular-nums;transition:color .4s}
.price.up{color:var(--call)}.price.dn{color:var(--put)}
.meta{color:var(--mut);font-size:12.5px;margin-top:2px}
.ring{position:relative;width:74px;height:74px;flex:none}
.ring svg{transform:rotate(-90deg)}
.ring .bg{stroke:var(--line)}.ring .fg{stroke:var(--acc);stroke-linecap:round;transition:stroke-dashoffset 1s linear}
.ring b{position:absolute;inset:0;display:grid;place-items:center;font-size:16px;font-variant-numeric:tabular-nums}
.ring small{position:absolute;bottom:-16px;left:0;right:0;text-align:center;color:var(--mut);font-size:10.5px}
.sig{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px}
.s{position:relative;overflow:hidden;border-radius:18px;padding:16px;text-align:center;background:var(--card2);
border:1px solid var(--line);transition:transform .3s}
.s .exp{color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:1px}
.s .act{font-size:30px;font-weight:900;margin:6px 0 2px;letter-spacing:1px}
.s .why{color:var(--mut);font-size:12px;min-height:16px}
.s.CALL{border-color:#22c55e88;background:radial-gradient(circle at 50% 0,#22c55e33,var(--card2) 70%);animation:glowG 1.8s ease-in-out infinite}
.s.PUT{border-color:#ef444488;background:radial-gradient(circle at 50% 100%,#ef444433,var(--card2) 70%);animation:glowR 1.8s ease-in-out infinite}
.s.CALL .act{color:var(--call)}.s.PUT .act{color:var(--put)}.s.NO .act{color:#5d6679;font-size:19px;white-space:nowrap}
@keyframes glowG{50%{box-shadow:0 0 28px #22c55e55}}@keyframes glowR{50%{box-shadow:0 0 28px #ef444455}}
.arrow{display:inline-block;animation:bob 1s ease-in-out infinite}
.PUT .arrow{animation-name:bobd}
@keyframes bob{50%{transform:translateY(-5px)}}@keyframes bobd{50%{transform:translateY(5px)}}
.bar{height:6px;border-radius:9px;background:var(--line);overflow:hidden;margin-top:10px}
.bar i{display:block;height:100%;border-radius:9px;background:linear-gradient(90deg,#3b82f6,#a78bfa);
transition:width .8s cubic-bezier(.2,.8,.2,1)}
.flash{position:fixed;inset:0;pointer-events:none;opacity:0;z-index:9}
.flash.go{animation:flash 1.2s ease-out}
@keyframes flash{0%{opacity:.55}100%{opacity:0}}
.toast{position:fixed;left:50%;top:14px;transform:translate(-50%,-140%);z-index:10;padding:12px 18px;border-radius:14px;
font-weight:800;font-size:16px;box-shadow:0 12px 30px #0008;transition:transform .45s cubic-bezier(.2,.9,.2,1.2)}
.toast.show{transform:translate(-50%,0)}
h2{font-size:13px;color:var(--mut);font-weight:600;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px}
.score{display:flex;align-items:center;gap:16px}
.big{font-size:30px;font-weight:800;font-variant-numeric:tabular-nums}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;flex:1}
.stat{background:var(--card2);border-radius:12px;padding:8px;text-align:center}
.stat b{display:block;font-size:18px;font-variant-numeric:tabular-nums}
.stat span{color:var(--mut);font-size:11px}
.g{color:var(--call)}.r{color:var(--put)}
.feed{display:flex;flex-direction:column;gap:8px}
.row{display:flex;align-items:center;justify-content:space-between;gap:8px;background:var(--card2);border-radius:12px;
padding:10px 12px;animation:slide .45s ease both;font-size:13.5px}
@keyframes slide{from{opacity:0;transform:translateX(-14px)}}
.badge{font-weight:800;font-size:12px;padding:4px 9px;border-radius:99px;white-space:nowrap;flex:none}
.badge.WIN{background:#22c55e22;color:var(--call)}.badge.LOSS{background:#ef444422;color:var(--put)}.badge.TIE{background:#f59e0b22;color:var(--warn)}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{font-size:12px;padding:5px 10px;border-radius:99px;background:var(--card2);border:1px solid var(--line);animation:pop .35s ease both}
.chip.b{color:var(--call);border-color:#22c55e55}.chip.s{color:var(--put);border-color:#ef444455}
@keyframes pop{from{opacity:0;transform:scale(.85)}}
.kv{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px}
.kv div{background:var(--card2);border-radius:12px;padding:8px 10px;font-size:12.5px}
.kv span{display:block;color:var(--mut);font-size:11px}
.closed{text-align:center;padding:30px 16px}
.moon{font-size:56px;display:inline-block;animation:float 4s ease-in-out infinite}
@keyframes float{50%{transform:translateY(-8px) rotate(-6deg)}}
.count{font-size:34px;font-weight:800;font-variant-numeric:tabular-nums;margin:10px 0 4px;
background:linear-gradient(90deg,#60a5fa,#a78bfa);-webkit-background-clip:text;background-clip:text;color:transparent}
.empty{color:var(--mut);font-size:13px;text-align:center;padding:8px}
.btn{background:var(--card2);border:1px solid var(--line);color:var(--txt);border-radius:99px;padding:6px 12px;font-size:12px}
.foot{color:#f59e0bcc;font-size:12px;text-align:center;margin-top:8px}
.hidden{display:none}
</style></head><body>
<div class=flash id=flash></div><div class=toast id=toast></div>
<div class=top>
  <div class=brand><div class=logo>FX</div><div><h1>Forex Signal Agent</h1><div class=sub id=sub>connecting…</div></div></div>
  <div class=live><span class="dot off" id=dot></span><span id=livetxt>waiting</span></div>
</div>

<div id=closedView class="card closed hidden">
  <div class=moon>🌙</div>
  <h2 style="margin-top:12px">Market closed</h2>
  <div class=count id=reopen>--:--:--</div>
  <div class=sub>until real forex reopens (Sunday 22:00 UTC · Monday 3:30 AM India)</div>
  <div class=sub style="margin-top:10px">Signals start automatically. Best hours: 5:30–9:30 PM India time.</div>
</div>

<div id=liveView class=hidden>
  <div class=card>
    <div class=hero>
      <div><div class=pair id=pair>—</div><div class=price id=price>—</div><div class=meta id=meta></div></div>
      <div class=ring><svg width=74 height=74><circle class=bg cx=37 cy=37 r=31 fill=none stroke-width=7 />
        <circle class=fg id=arc cx=37 cy=37 r=31 fill=none stroke-width=7 stroke-dasharray=194.8 stroke-dashoffset=0 /></svg>
        <b id=secs>--</b><small>next candle</small></div>
    </div>
  </div>
  <div class=card><h2>Signal</h2><div class=sig id=sig></div></div>
  <div class=card><h2>Live score <button class=btn id=snd style="float:right">🔔 enable sound</button></h2>
    <div id=score class=empty>No finished signals yet — every signal is scored automatically at expiry.</div></div>
  <div class=card><h2>Recent results</h2><div class=feed id=feed><div class=empty>Results appear here.</div></div></div>
  <div class=card><h2>Analysis</h2><div class=chips id=chips></div><div class=kv id=kv></div></div>
</div>
<div class=foot>Demo account only · signals are probabilities, not guarantees</div>

<script>
const $=id=>document.getElementById(id);
let lastKey="",lastPrice=null,lastRes="",tf=1,sound=false,actx=null;
$('snd').onclick=()=>{sound=!sound;$('snd').textContent=sound?'🔔 sound on':'🔕 sound off';
  if(sound&&!actx){try{actx=new (window.AudioContext||window.webkitAudioContext)();}catch(e){}}};
function beep(up){if(!sound||!actx)return;const o=actx.createOscillator(),g=actx.createGain();
  o.frequency.value=up?880:440;o.connect(g);g.connect(actx.destination);g.gain.setValueAtTime(.25,actx.currentTime);
  g.gain.exponentialRampToValueAtTime(.001,actx.currentTime+.6);o.start();o.stop(actx.currentTime+.6);}
function alertSig(a,exp){const up=a==='CALL';const f=$('flash');f.style.background=up?'#22c55e':'#ef4444';
  f.classList.remove('go');void f.offsetWidth;f.classList.add('go');
  const t=$('toast');t.textContent=(up?'▲ CALL ':'▼ PUT ')+exp+' — place it on your demo';
  t.style.background=up?'#16a34a':'#dc2626';t.classList.add('show');setTimeout(()=>t.classList.remove('show'),4500);
  if(navigator.vibrate)navigator.vibrate([200,80,200]);beep(up);}
function tick(){const now=Date.now()/1000,p=tf*60,left=Math.ceil(p-(now%p));
  $('secs').textContent=left;$('arc').style.strokeDashoffset=194.8*(1-left/p);
  const d=new Date(),day=d.getUTCDay(),h=d.getUTCHours();
  let open=new Date(Date.UTC(d.getUTCFullYear(),d.getUTCMonth(),d.getUTCDate(),22,0,0));
  const add=day===0?(h>=22?7:0):(7-day);open.setUTCDate(open.getUTCDate()+add);
  let s=Math.max(0,(open-d)/1000|0);const hh=s/3600|0,mm=(s%3600)/60|0,ss=s%60;
  $('reopen').textContent=(hh>=24?Math.floor(hh/24)+'d ':'')+String(hh%24).padStart(2,'0')+':'+String(mm).padStart(2,'0')+':'+String(ss).padStart(2,'0');}
setInterval(tick,1000);tick();
function esc(x){return String(x).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
function render(d){
  if(!d){$('sub').textContent='waiting for the first candle…';return;}
  if(d.status&&!d.signals){$('closedView').classList.remove('hidden');$('liveView').classList.add('hidden');
    $('dot').className='dot off';$('livetxt').textContent='closed';$('sub').textContent=d.status.split('(')[0];return;}
  $('closedView').classList.add('hidden');$('liveView').classList.remove('hidden');
  $('dot').className='dot';$('livetxt').textContent='LIVE';tf=d.tf||1;
  $('pair').textContent=d.asset;const pe=$('price');pe.textContent=d.price.toFixed(5);
  if(lastPrice!==null){pe.className='price '+(d.price>lastPrice?'up':d.price<lastPrice?'dn':'');}lastPrice=d.price;
  const ind=d.indicators||{};
  $('sub').textContent='candle '+d.candle_time.slice(11,16)+' UTC'+(ind.session?' · '+ind.session:'');
  $('meta').textContent=[ind.regime,ind['higher TF']?'HTF '+ind['higher TF']:'',ind.strategy?'rule: '+ind.strategy:''].filter(Boolean).join(' · ');
  let h='';for(const[k,v]of Object.entries(d.signals||{})){const c=v.action==='NO TRADE'?'NO':v.action;
    const arrow=c==='CALL'?'<span class=arrow>▲</span> ':c==='PUT'?'<span class=arrow>▼</span> ':'';
    const conf=v.confidence?Math.round(v.confidence*100):0;
    h+=`<div class="s ${c}"><div class=exp>${esc(k)} expiry</div><div class=act>${arrow}${c==='NO'?'NO TRADE':c}</div>
      <div class=why>${esc(v.reason||'')}</div>${v.confidence&&v.confidence<1?`<div class=bar><i style="width:${conf}%"></i></div>
      <div class=why style="margin-top:4px">AI lean ${esc(v.lean)} ${conf}%</div>`:''}</div>`;
    const key=d.candle_time+k+v.action;if((c==='CALL'||c==='PUT')&&key!==lastKey){lastKey=key;alertSig(c,k);}}
  $('sig').innerHTML=h;
  const sc=d.score||{};const tags=Object.keys(sc);
  if(tags.length){$('score').className='';$('score').innerHTML=tags.map(t=>{const s=sc[t],wr=Math.round(s.win_rate*100);
    const col=s.win_rate>=0.556?'g':s.win_rate<0.5?'r':'';
    return `<div class=score style="margin-bottom:10px"><div class=ring style="width:74px;height:74px"><svg width=74 height=74>
      <circle class=bg cx=37 cy=37 r=31 fill=none stroke-width=7 /><circle cx=37 cy=37 r=31 fill=none stroke-width=7
      stroke="${s.win_rate>=0.556?'#22c55e':s.win_rate<0.5?'#ef4444':'#f59e0b'}" stroke-linecap=round stroke-dasharray=194.8
      stroke-dashoffset="${194.8*(1-s.win_rate)}" style="transition:stroke-dashoffset 1s"/></svg><b class=${col}>${wr}%</b></div>
      <div class=stats><div class=stat><b class=g>${s.wins}</b><span>wins</span></div><div class=stat><b class=r>${s.losses}</b><span>losses</span></div>
      <div class=stat><b class=${s.profit>=0?'g':'r'}>${s.profit>=0?'+':''}${s.profit.toFixed(2)}</b><span>stakes</span></div></div></div>
      <div class=sub>${esc(t)} · break-even 55.6% · ${s.wins+s.losses} scored${d.pending?` · ${d.pending} open`:''}</div>`}).join('');}
  const res=d.results||[];if(res.length){const k=JSON.stringify(res[0]);if(k!==lastRes){lastRes=k;
    $('feed').innerHTML=res.map(r=>`<div class=row><span><b class=${r.action==='CALL'?'g':'r'}>${r.action==='CALL'?'▲':'▼'} ${r.action}</b>
      <span class=sub> ${esc(r.time.slice(11,16))} · ${r.expiry}m · ${r.entry.toFixed(5)}→${r.close.toFixed(5)}</span></span>
      <span class="badge ${r.result}">${r.result==='WIN'?'✔ WIN':r.result==='LOSS'?'✘ LOSS':'= TIE'}</span></div>`).join('');}}
  const chips=[];(d.patterns||[]).forEach(p=>chips.push(`<span class="chip ${/bull/.test(p)?'b':/bear/.test(p)?'s':''}">◆ ${esc(p)}</span>`));
  if(ind.bullish&&ind.bullish!=='-')ind.bullish.split(', ').forEach(x=>chips.push(`<span class="chip b">▲ ${esc(x)}</span>`));
  if(ind.bearish&&ind.bearish!=='-')ind.bearish.split(', ').forEach(x=>chips.push(`<span class="chip s">▼ ${esc(x)}</span>`));
  $('chips').innerHTML=chips.join('')||'<div class=empty>No patterns or indicator signals on this candle.</div>';
  const kv=[['RSI 14',ind.RSI14],['ADX',ind.ADX],['Trend votes',ind['trend votes']],['Reversal votes',ind['reversal votes']]].filter(x=>x[1]!==undefined);
  $('kv').innerHTML=kv.map(([k,v])=>`<div><span>${k}</span>${esc(v)}</div>`).join('');}
async function poll(){try{const r=await fetch('/api/latest',{cache:'no-store'});render(await r.json());}
  catch(e){$('dot').className='dot off';$('livetxt').textContent='offline';$('sub').textContent='agent not running? start it in Termux';}}
poll();setInterval(poll,2000);
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
