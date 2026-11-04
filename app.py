"""Uber-style marketplace brain: simulator + demand forecaster + multi-agent assistant API.
Run:  pip install -r requirements.txt && python app.py   ->  http://localhost:5000
Optional: set ANTHROPIC_API_KEY to let an LLM phrase the answers (facts always come from the tools)."""
import os, math
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="static")
N, DRIVERS, SCALE, STADIUM = 8, 40, 0.06, (6, 6)

def label(r, c):
    if (r, c) == STADIUM: return f"Stadium ({r},{c})"
    if r < 3 and c < 3: return f"Office district ({r},{c})"
    if 4 <= r <= 6 and c < 3: return f"Night market ({r},{c})"
    return f"Zone ({r},{c})"

# ---------- Simulator: ground-truth demand ----------
def base(h):
    g = np.ones((N, N)) * (1 + 1.5 * math.exp(-((h % 24 - 8) ** 2) / 20))
    g[:3, :3] += 6 * math.exp(-((h % 24 - 9) ** 2) / 8) + 6 * math.exp(-((h % 24 - 18) ** 2) / 8)
    g[4:7, :3] += 7 * math.exp(-((h % 24 - 21) ** 2) / 10)
    return g * SCALE * 4

def event_boost():
    b = np.zeros((N, N)); r, c = STADIUM
    b[r, c] = 9
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        if 0 <= r + dr < N and 0 <= c + dc < N: b[r + dr, c + dc] = 2
    return b

def true_demand(h, event, surge):
    d = base(h) + (event_boost() if event else 0)
    return d * (1.6 if surge else 1)

# ---------- Forecaster: seasonal average + live signals, with stale fallback ----------
def forecast(h, event, surge, model_down=False):
    if model_down:                       # stale model: last week's average, blind to events/surges
        return base(h)
    return (base(h) + (event_boost() if event else 0)) * (1.6 if surge else 1)

def scores(fc, supply): return fc / (1 + supply)

def guide(r, c, h, event, surge, down, supply, radius=3):
    s = scores(forecast(h, event, surge, down), supply); best = (-1, r, c)
    for i in range(N):
        for j in range(N):
            if abs(i - r) + abs(j - c) <= radius and s[i, j] > best[0]: best = (s[i, j], i, j)
    _, i, j = best; fc = forecast(h, event, surge, down)
    return dict(zone=label(i, j), r=i, c=j, distance=abs(i - r) + abs(j - c),
                expected_requests_15min=round(float(fc[i, j]), 1), drivers_there=int(supply[i, j]),
                stay_here=(i, j) == (r, c), stale_data=down)

# ---------- Simulation: baseline vs forecast-guided drivers ----------
def simulate(strategy, seed, event=True, surge=False, down=False, ticks=48, start=16):
    rng = np.random.default_rng(seed)
    pos = rng.integers(0, N, (DRIVERS, 2)); busy = np.zeros(DRIVERS, int)
    idle = trips = lost = 0
    for t in range(ticks):
        h = start + t * 0.25
        ev = event and 21 <= h < 22.5
        req = rng.poisson(true_demand(h, ev, surge))
        busy = np.maximum(busy - 1, 0)
        assigned = np.zeros(DRIVERS, bool)
        for r in range(N):
            for c in range(N):
                if req[r, c] == 0: continue
                ids = [k for k in range(DRIVERS) if busy[k] == 0 and not assigned[k] and tuple(pos[k]) == (r, c)]
                n = min(len(ids), req[r, c]); lost += req[r, c] - n
                for k in ids[:n]:
                    assigned[k] = True; busy[k] = 2; trips += 1; pos[k] = rng.integers(0, N, 2)
        free = [k for k in range(DRIVERS) if busy[k] == 0 and not assigned[k]]
        idle += len(free)
        if strategy == "forecast":
            soon = event and 20.5 <= h < 22.5            # event schedule is known ahead of time
            fc = forecast(h + 0.5, soon, surge, down)
            sup = np.zeros((N, N))
            for k in free: sup[tuple(pos[k])] += 1
            for k in free:
                r, c = pos[k]; s = scores(fc, sup); tgt = (r, c, -1)
                for i in range(N):
                    for j in range(N):
                        if abs(i - r) + abs(j - c) <= 5 and s[i, j] > tgt[2]: tgt = (i, j, s[i, j])
                sup[r, c] -= 1; nr, nc = r, c
                for _ in range(2):                       # drivers can cover 2 zones per 15 min
                    step_r = int(np.sign(tgt[0] - nr)); nr += step_r
                    if step_r == 0: nc += int(np.sign(tgt[1] - nc))
                pos[k] = (nr, nc); sup[nr, nc] += 1
        else:
            for k in free:
                if rng.random() < 0.5: pos[k] = np.clip(pos[k] + rng.integers(-1, 2, 2), 0, N - 1)
    return dict(idle_pct=100 * idle / (DRIVERS * ticks), trips=trips, lost=int(lost))

def supply_snapshot(seed=7): return np.random.default_rng(seed).multinomial(DRIVERS, [1 / N ** 2] * N ** 2).reshape(N, N)

def params(src):
    g = lambda k, d=False: str(src.get(k, d)).lower() in ("1", "true", "yes")
    return float(src.get("hour", 21)), g("event"), g("surge"), g("down")

# ---------- Multi-agent assistant ----------
DRIVER = {"D1": dict(name="Aman", trips_today=14, earnings=186.5, online_hours=6.5, tips=22.0)}
HELP = {"document": "Upload licence, RC and insurance in Profile > Documents; approval takes about 24 hours.",
        "payout": "Payouts reach your bank every Monday; instant cash-out is available in Wallet.",
        "rating": "Your rating is the average of your last 500 rated trips."}

def route(m):
    m = m.lower()
    if any(k in m for k in ("earn", "paid", "money", "income", "trips", "tips", "kamaya", "kamai")): return "earnings"
    if any(k in m for k in ("where", "go", "zone", "busy", "demand", "surge", "stadium", "event", "kahan", "jaun")): return "guidance"
    return "help"

def run_agent(agent, msg, ctx):
    h, ev, su, dn = ctx["h"], ctx["ev"], ctx["su"], ctx["dn"]; r, c = ctx["zone"]
    if agent == "earnings":
        d = DRIVER["D1"]; per = d["earnings"] / d["online_hours"]
        facts = dict(d, per_hour=round(per, 2))
        text = f"Today you did {d['trips_today']} trips and earned ${d['earnings']:.2f} (+${d['tips']:.2f} tips) in {d['online_hours']}h, about ${per:.2f}/hour."
    elif agent == "guidance":
        facts = guide(r, c, h, ev, su, dn, supply_snapshot())
        text = ("Stay where you are; " if facts["stay_here"] else f"Head to {facts['zone']}, {facts['distance']} zones away; ") + \
               f"about {facts['expected_requests_15min']} requests are expected in 15 min with {facts['drivers_there']} drivers there."
        if dn: text += " Note: the live model is down, so this uses last week's average and may miss events or surges."
    else:
        key = next((k for k in HELP if k in msg.lower()), None)
        facts = {"answer": HELP.get(key)}; text = HELP.get(key) or "I can help with earnings, where to drive, documents, payouts and ratings."
    return facts, text

def phrase_with_llm(msg, agent, facts, fallback):
    if not os.getenv("ANTHROPIC_API_KEY"): return fallback
    try:
        import anthropic
        r = anthropic.Anthropic().messages.create(
            model=os.getenv("MODEL", "claude-sonnet-4-6"), max_tokens=200,
            system="You are a ride-hailing driver assistant. Answer in 2 short sentences using ONLY the FACTS given. Never invent numbers. Mention if data is stale.",
            messages=[{"role": "user", "content": f"Agent: {agent}\nFACTS: {facts}\nDriver asked: {msg}"}])
        return r.content[0].text
    except Exception:
        return fallback                      # graceful fallback if the LLM is unavailable

# ---------- API ----------
@app.get("/")
def index(): return send_from_directory("static", "index.html")

@app.get("/api/state")
def state():
    h, ev, su, dn = params(request.args); r, c = int(request.args.get("r", 2)), int(request.args.get("c", 2))
    sup = supply_snapshot(); fc = forecast(h, ev, su, dn)
    return jsonify(forecast=np.round(fc, 2).tolist(), supply=sup.tolist(), best=guide(r, c, h, ev, su, dn, sup))

@app.get("/api/compare")
def compare():
    _, ev, su, dn = params(request.args); out = {}
    for s in ("random", "forecast"):
        runs = [simulate(s, seed, ev, su, dn) for seed in range(5)]
        out[s] = {k: round(float(np.mean([x[k] for x in runs])), 1) for k in runs[0]}
    out["idle_reduction_pct"] = round(100 * (out["random"]["idle_pct"] - out["forecast"]["idle_pct"]) / out["random"]["idle_pct"], 1)
    return jsonify(out)

@app.post("/api/chat")
def chat():
    b = request.get_json(force=True, silent=True) or {}
    msg = (b.get("message") or "").strip()
    if not msg: return jsonify(error="message is required"), 400
    ctx = dict(h=float(b.get("hour", 21)), ev=bool(b.get("event")), su=bool(b.get("surge")),
               dn=bool(b.get("down")), zone=tuple(b.get("zone", [2, 2])))
    agent = route(msg); facts, text = run_agent(agent, msg, ctx)
    return jsonify(agent=agent, reply=phrase_with_llm(msg, agent, facts, text), data=facts)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
