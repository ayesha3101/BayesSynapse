"""BayeSynapse backend. POST /api/solve  ->  exact inference on a Boolean Bayesian network (stdlib only)."""
import json
from http.server import BaseHTTPRequestHandler
from itertools import product

MAX = 10
T = lambda b: "T" if b else "F"
fmt = lambda d: ", ".join(f"{k}={T(v)}" for k, v in d.items())


# ---------- validation ----------
def acyclic(vs, ps):
    done, left = set(), list(vs)
    while left:
        ready = [v for v in left if all(p in done for p in ps[v])]
        if not ready:
            return False
        done.update(ready)
        left = [v for v in left if v not in done]
    return True


def check(net):
    vs, ps, cpt = net["variables"], net["parents"], net["cpt"]
    if not (isinstance(vs, list) and 1 <= len(vs) <= MAX and len(set(vs)) == len(vs)):
        raise ValueError("Need 1 to 10 unique variables.")
    for v in vs:
        if len(set(ps[v])) != len(ps[v]) or any(p not in vs or p == v for p in ps[v]):
            raise ValueError(f"Bad parents for {v}.")
        for c in product("TF", repeat=len(ps[v])):
            p = cpt[v]["".join(c)]
            if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 <= p <= 1:
                raise ValueError(f"Probabilities for {v} must be between 0 and 1.")
    if not acyclic(vs, ps):
        raise ValueError("The network has a cycle.")


# ---------- shared ----------
def prob(net, v, val, a):
    """P(v=val | parents) where a holds the parents' values."""
    key = "".join(T(a[p]) for p in net["parents"][v])
    p = net["cpt"][v][key]
    return p if val else 1 - p


def label(net, v, a):
    ps = net["parents"][v]
    cond = " | " + ", ".join(f"{p}={T(a[p])}" for p in ps) if ps else ""
    return f"P({v}={T(a[v])}{cond})"


# ---------- method 1: enumeration ----------
def enum(net, fixed, steps):
    vs = net["variables"]
    hidden = [v for v in vs if v not in fixed]
    steps.append("  sum over " + (", ".join(hidden) or "nothing") + " of the joint probability")
    total, n = 0.0, 0
    for c in product([True, False], repeat=len(hidden)):
        a = {**fixed, **dict(zip(hidden, c))}
        j, parts = 1.0, []
        for v in vs:
            p = prob(net, v, a[v], a)
            j *= p
            parts.append(f"{label(net, v, a)}={p:.4g}")
        total += j
        n += 1
        if n <= 8:
            steps.append("  " + " × ".join(parts) + f" = {j:.4g}")
    if n > 8:
        steps.append(f"  ... and {n - 8} more terms")
    steps.append(f"  total = {total:.6g}")
    return total


# ---------- method 2: variable elimination ----------
# A factor is (variable list, {tuple of True/False: value}).
def restrict(f, fixed):
    sc, t = f
    keep = [v for v in sc if v not in fixed]
    out = {}
    for c, p in t.items():
        a = dict(zip(sc, c))
        if all(a[v] == fixed[v] for v in sc if v in fixed):
            out[tuple(a[v] for v in keep)] = p
    return keep, out


def mul(f, g):
    sc = f[0] + [v for v in g[0] if v not in f[0]]
    out = {}
    for c in product([True, False], repeat=len(sc)):
        a = dict(zip(sc, c))
        out[c] = f[1][tuple(a[v] for v in f[0])] * g[1][tuple(a[v] for v in g[0])]
    return sc, out


def sum_out(f, var):
    sc, t = f
    keep = [v for v in sc if v != var]
    out = {}
    for c, p in t.items():
        a = dict(zip(sc, c))
        k = tuple(a[v] for v in keep)
        out[k] = out.get(k, 0.0) + p
    return keep, out


def ve(net, fixed, steps):
    vs, ps = net["variables"], net["parents"]
    name = lambda f: "f(" + ",".join(f[0]) + ")"
    fs = []
    for v in vs:  # one factor per CPT
        sc = [v] + ps[v]
        fs.append((sc, {c: prob(net, v, c[0], dict(zip(sc, c)))
                        for c in product([True, False], repeat=len(sc))}))
    steps.append("  1. one factor per CPT: " + ", ".join(map(name, fs)))
    fs = [restrict(f, fixed) for f in fs]
    steps.append(f"  2. restrict to {fmt(fixed)}: " + ", ".join(map(name, fs)))
    hidden = [v for v in vs if v not in fixed]
    for _ in range(len(hidden)):  # smallest resulting factor first
        v = min(hidden, key=lambda x: len({y for f in fs if x in f[0] for y in f[0]}))
        hidden.remove(v)
        inv = [f for f in fs if v in f[0]]
        rest = [f for f in fs if v not in f[0]]
        g = inv[0]
        for f in inv[1:]:
            g = mul(g, f)
        g = sum_out(g, v)
        steps.append(f"  3. eliminate {v}: " + " × ".join(map(name, inv)) + f" -> {name(g)}")
        fs = rest + [g]
    r = 1.0
    for f in fs:
        r *= f[1][()]
    steps.append(f"  4. multiply what is left = {r:.6g}")
    return r


# ---------- solve ----------
def solve(net, query, evidence, method):
    check(net)
    names = net["variables"]
    if not query or any(v not in names for v in list(query) + list(evidence)):
        raise ValueError("Pick a query variable from the network.")
    if set(query) & set(evidence):
        raise ValueError("A variable can't be both query and evidence.")
    steps = []

    def P(fixed, title):
        steps.append(f"{title} = P({fmt(fixed)})  [{method}]")
        return (ve if method == "elimination" else enum)(net, fixed, steps)

    num = P({**query, **evidence}, "Numerator" if evidence else "Result")
    if not evidence:
        return {"formula": f"P({fmt(query)})", "steps": steps, "value": num,
                "answer": f"P({fmt(query)}) = {num:.6g}"}
    den = P(evidence, "Denominator")
    if den == 0:
        raise ValueError("The evidence has probability 0, so the conditional is undefined.")
    q, e = fmt(query), fmt(evidence)
    return {"formula": f"P({q} | {e}) = P({q}, {e}) / P({e})", "steps": steps, "value": num / den,
            "answer": f"P({q} | {e}) = {num:.6g} / {den:.6g} = {num / den:.6g}"}


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            method = "elimination" if body.get("method") == "elimination" else "enumeration"
            out, code = solve(body["network"], body["query"], body.get("evidence", {}), method), 200
        except ValueError as e:
            out, code = {"error": str(e)}, 400
        except (KeyError, TypeError, AttributeError):
            out, code = {"error": "Invalid or incomplete network."}, 400
        data = json.dumps(out).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
