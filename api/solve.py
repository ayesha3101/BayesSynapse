"""
BayeSynapse backend.  POST /api/solve  ->  exact inference on a discrete Bayesian network.
Every variable has 2 to 6 named states (stdlib only).

Network format:
  variables: ["Weather", "Umbrella"]
  states:    {"Weather": ["Sunny", "Rainy"], "Umbrella": ["Yes", "No"]}
  parents:   {"Weather": [], "Umbrella": ["Weather"]}
  cpt:       {"Weather": {"": [0.7, 0.3]},
              "Umbrella": {"Sunny": [0.1, 0.9], "Rainy": [0.9, 0.1]}}
A cpt key is the parents' states joined by "," in parent order ("" if no parents).
Its value lists P(variable = each state), in the order of that variable's states.
"""
import json
import re
from http.server import BaseHTTPRequestHandler
from itertools import product

MAX_VARS, MAX_STATES, MAX_CELLS = 10, 6, 200_000
MAX_BODY = 1_000_000                       # bytes; real requests are a few KB
NAME = re.compile(r"[A-Za-z0-9_]{1,12}")   # names go into table keys and the answer text, so keep them plain
fmt = lambda d: ", ".join(f"{k}={v}" for k, v in d.items())


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
    vs, st, ps, cpt = net["variables"], net["states"], net["parents"], net["cpt"]
    if not (isinstance(vs, list) and 1 <= len(vs) <= MAX_VARS and len(set(vs)) == len(vs)):
        raise ValueError("Need 1 to 10 different variables.")
    if not all(isinstance(v, str) and NAME.fullmatch(v) for v in vs):
        raise ValueError("Variable names must be 1 to 12 letters, numbers or _.")
    for v in vs:
        if not (isinstance(st[v], list) and 2 <= len(st[v]) <= MAX_STATES and len(set(st[v])) == len(st[v])):
            raise ValueError(f"{v} needs 2 to 6 different states.")
        if not all(isinstance(x, str) and NAME.fullmatch(x) for x in st[v]):
            raise ValueError(f"State names for {v} must be 1 to 12 letters, numbers or _.")
        if len(set(ps[v])) != len(ps[v]) or any(p not in vs or p == v for p in ps[v]):
            raise ValueError(f"Bad dependencies for {v}.")
    for v in vs:
        for combo in product(*[st[p] for p in ps[v]]):
            row = cpt[v][",".join(combo)]
            if (len(row) != len(st[v]) or abs(sum(row) - 1) > 1e-5
                    or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not 0 <= x <= 1 for x in row)):
                raise ValueError(f"The probabilities for {v} must be between 0 and 1 and add up to 1.")
    if not acyclic(vs, ps):
        raise ValueError("The network has a loop.")


# ---------- shared ----------
def prob(net, v, val, a):
    """P(v=val | its parents' states in a)."""
    key = ",".join(a[p] for p in net["parents"][v])
    return net["cpt"][v][key][net["states"][v].index(val)]


def label(net, v, a):
    ps = net["parents"][v]
    cond = " | " + ", ".join(f"{p}={a[p]}" for p in ps) if ps else ""
    return f"P({v}={a[v]}{cond})"


def cells(st, names):
    n = 1
    for v in names:
        n *= len(st[v])
    return n


# ---------- method 1: enumeration ----------
def enum(net, fixed, steps):
    vs, st = net["variables"], net["states"]
    hidden = [v for v in vs if v not in fixed]
    if cells(st, hidden) > MAX_CELLS:
        raise ValueError("Too many combinations for enumeration. Try variable elimination.")
    steps.append("  sum over " + (", ".join(hidden) or "nothing") + " of the joint probability")
    total, n = 0.0, 0
    for c in product(*[st[v] for v in hidden]):
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
# A factor is (variable list, {tuple of states: value}).
def restrict(f, fixed):
    sc, t = f
    keep = [v for v in sc if v not in fixed]
    out = {}
    for c, p in t.items():
        a = dict(zip(sc, c))
        if all(a[v] == fixed[v] for v in sc if v in fixed):
            out[tuple(a[v] for v in keep)] = p
    return keep, out


def mul(f, g, st):
    sc = f[0] + [v for v in g[0] if v not in f[0]]
    if cells(st, sc) > MAX_CELLS:
        raise ValueError("This network is too large to work out here.")
    out = {}
    for c in product(*[st[v] for v in sc]):
        a = dict(zip(sc, c))
        out[c] = f[1][tuple(a[v] for v in f[0])] * g[1][tuple(a[v] for v in g[0])]
    return sc, out


def sum_out(f, var):
    sc, t = f
    keep = [v for v in sc if v != var]
    out = {}
    for c, p in t.items():
        k = tuple(dict(zip(sc, c))[v] for v in keep)
        out[k] = out.get(k, 0.0) + p
    return keep, out


def ve(net, fixed, steps):
    vs, st, ps = net["variables"], net["states"], net["parents"]
    name = lambda f: "f(" + ",".join(f[0]) + ")"
    fs = []
    for v in vs:  # one factor per CPT
        sc = [v] + ps[v]
        if cells(st, sc) > MAX_CELLS:
            raise ValueError("This network is too large to work out here.")
        fs.append((sc, {c: prob(net, v, c[0], dict(zip(sc, c))) for c in product(*[st[x] for x in sc])}))
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
            g = mul(g, f, st)
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
    st = net["states"]
    if not query:
        raise ValueError("Pick a variable to ask about.")
    for v, x in {**query, **evidence}.items():
        if v not in net["variables"] or x not in st[v]:
            raise ValueError(f"Unknown variable or state: {v}={x}")
    if set(query) & set(evidence):
        raise ValueError("A variable can't be both the question and something we know.")
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
        raise ValueError("What we know has probability 0, so the answer is undefined.")
    q, e = fmt(query), fmt(evidence)
    return {"formula": f"P({q} | {e}) = P({q}, {e}) / P({e})", "steps": steps, "value": num / den,
            "answer": f"P({q} | {e}) = {num:.6g} / {den:.6g} = {num / den:.6g}"}


class TooLarge(ValueError):
    pass


class handler(BaseHTTPRequestHandler):
    def read_body(self):
        """Read the request body, but never trust the declared size."""
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise ValueError("Bad Content-Length header.") from None
        if n < 0:
            raise ValueError("Bad Content-Length header.")
        if n > MAX_BODY:
            raise TooLarge("Request is too large.")
        return self.rfile.read(n)

    def do_POST(self):
        try:
            body = json.loads(self.read_body())
            method = "elimination" if body.get("method") == "elimination" else "enumeration"
            out, code = solve(body["network"], body["query"], body.get("evidence", {}), method), 200
        except TooLarge as e:
            out, code = {"error": str(e)}, 413
        except ValueError as e:
            out, code = {"error": str(e)}, 400
        except (KeyError, TypeError, AttributeError, IndexError):
            out, code = {"error": "Invalid or incomplete network."}, 400
        data = json.dumps(out).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)