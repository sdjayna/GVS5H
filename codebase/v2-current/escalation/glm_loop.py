"""A fixed sequence of steps for long-thinking models ("loop" engine in run_bench).

The harness decides what happens next from measured state; the model only does the work, one artefact
per call, so nothing has to be parsed out of a mixed reply:

  derive     thinking on       the method, in prose (if cut off: a thinking-off summary of the reasoning)
  implement  thinking off      the program; the reply is prefilled with a python fence and stopped at the
                               closing fence, so the reply is the code
  brute      thinking off      a slow, obviously correct solver for small inputs (same format)
  gen        thinking off      a random valid input: `gen.py <seed> small` or `gen.py <seed> max`
  classify   thinking off      does a test input have exactly one correct output? one word
  checker    thinking off      only when it does not: judges an output against the input

The harness tests every program: public samples, the brute force (or the checker) on generated small
inputs, and the time on one maximum-size input. A failure goes back to implement with the exact failing
case; after MAX_FIXES failed fixes in a row the method is re-derived with the history of what failed.
A program is accepted only when every check it can run passes. Works for stdin and for functional
(LeetCode-style) problems, which are run through a small wrapper.

Settings (environment, all optional):
  LOOP_BUDGETS     per-step output tokens, e.g. "derive=160000,implement=16000". Defaults below, each
                   with the rule that set it.
  LOOP_MAX_ROUNDS  implement rounds per problem (default 8)
  LOOP_WS          workspace root (default runs/ws-loop)
Model and endpoint: MULTIAGENT_MODEL ("openai:<name>"), ESCALATION_OPENAI_BASE, OPENAI_API_KEY,
ESCALATION_CLOUD_TIMEOUT (seconds per call).
"""
import ast, hashlib, json, os, re, subprocess, sys, tempfile, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
URL = os.environ.get("ESCALATION_OPENAI_BASE", "http://127.0.0.1:8000/v1/chat/completions")
KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = os.environ.get("MULTIAGENT_MODEL", "openai:model").split(":", 1)[-1]
TIMEOUT = int(os.environ.get("ESCALATION_CLOUD_TIMEOUT", "21600"))
WS_ROOT = os.environ.get("LOOP_WS", os.path.join(ROOT, "runs", "ws-loop"))
MAX_ROUNDS = int(os.environ.get("LOOP_MAX_ROUNDS", "8"))
MAX_FIXES = 2            # failed fixes in a row before the method is re-derived
MAX_ADJUDICATIONS = 2    # disputes with the brute force settled by a second, independent brute force
STRESS_CASES = 40        # generated small inputs per test
TIME_LIMIT = 6.0         # seconds; LiveCodeBench's per-test limit
RUN_TIMEOUT = 10.0       # seconds for one run of a brute force or a small case

# Budgets, each set by a stated rule from logged calls (27 Oct 2026), to be re-derived from the tuning set:
BUDGETS = {
    "derive": 160000,    # covers the longest single-call answer that finished (149,764 tokens)
    "summary": 4000,     # summaries of cut-off reasoning used 529-2,000 characters of output
    "implement": 16000,  # successful code-only replies used 1,789-13,987 tokens
    "brute": 8000,       # half the implement budget: a brute force is shorter than the solution
    "gen": 4000,
    "checker": 8000,
    "classify": 16,
    "hand": 32000,       # a tiny input worked by hand; a fifth of the derive budget
}
BUDGETS.update({k: int(v) for k, v in (p.split("=", 1) for p in os.environ.get("LOOP_BUDGETS", "").split(",") if "=" in p)})

FENCE_OPEN = "```python\n"
FENCE_STOP = "\n```"


# ---------------------------------------------------------------- model calls

def _post(body):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read())


def _chat(ws, step, messages, budget, thinking, prefill=None, stop=None):
    """One call, recorded in the workspace transcript. No automatic retry: a timed-out request is not
    re-sent, so an abandoned generation can never be duplicated on the server."""
    sent = messages + ([{"role": "assistant", "content": prefill}] if prefill else [])
    body = {"model": MODEL, "messages": sent, "max_tokens": budget, "temperature": 0.2, "stream": False}
    if not thinking:
        body["reasoning_effort"] = "none"
    if stop:
        body["stop"] = stop
    t0 = time.time()
    try:
        out = _post(body)
        ch = out["choices"][0]
        msg = ch.get("message") or {}
        text, finish = msg.get("content") or "", ch.get("finish_reason")
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        tokens = (out.get("usage") or {}).get("completion_tokens")
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as e:
        text, finish, reasoning, tokens = "", "error", "", None
        print(f"    [loop] {step}: request failed: {e}", file=sys.stderr, flush=True)
    rec = {"t": time.time(), "seconds": round(time.time() - t0, 1), "step": step, "request": sent,
           "response": text, "reasoning": reasoning, "finish_reason": finish, "completion_tokens": tokens,
           "budget": budget, "thinking": thinking}
    with open(os.path.join(ws, "transcript.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    return text, reasoning, finish, tokens


def clean_code(text):
    """The program from a prefilled, stopped reply: drop a repeated opening fence and any closing fence;
    empty unless it parses and contains at least one statement."""
    code = re.sub(r"\A\s*```[ \t]*(?:python|py)?[ \t]*\n", "", text or "", count=1)
    code = code.split("\n```", 1)[0].rstrip() + "\n"
    try:
        return code if ast.parse(code).body else ""
    except SyntaxError:
        return ""


def _code_call(ws, step, system, user, budget):
    text, _, finish, _ = _chat(ws, step, [{"role": "system", "content": system}, {"role": "user", "content": user}],
                               budget, thinking=False, prefill=FENCE_OPEN, stop=[FENCE_STOP])
    return clean_code(text), finish


# ---------------------------------------------------------------- running programs

def _fn_name(starter):
    m = re.search(r"def\s+(\w+)\s*\(\s*self", starter or "")
    return m.group(1) if m else None


def _wrap(code, fn):
    """For functional problems: read one JSON value per line as the arguments, print the JSON result."""
    if not fn:
        return code
    return ("from typing import *\nimport collections, heapq, bisect, itertools, functools, math, string\n"
            f"{code}\n\nimport sys as _s, json as _j\n"
            "_a = [_j.loads(_l) for _l in _s.stdin.read().split('\\n') if _l.strip()]\n"
            f"print(_j.dumps(Solution().{fn}(*_a)))\n")


def run(code, stdin, timeout, fn=None, args=()):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(_wrap(code, fn))
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, f.name, *args], input=stdin, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, timeout, "timeout"
    finally:
        os.unlink(f.name)
    if r.returncode != 0:
        return None, time.time() - t0, (r.stderr or "runtime error")[-300:]
    return r.stdout, time.time() - t0, None


def same_output(got, want, fn):
    if fn:
        try:
            return json.loads(got) == json.loads(want)
        except (ValueError, TypeError):
            return False
    return (got or "").split() == (want or "").split()


# ---------------------------------------------------------------- the checks

def test(state, code):
    """Samples, then the brute force or checker on generated small inputs, then the time on a maximum input.
    Returns (ok, failure description or None, details)."""
    fn, d = state["fn"], {"samples": None, "stress": None, "time": None}
    passed = 0
    for t in state["tests"]:
        out, _, err = run(code, t.input, RUN_TIMEOUT, fn)
        good = out is not None and (same_output(out, t.output, fn) if state["unique"] else _checker_ok(state, t.input, out))
        if not good:
            d["samples"] = f"{passed}/{len(state['tests'])}"
            got = f"<{err}>" if out is None else out.strip()[:300]
            return False, (f"fails public sample {passed + 1}: input={t.input[:400]!r} expected={t.output.strip()[:300]!r} "
                           f"got={got!r}"), d
        passed += 1
    d["samples"] = f"{passed}/{len(state['tests'])}"
    if state.get("gen"):
        checked = 0
        for seed in range(1, STRESS_CASES + 1):
            inp, _, _ = run(state["gen"], "", RUN_TIMEOUT, None, (str(seed), "small"))
            if not inp:
                continue
            out, _, err = run(code, inp, RUN_TIMEOUT, fn)
            if out is None:
                d["stress"] = f"crash after {checked}"
                return False, f"crashes on a generated input: input={inp[:400]!r} error={err[:200]!r}", d
            if state["unique"]:
                if not state.get("brute"):
                    continue
                want, _, _ = run(state["brute"], inp, RUN_TIMEOUT, fn)
                if want is None:
                    continue
                if not same_output(out, want, fn):
                    d["stress"] = f"mismatch after {checked}"
                    d["dispute"] = {"input": inp, "brute": want, "program": out}
                    return False, (f"disagrees with the brute force on a generated input: input={inp[:400]!r} "
                                   f"brute={want.strip()[:300]!r} program={out.strip()[:300]!r}"), d
            elif not _checker_ok(state, inp, out):
                d["stress"] = f"checker rejects after {checked}"
                return False, f"the checker rejects its output on a generated input: input={inp[:400]!r} output={out.strip()[:300]!r}", d
            checked += 1
        d["stress"] = f"{checked} agree"
        big, _, _ = run(state["gen"], "", 30, None, ("0", "max"))
        if big:
            out, secs, err = run(code, big, TIME_LIMIT * 2, fn)
            d["time"] = round(secs, 2)
            if err == "timeout" or secs > TIME_LIMIT:
                return False, (f"too slow: a maximum-size input ({len(big)} characters) took over {TIME_LIMIT:g} s; "
                               f"make it faster without changing its answers"), d
            if err:
                return False, f"crashes on a maximum-size input: {err[:200]!r}", d
    return True, None, d


def _checker_ok(state, inp, out):
    if not state.get("checker"):
        return True   # no checker could be built: only crashes count on this problem
    with tempfile.NamedTemporaryFile("w", suffix=".in", delete=False) as a, tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as b:
        a.write(inp)
        b.write(out or "")
    verdict, _, _ = run(state["checker"], "", RUN_TIMEOUT, None, (a.name, b.name))
    os.unlink(a.name); os.unlink(b.name)
    return bool(verdict) and verdict.strip().upper().startswith("OK")


# ---------------------------------------------------------------- the steps

CODE_SYS = ("You are an expert competitive programmer. Reply with the complete Python program only, "
            "with no explanation before or after it.")


def _io_note(fn):
    return (f"Write class Solution with the method {fn}, as in the starter code." if fn else
            "The program reads standard input and writes standard output, exactly as the statement specifies.")


def derive(ws, problem, history):
    past = "\n".join(f"- Attempt {i + 1}: {h}" for i, h in enumerate(history[-6:])) or "(none)"
    user = (f"{problem}\n\nEarlier attempts and how they failed:\n{past}\n\nWork out a correct algorithm that is "
            "fast enough for the limits. Do not write code. End with a section headed METHOD that states the "
            "algorithm, data structures, complexity and edge cases precisely enough to implement directly.")
    text, reasoning, finish, _ = _chat(ws, "derive", [{"role": "user", "content": user}], BUDGETS["derive"], thinking=True)
    if finish == "stop" and text.strip():
        m = re.search(r"METHOD\W*\n?(.*)", text, re.S | re.I)
        return (m.group(1) if m else text).strip()[-6000:], False
    tail = (reasoning or text)[-60000:]
    summary, _, _, _ = _chat(ws, "summary", [{"role": "user", "content": (
        f"{problem}\n\nThis reasoning about the problem was cut off:\n{tail}\n\nState the best method it reached, "
        "precisely enough to implement: algorithm, data structures, complexity, open doubts. No code.")}],
        BUDGETS["summary"], thinking=False)
    return summary.strip()[-6000:], True


def implement(ws, problem, method, prev, failure, fn):
    user = f"{problem}\n\n{_io_note(fn)}\n\nMETHOD:\n{method}\n"
    if prev and failure:
        user += f"\nThe current program:\n```python\n{prev}```\nIt {failure}\nFix it."
    else:
        user += "\nImplement this method."
    return _code_call(ws, "implement", CODE_SYS, user, BUDGETS["implement"])


def adjudicate(ws, state, problem, dispute, log):
    """The program and the brute force disagree: one of them is wrong. A second brute force, written
    independently by a different method, votes on the disputed input. Returns "program", "brute" or None
    (the side judged wrong). A wrong brute force is replaced; three-way disagreement drops the comparison."""
    fn = state["fn"]
    second, _ = _code_call(ws, "brute2", CODE_SYS, (
        f"{problem}\n\n{_io_note(fn)}\n\nWrite an obviously correct solution for small inputs only, using a "
        "different method from the most obvious one (for example direct simulation instead of search, or "
        "exhaustive enumeration of every candidate answer). Speed does not matter; correctness does."), BUDGETS["brute"])
    valid = second and all(same_output(run(second, t.input, RUN_TIMEOUT, fn)[0], t.output, fn) for t in state["tests"])
    if not valid:
        log("    [loop] dispute: no valid second brute force; the program is held to the first")
        return "program"
    vote, _, _ = run(second, dispute["input"], RUN_TIMEOUT, fn)
    if vote is not None and same_output(vote, dispute["program"], fn):
        state["brute"] = second
        with open(os.path.join(ws, "brute.py"), "w") as f:
            f.write(second)
        log("    [loop] dispute: the second brute force agrees with the program; brute force replaced")
        return "brute"
    if vote is not None and same_output(vote, dispute["brute"], fn):
        # Two programs from the same model can share one misreading of the statement, so a third program
        # is not an independent witness. Work the tiny input by hand instead: a different mode, not a
        # different program.
        hand = hand_answer(ws, problem, dispute["input"], fn)
        if hand is not None and same_output(hand, dispute["program"], fn):
            state["brute"] = None
            log("    [loop] dispute: both brute forces agree, but the hand-worked answer agrees with the program; "
                "the brute-force comparison is dropped for this problem")
            return "brute"
        if hand is not None and same_output(hand, dispute["brute"], fn):
            log("    [loop] dispute: two brute forces and the hand-worked answer agree; the program is wrong")
            return "program"
        log("    [loop] dispute: no hand-worked answer agrees with the program; the two brute forces stand")
        return "program"
    state["brute"] = None
    log("    [loop] dispute: three different answers; the brute-force comparison is dropped for this problem")
    return None


def hand_answer(ws, problem, inp, fn):
    """The correct output for one tiny input, worked step by step with thinking on."""
    fmt = "the JSON value the method returns" if fn else "exactly what the program must print"
    text, _, finish, _ = _chat(ws, "hand", [{"role": "user", "content": (
        f"{problem}\n\nWork out by hand, step by step, the correct output for this input:\n{inp}\n"
        f"Check every condition in the statement. End with a line ANSWER: followed by {fmt}.")}],
        BUDGETS["hand"], thinking=True)
    m = list(re.finditer(r"ANSWER:\s*(.+)", text or ""))
    return m[-1].group(1).strip() if (finish == "stop" and m) else None


def build_verifiers(ws, state, problem):
    fn = state["fn"]
    verdict, _, _, _ = _chat(ws, "classify", [{"role": "user", "content": (
        f"{problem}\n\nFor a single test input of this problem, is there exactly one correct output? Answer with "
        "one word: UNIQUE if every correct program must print the same output, MULTIPLE if several outputs are "
        "accepted (any valid answer, a construction, or a tolerance).")}], BUDGETS["classify"], thinking=False)
    state["unique"] = not verdict.strip().upper().startswith("MULTIPLE")
    gen, _ = _code_call(ws, "gen", CODE_SYS, (
        f"{problem}\n\nWrite a generator of valid inputs for this problem. It is run as `python gen.py SEED MODE`: "
        "seed the random generator with int(sys.argv[1]); MODE is `small` (tiny inputs a brute force can solve "
        "in well under a second) or `max` (one input at the maximum constraints). Print exactly one input in "
        + ("the format of the test inputs: one JSON value per line, one line per argument of the method." if fn
           else "the exact input format.")), BUDGETS["gen"])
    state["gen"] = gen or None
    if state["unique"]:
        brute, _ = _code_call(ws, "brute", CODE_SYS, (
            f"{problem}\n\n{_io_note(fn)}\n\nWrite the simplest obviously correct solution, for small inputs only: "
            "exhaustive search or direct simulation. Speed does not matter; correctness does."), BUDGETS["brute"])
        ok = brute and all(same_output(run(brute, t.input, RUN_TIMEOUT, fn)[0], t.output, fn) for t in state["tests"])
        state["brute"] = brute if ok else None
    else:
        checker, _ = _code_call(ws, "checker", CODE_SYS, (
            f"{problem}\n\nWrite a checker. It is run as `python checker.py INPUT_FILE OUTPUT_FILE`, reads a test "
            "input and a program's output, and prints OK if the output is a correct answer for that input, "
            "otherwise WRONG and a reason. Check every condition in the statement."), BUDGETS["checker"])
        ok = checker and all(_checker_ok({"checker": checker}, t.input, t.output) for t in state["tests"])
        state["checker"] = checker if ok else None
    for name in ("gen", "brute", "checker"):
        if state.get(name):
            with open(os.path.join(ws, name + ".py"), "w") as f:
                f.write(state[name])


# ---------------------------------------------------------------- the loop

def loop_solve(problem_text, spec, log=None, status_out=None, tests=None):
    log = log or (lambda *a, **k: None)
    ws = os.path.join(WS_ROOT, hashlib.sha1(problem_text.encode()).hexdigest()[:12])
    os.makedirs(ws, exist_ok=True)
    open(os.path.join(ws, "transcript.jsonl"), "w").close()
    starter = re.search(r"```python\n(.*?)```", problem_text, re.S)
    fn = _fn_name(starter.group(1)) if starter else None
    state = {"fn": fn, "tests": [t for t in (tests or []) if str(getattr(t, "testtype", "")).lower().endswith(("stdin", "functional"))],
             "unique": True, "adjudications": 0}
    build_verifiers(ws, state, problem_text)
    log(f"    [loop] verifiers: unique={state['unique']} gen={bool(state.get('gen'))} "
        f"brute={bool(state.get('brute'))} checker={bool(state.get('checker'))}")

    history, method, code, failure, fixes, best = [], None, None, None, MAX_FIXES, None
    for rnd in range(1, MAX_ROUNDS + 1):
        if method is None or fixes >= MAX_FIXES:
            method, cut = derive(ws, problem_text, history)
            log(f"    [loop] round {rnd}: derived a method ({len(method)} chars{', from a cut-off summary' if cut else ''})")
            code, failure, fixes = None, None, 0
        new, finish = implement(ws, problem_text, method, code, failure, fn)
        if not new:
            failure, fixes = "wrote no program", fixes + 1
            history.append(f"method {method[:300]!r}: no program written ({finish})")
            log(f"    [loop] round {rnd}: no program ({finish})")
            continue
        code = new
        ok, failure, d = test(state, code)
        while not ok and d.get("dispute") and state["adjudications"] < MAX_ADJUDICATIONS:
            state["adjudications"] += 1
            if adjudicate(ws, state, problem_text, d["dispute"], log) == "program":
                break
            ok, failure, d = test(state, code)   # re-test against the corrected or dropped comparison
        log(f"    [samples] {d['samples']} | [stress] {d['stress']} | time {d['time']}")
        if ok:
            best = code
            log(f"    [loop] round {rnd}: accepted")
            break
        if failure.startswith("too slow") and best is None:
            best = code   # correct on everything it could check: the best fallback so far
        fixes += 1
        history.append(f"method {method[:300]!r}: program {failure[:300]}")
        log(f"    [loop] round {rnd}: {failure[:120]}")

    final = best or code or ""
    recs = [json.loads(l) for l in open(os.path.join(ws, "transcript.jsonl")) if l.strip()]
    if status_out is not None:
        status_out.update({"ws": ws, "n_calls": len(recs),
                           "truncated_calls": sum(r.get("finish_reason") == "length" for r in recs),
                           "completion_tokens": sum(r.get("completion_tokens") or 0 for r in recs),
                           "finish_reason": "stop" if best else "unsolved"})
    with open(os.path.join(ws, "solution.py"), "w") as f:
        f.write(final)
    return f"```python\n{final}```" if final else ""
