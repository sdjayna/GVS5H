"""AtCoder-style special judges for the 7 problems in lcb100_hardest_v6 that accept more than one
correct output. LiveCodeBench's grader compares output text exactly, so on these problems it can
reject a correct answer. Each judge decides from the problem statement, not from the expected file.

The paper's appendix ("Two corrections to the LiveCodeBench evaluator") already judges abc396_e,
arc190_a, arc195_c and abc385_f by the contest's rule; those judges are re-implemented here so
that new runs can be scored the same way. abc397_d, abc398_f, arc191_c and arc192_b are additions.

Grader faults this corrects:
- abc396_e, abc398_f, abc397_d, arc191_c, arc190_a: "any of them is accepted" (or equivalent).
- arc192_b: the judge is case-insensitive.
- arc195_c: a construction problem ("show one example"): any valid placement is correct.
- abc397_d test 41 (N = 27) expects "3 0", but the problem requires positive integers and none
  exist, so the correct output is -1.

Run from codebase/v2-current:
  LCB_RELEASE=release_v6 python escalation/judge_special.py --selftest
  LCB_RELEASE=release_v6 python escalation/judge_special.py --single runs/single.jsonl --harness runs/multi.jsonl
--selftest feeds every official expected answer through its own judge; only abc397_d test 41 is
expected to be rejected, because its official answer is invalid.
"""
import argparse, json, os, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RUNS = os.path.join(ROOT, "runs")
sys.path.insert(0, os.path.join(ROOT, "codebase", "livecodebench"))


def judge_abc396_e(inp, exp, got):
    e, g = exp.split(), got.split()
    if e == ["-1"]:
        return g == ["-1"], "expected -1"
    t = inp.split()
    n, m = int(t[0]), int(t[1])
    edges = [(int(t[2 + 3 * i]), int(t[3 + 3 * i]), int(t[4 + 3 * i])) for i in range(m)]
    if len(g) != n:
        return False, f"printed {len(g)} values, need {n}"
    try:
        a = [int(x) for x in g]
    except ValueError:
        return False, "non-integer output"
    if any(v < 0 for v in a):
        return False, "negative value"
    for x, y, z in edges:
        if a[x - 1] ^ a[y - 1] != z:
            return False, f"A{x} xor A{y} != {z}"
    if sum(a) != sum(int(v) for v in e):
        return False, f"sum {sum(a)} is not the minimum {sum(int(v) for v in e)}"
    return True, "ok"


def cube_difference_pair_exists(n):
    # x^3 - y^3 = d (x^2 + xy + y^2) with d = x - y >= 1, so d^3 < n and d <= 10^6 for n <= 10^18.
    from math import isqrt
    d = 1
    while d * d * d < n:
        if n % d == 0:
            disc = 12 * (n // d) - 3 * d * d
            if disc >= 0:
                r = isqrt(disc)
                if r * r == disc and (r - 3 * d) > 0 and (r - 3 * d) % 6 == 0:
                    return True
        d += 1
    return False


def judge_abc397_d(inp, exp, got):
    # Judged from the statement, not the expected file: LiveCodeBench's test 41 (N = 27)
    # expects "3 0", but y = 0 is not a positive integer and no positive pair exists.
    n, g = int(inp.split()[0]), got.split()
    if g == ["-1"]:
        return (not cube_difference_pair_exists(n)), "a positive pair exists"
    if len(g) != 2:
        return False, f"printed {len(g)} values, need 2"
    try:
        x, y = int(g[0]), int(g[1])
    except ValueError:
        return False, "non-integer output"
    if x <= 0 or y <= 0:
        return False, "non-positive value"
    return (x ** 3 - y ** 3 == n), "x^3 - y^3 != N"


def judge_abc398_f(inp, exp, got):
    s, e, g = inp.split()[0], exp.split(), got.split()
    if len(g) != 1:
        return False, f"printed {len(g)} tokens, need 1"
    t = g[0]
    if not t.startswith(s):
        return False, "does not start with S"
    if t != t[::-1]:
        return False, "not a palindrome"
    return (len(t) == len(e[0])), f"length {len(t)} is not the shortest {len(e[0])}"


def judge_arc191_c(inp, exp, got):
    cases = [int(x) for x in inp.split()[1:]]
    g = got.split()
    if len(g) != 2 * len(cases):
        return False, f"printed {len(g)} values, need {2 * len(cases)}"
    for i, n in enumerate(cases):
        try:
            a, m = int(g[2 * i]), int(g[2 * i + 1])
        except ValueError:
            return False, "non-integer output"
        if not (1 <= a <= 10 ** 18 and 1 <= m <= 10 ** 18):
            return False, f"case {i}: A or M out of range"
        one = 1 % m
        if pow(a, n, m) != one:
            return False, f"case {i}: A^N - 1 is not a multiple of M"
        primes, k, p = set(), n, 2
        while p * p <= k:
            while k % p == 0:
                primes.add(p)
                k //= p
            p += 1
        if k > 1:
            primes.add(k)
        for q in primes:
            if pow(a, n // q, m) == one:
                return False, f"case {i}: a smaller n ({n // q}) works"
    return True, "ok"


def judge_arc190_a(inp, exp, got):
    e, g = exp.split(), got.split()
    if e == ["-1"]:
        return g == ["-1"], "expected -1"
    t = inp.split()
    n, m = int(t[0]), int(t[1])
    ranges = [(int(t[2 + 2 * i]), int(t[3 + 2 * i])) for i in range(m)]
    if len(g) != m + 1:
        return False, f"printed {len(g)} values, need {m + 1}"
    try:
        k, ops = int(g[0]), [int(x) for x in g[1:]]
    except ValueError:
        return False, "non-integer output"
    if k != int(e[0]):
        return False, f"cost {k} is not the minimum {e[0]}"
    if any(o not in (0, 1, 2) for o in ops) or sum(1 for o in ops if o) != k:
        return False, "operations do not match the stated cost"
    diff = [0] * (n + 2)
    for (l, r), o in zip(ranges, ops):
        if o == 1:
            diff[l] += 1
            diff[r + 1] -= 1
        elif o == 2:
            diff[1] += 1
            diff[l] -= 1
            diff[r + 1] += 1
            diff[n + 1] -= 1
    run_sum = 0
    for j in range(1, n + 1):
        run_sum += diff[j]
        if run_sum <= 0:
            return False, f"x_{j} is still 0"
    return True, "ok"


def judge_arc195_c(inp, exp, got):
    t = inp.split()
    cases = [(int(t[1 + 2 * i]), int(t[2 + 2 * i])) for i in range(int(t[0]))]
    e, g = exp.split(), got.split()
    pe = pg = 0
    for i, (red, blue) in enumerate(cases):
        n = red + blue
        want = e[pe]
        pe += 1 + (3 * n if want == "Yes" else 0)
        if pg >= len(g):
            return False, f"case {i}: output ends early"
        have = g[pg]
        pg += 1
        if have != want:
            return False, f"case {i}: printed {have!r}, the answer is {want}"
        if want == "No":
            continue
        if pg + 3 * n > len(g):
            return False, f"case {i}: fewer than {n} pieces printed"
        pts = []
        for _ in range(n):
            p, r, c = g[pg], g[pg + 1], g[pg + 2]
            pg += 3
            if p not in ("R", "B"):
                return False, f"case {i}: piece colour {p!r}"
            try:
                r, c = int(r), int(c)
            except ValueError:
                return False, f"case {i}: non-integer square"
            if not (1 <= r <= 10 ** 9 and 1 <= c <= 10 ** 9):
                return False, f"case {i}: square off the board"
            pts.append((p, r, c))
        if sum(p == "R" for p, _, _ in pts) != red:
            return False, f"case {i}: wrong number of red pieces"
        if len({(r, c) for _, r, c in pts}) != n:
            return False, f"case {i}: two pieces on one square"
        for j, (p, r, c) in enumerate(pts):
            _, r2, c2 = pts[(j + 1) % n]
            dr, dc = abs(r - r2), abs(c - c2)
            if not ((p == "R" and dr + dc == 1) or (p == "B" and dr == 1 and dc == 1)):
                return False, f"case {i}: piece {j + 1} ({p}) cannot move to piece {(j + 1) % n + 1}"
    if pg != len(g):
        return False, "extra output"
    return True, "ok"


def judge_arc192_b(inp, exp, got):
    return got.strip().lower() == exp.strip().lower(), "wrong winner"


JUDGES = {"abc396_e": judge_abc396_e, "abc397_d": judge_abc397_d, "abc398_f": judge_abc398_f,
          "arc191_c": judge_arc191_c, "arc190_a": judge_arc190_a, "arc192_b": judge_arc192_b,
          "arc195_c": judge_arc195_c}


def run(code, tests, judge):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    passed, first = 0, None
    for i, (inp, exp) in enumerate(tests):
        try:
            out = subprocess.run([sys.executable, path], input=inp, capture_output=True, text=True, timeout=10).stdout
            ok, why = judge(inp, exp, out)
        except subprocess.TimeoutExpired:
            ok, why = False, "timeout (10 s)"
        passed += ok
        if not ok and first is None:
            first = (i, why)
    return passed, first


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true", help="judge the official expected answers")
    ap.add_argument("--single", default=os.path.join(RUNS, "ckpt-single.jsonl"), help="single-call checkpoint")
    ap.add_argument("--harness", default=os.path.join(RUNS, "ckpt-multi.jsonl"), help="harness checkpoint")
    ap.add_argument("--out", default=os.path.join(RUNS, "special-judge-verdicts.json"))
    args = ap.parse_args()
    from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
    by = {p.question_id: p for p in load_code_generation_dataset(release_version=os.environ.get("LCB_RELEASE", "release_v6"))}
    if args.selftest:
        for qid, judge in JUDGES.items():
            io = json.loads(by[qid].get_evaluation_sample()["input_output"])
            results = [judge(i, o, o) for i, o in zip(io["inputs"], io["outputs"])]
            bad = [(n, why) for n, (ok, why) in enumerate(results) if not ok]
            print(f"selftest {qid}: official answers accepted {len(results) - len(bad)}/{len(results)}" + (f" | first rejection: test {bad[0][0]}: {bad[0][1]}" if bad else ""))
        return
    verdicts = {}
    for qid, judge in JUDGES.items():
        io = json.loads(by[qid].get_evaluation_sample()["input_output"])
        tests = list(zip(io["inputs"], io["outputs"]))
        for label, path in (("single call", args.single), ("harness", args.harness)):
            recs = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
            rec = next((r for r in recs if r["question_id"] == qid), None)
            if rec is None:
                continue
            code = rec.get("code") or ""
            if not code.strip():
                print(f"{qid} | {label}: no code recorded")
                verdicts[f"{label}:{qid}"] = False
                continue
            passed, first = run(code, tests, judge)
            ok = passed == len(tests)
            verdicts[f"{label}:{qid}"] = ok
            print(f"{qid} | {label}: {passed}/{len(tests)} accepted" + ("" if ok else f" | first rejection: test {first[0]}: {first[1]}"))
    json.dump(verdicts, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
