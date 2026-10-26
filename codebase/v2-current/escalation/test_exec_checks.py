"""Tests for version 3 verification (MULTIAGENT_EXEC_CHECKS=1). No dataset needed.

  MULTIAGENT_HARNESS_FIXES=1 MULTIAGENT_EXEC_CHECKS=1 python escalation/test_exec_checks.py
"""
import os, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MULTIAGENT_MODEL", "openai:test")


def main():
    import multiagent as ma
    assert ma.FIXES and ma.EXEC, "run with MULTIAGENT_HARNESS_FIXES=1 MULTIAGENT_EXEC_CHECKS=1"
    fails = 0

    def check(name, cond):
        nonlocal fails
        print(("PASS " if cond else "FAIL ") + name)
        fails += 0 if cond else 1

    gen = "import random, sys\nrandom.seed(int(sys.argv[1]))\nprint(random.randint(1, 30))\n"
    brute = "n = int(input())\nprint(sum(range(n + 1)))\n"
    good = "n = int(input())\nprint(n * (n + 1) // 2)\n"
    wrong = "n = int(input())\nprint(n * (n + 1) // 2 + (1 if n > 20 else 0))\n"
    crash = "n = int(input())\nraise SystemExit(3)\n"

    with tempfile.TemporaryDirectory() as ws:
        ma._write(ws, "gen.py", gen)
        ma._write(ws, "brute.py", brute)

        ma._write(ws, "solution.py", good)
        st = ma._stress_check(ws, multi=False)
        check("agreeing solution: no mismatch, all cases checked", st["ran"] and not st.get("mismatch") and st["checked"] == ma.STRESS_CASES)
        check("agreeing solution: feedback says they agree", "agree on all" in ma._stress_feedback(st))

        ma._write(ws, "solution.py", wrong)
        st = ma._stress_check(ws, multi=False)
        m = st.get("mismatch") or {}
        n = int(m.get("input", "0").strip() or 0)
        check("wrong solution: counterexample found", bool(m) and n > 20)
        check("wrong solution: counterexample carries both outputs", m.get("brute") == str(n * (n + 1) // 2) and m.get("solution") == str(n * (n + 1) // 2 + 1))
        check("wrong solution: feedback says DISAGREE", "DISAGREE" in ma._stress_feedback(st))

        ma._write(ws, "solution.py", crash)
        st = ma._stress_check(ws, multi=False)
        check("crashing solution reported as a mismatch", bool(st.get("mismatch")) and st["mismatch"]["solution"].startswith("<"))

        ma._write(ws, "solution.py", good)
        st = ma._stress_check(ws, multi=True)
        check("multi-answer problem: stress comparison skipped", not st.get("mismatch") and "skipped" in st)
        check("multi-answer problem: feedback explains the skip", "skipped" in ma._stress_feedback(st))

        ma._write(ws, "brute.py", "raise SystemExit(1)\n")
        st = ma._stress_check(ws, multi=False)
        check("broken brute force: cases skipped, no false mismatch", not st.get("mismatch") and st.get("brute_errors") == ma.STRESS_CASES)
        ma._write(ws, "brute.py", brute)

        ma._write(ws, "biggen.py", "print(10 ** 6)\n")
        st = ma._stress_check(ws, multi=False)
        check("fast solution on big input: timed, not too slow", st.get("big") and not ma._too_slow(st) and "solved in" in ma._stress_feedback(st))

        ma._write(ws, "solution.py", "import time\nn = int(input())\ntime.sleep(1.2)\nprint(n * (n + 1) // 2)\n")
        old = ma.STRESS_TIME_LIMIT
        ma.STRESS_TIME_LIMIT = 0.5
        ma._write(ws, "gen.py", "")
        st = ma._stress_check(ws, multi=False)
        check("slow solution on big input: reported too slow", ma._too_slow(st) and "too slow" in ma._stress_feedback(st))
        ma.STRESS_TIME_LIMIT = old

        ma._write(ws, "solution.py", crash)
        st = ma._stress_check(ws, multi=False)
        check("crash on big input: reported as a crash", st["big"]["error"] and "crashed" in ma._stress_feedback(st))

    with tempfile.TemporaryDirectory() as ws:
        ma._write(ws, "solution.py", good)
        check("no verification programs supplied: nothing to report", ma._stress_feedback(ma._stress_check(ws, False)) == "")

    reply = ("### CODE\n```python\nprint(1)\n```\n### BRUTE\n```python\nprint(2)\n```\n"
             "### GEN\n```python\nprint(3)\n```\n### BIGGEN\n```python\nprint(4)\n```\n"
             "### NOTES\n- x\n### NEXT\nnone\n### STATUS\nsolved\n")
    sec = ma._sections(reply)
    got = [ma._extract_py(sec.get(k, "")) for k in ("CODE", "BRUTE", "GEN", "BIGGEN")]
    check("worker reply: CODE, BRUTE, GEN and BIGGEN all parsed", got == ["print(1)", "print(2)", "print(3)", "print(4)"])
    check("CODE stays the first python block (parser fallback safety)", ma._extract_py(reply) == "print(1)")
    src = open(os.path.join(HERE, "multiagent.py")).read()
    check("worker format includes the new sections only when enabled", '(EXEC_FMT if (FIXES and EXEC) else "")' in src)

    off = subprocess.run([sys.executable, "-c", "import multiagent as m; print(m.EXEC)"], cwd=HERE,
                         capture_output=True, text=True, env={**os.environ, "MULTIAGENT_EXEC_CHECKS": ""})
    check("off by default", off.stdout.strip() == "False")

    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
