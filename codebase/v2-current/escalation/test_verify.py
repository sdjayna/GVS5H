"""Tests for the stress-before-done rule (MULTIAGENT_REQUIRE_STRESS=1). No dataset or server needed.

  MULTIAGENT_HARNESS_FIXES=1 MULTIAGENT_EXEC_CHECKS=1 MULTIAGENT_IMPLEMENT_SPLIT=1 MULTIAGENT_REQUIRE_STRESS=1 python escalation/test_verify.py
"""
import json, os, subprocess, sys, tempfile, types
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MULTIAGENT_MODEL", "openai:test")

GOOD = "n = int(input())\nprint(n * (n + 1) // 2)\n"
WRONG = "n = int(input())\nprint(n * (n + 1) // 2 + (1 if n > 20 else 0))\n"
BRUTE = "n = int(input())\nprint(sum(range(n + 1)))\n"
GEN = "import random, sys\nrandom.seed(int(sys.argv[1]))\nprint(random.randint(1, 30))\n"
TESTS = [types.SimpleNamespace(testtype="stdin", input="3\n", output="6"),
         types.SimpleNamespace(testtype="stdin", input="4\n", output="10")]
fence = lambda c: "```python\n" + c + "```\n"


def main():
    import multiagent as ma
    assert ma.FIXES and ma.EXEC and ma.REQUIRE_STRESS, "run with the four flags in the docstring"
    fails = 0

    def check(name, cond):
        nonlocal fails
        print(("PASS " if cond else "FAIL ") + name)
        fails += 0 if cond else 1

    spec, log = {"kind": "code", "solver_system": "an expert."}, (lambda *a, **k: None)
    ws = tempfile.mkdtemp()
    ma._write(ws, "solution.py", GOOD)
    st, nd, n = ma._verify_gate("done", "", ws, spec, TESTS, "PROBLEM", None, 0, log)
    check("done on an unverified solution becomes a verification task", st == "continue" and nd.startswith("IMPLEMENT: Verify") and n == 1)
    st, nd2, n = ma._verify_gate("done", "", ws, spec, TESTS, "PROBLEM", None, 1, log)
    check("second override uses a different task text (the no-progress guard compares texts)", st == "continue" and nd2 != nd and n == 2)
    st, _, n = ma._verify_gate("done", "", ws, spec, TESTS, "PROBLEM", None, 2, log)
    check("at most two overrides, then done is accepted", st == "done" and n == 2)
    st, _, _ = ma._verify_gate("done", "", ws, spec, TESTS, "PROBLEM", GOOD, 0, log)
    check("a solution that passed a stress test is accepted", st == "done")
    st, _, _ = ma._verify_gate("done", "", ws, spec, TESTS, "If there are multiple solutions, print any of them.", None, 0, log)
    check("problems that accept more than one output are exempt", st == "done")
    st, nd, _ = ma._verify_gate("continue", "next task", ws, spec, TESTS, "PROBLEM", None, 0, log)
    check("continue is left alone", st == "continue" and nd == "next task")
    st, _, _ = ma._verify_gate("done", "", ws, {"kind": "answer"}, TESTS, "PROBLEM", None, 0, log)
    check("non-code problems are exempt", st == "done")

    # end to end: the planner writes a sample-passing but wrong program; the manager says done at once
    script = []

    def fake_chat(model, messages, temperature=0.2, meta=None):
        # one level below _chat, so every call is still recorded in the transcript (harvest reads it)
        reply = script.pop(0)(None)
        if meta is not None:
            meta["finish_reason"] = "stop"
        return reply

    ma.chat = fake_chat
    replies = [
        lambda r: "### PLAN\nx\n" + fence(WRONG + "# padded so the harvester accepts it" * 3),   # plan writes code
        lambda r: "### PROPOSALS\n- a\n",                                                         # ideation
        lambda r: "### STATUS\ndone\n### NEXT\n\n### TASKS\n- [done] x\n",                          # manager: done at once
        lambda r: ("### CODE\n" + fence(WRONG) + "### BRUTE\n" + fence(BRUTE) + "### GEN\n" + fence(GEN)
                   + "### NOTES\n- **x:** y\n### NEXT\nnone\n### STATUS\nsolved\n"),              # verification worker
        lambda r: "### STATUS\ndone\n### NEXT\n\n### TASKS\n- [done] x\n",                          # manager: done again
        lambda r: ("### CODE\n" + fence(GOOD) + "### BRUTE\n" + fence(BRUTE) + "### GEN\n" + fence(GEN)
                   + "### NOTES\n- **x:** y\n### NEXT\nnone\n### STATUS\nsolved\n"),              # second verification fixes it
        lambda r: "### STATUS\ndone\n### NEXT\n\n### TASKS\n- [done] x\n",                          # manager: done
    ]
    script[:] = replies
    calls_before = len(script)
    lines = []
    ma.WS_ROOT = tempfile.mkdtemp()
    ma.multiagent_solve("PROBLEM: print the n-th triangular number.", spec, log=lines.append, tests=TESTS)
    sol = open(os.path.join(ma.WS_ROOT, os.listdir(ma.WS_ROOT)[0], "solution.py")).read()
    check("end to end: the planner's sample-passing program is harvested", any("[harvest]" in l for l in lines))
    check("end to end: done on the unverified harvested program is refused", any("[verify]" in l for l in lines))
    check("end to end: the stress test caught the wrong program", any("[stress] mismatch=True" in l for l in lines))
    check("end to end: the final solution is the verified correct one", sol.strip() == GOOD.strip())
    check("end to end: every scripted reply was used", not script)

    off = subprocess.run([sys.executable, "-c", "import multiagent as m; print(m.REQUIRE_STRESS)"], cwd=HERE,
                         capture_output=True, text=True, env={**os.environ, "MULTIAGENT_REQUIRE_STRESS": ""})
    check("off by default", off.stdout.strip() == "False")

    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
