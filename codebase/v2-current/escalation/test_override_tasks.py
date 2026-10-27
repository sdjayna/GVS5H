"""Tests for implement-style override tasks (MULTIAGENT_OVERRIDE_IMPLEMENT=1). No dataset or server needed.

  MULTIAGENT_HARNESS_FIXES=1 MULTIAGENT_EXEC_CHECKS=1 MULTIAGENT_IMPLEMENT_SPLIT=1 MULTIAGENT_OVERRIDE_IMPLEMENT=1 python escalation/test_override_tasks.py
"""
import os, subprocess, sys, tempfile, types
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MULTIAGENT_MODEL", "openai:test")

WRONG = "n = int(input())\nprint(n * n)\n# long enough for the harvester, but wrong on the samples ...........\n"
GOOD = "n = int(input())\nprint(n * (n + 1) // 2)\n"
TESTS = [types.SimpleNamespace(testtype="stdin", input="3\n", output="6"),
         types.SimpleNamespace(testtype="stdin", input="4\n", output="10")]
fence = lambda c: "```python\n" + c + "```\n"


def main():
    import multiagent as ma
    assert ma.FIXES and ma.SPLIT and ma.OVERRIDE_IMPLEMENT, "run with the flags in the docstring"
    fails = 0

    def check(name, cond):
        nonlocal fails
        print(("PASS " if cond else "FAIL ") + name)
        fails += 0 if cond else 1

    t = ma._fix_samples_task({"passed": 1, "total": 2, "fail": {"input": "3\n", "expected": "6", "got": "9"}})
    check("sample-fix task is an implement task quoting the failing case", t.startswith("IMPLEMENT:") and "'3\\n'" in t and "'9'" in t)
    t = ma._fix_stress_task({"mismatch": {"input": "21\n", "brute": "231", "solution": "232"}})
    check("stress-fix task quotes the counterexample", t.startswith("IMPLEMENT:") and "'231'" in t and "'232'" in t)
    t = ma._fix_stress_task({"big": {"seconds": 12.0, "error": "timeout", "input_chars": 5347003}})
    check("too-slow task asks for speed and gives the input size", t.startswith("IMPLEMENT:") and "5347003" in t and "faster" in t)

    roles = []
    script = []

    def fake_chat(model, messages, temperature=0.2, meta=None):
        meta["finish_reason"] = "stop"
        return script.pop(0)

    ma.chat = fake_chat
    ma.REQUIRE_STRESS = False   # this test covers the override tasks only; test_verify covers the gate
    real_chat_once = ma._chat_once

    def spy(ws, role, messages, temperature, meta=None):
        roles.append(role)
        return real_chat_once(ws, role, messages, temperature, meta)

    ma._chat_once = spy
    script[:] = [
        "### PLAN\nx\n### TASKS\n- [todo] write it\n",
        "### PROPOSALS\n- a\n",
        "### STATUS\ncontinue\n### NEXT\nDerive the formula.\n### TASKS\n- [todo] x\n",
        "### CODE\n" + fence(WRONG) + "### NOTES\n- **x:** y\n### NEXT\nnone\n### STATUS\nsolved\n",
        "### STATUS\ndone\n### NEXT\n\n### TASKS\n- [done] x\n",   # done while the samples fail, no next task
        "### CODE\n" + fence(GOOD) + "### NOTES\n- **x:** y\n### NEXT\nnone\n### STATUS\nsolved\n",
        "### STATUS\ndone\n### NEXT\n\n### TASKS\n- [done] x\n",
    ]
    lines = []
    ma.WS_ROOT = tempfile.mkdtemp()
    ma.multiagent_solve("PROBLEM: print the n-th triangular number.", {"kind": "code", "solver_system": "an expert."},
                        log=lines.append, tests=TESTS)
    check("end to end: the manager's done is overridden because the samples fail",
          any("overriding 'done' -- sample tests still failing" in l for l in lines))
    check("end to end: the fix task runs as an implement worker", "implement:2" in roles)
    check("end to end: every scripted reply was used", not script)

    off = subprocess.run([sys.executable, "-c", "import multiagent as m; print(m.OVERRIDE_IMPLEMENT)"], cwd=HERE,
                         capture_output=True, text=True, env={**os.environ, "MULTIAGENT_OVERRIDE_IMPLEMENT": ""})
    check("off by default", off.stdout.strip() == "False")

    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
