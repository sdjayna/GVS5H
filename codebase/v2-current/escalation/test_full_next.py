"""Tests for the whole-task parser and automatic implement routing. No dataset or server needed.

  MULTIAGENT_HARNESS_FIXES=1 MULTIAGENT_IMPLEMENT_SPLIT=1 MULTIAGENT_FULL_NEXT=1 MULTIAGENT_AUTO_IMPLEMENT=1 python escalation/test_full_next.py
"""
import os, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MULTIAGENT_MODEL", "openai:test")

# the manager reply that lost its method on abc398_g (27 Oct, version 4.2), shortened
REAL_NEXT = ("IMPLEMENT: Solve the game by computing the parity of the total number of moves, which equals "
             "ab - M. Method, fully worked out:\n\n1. Read N, M, and edges. Build adjacency lists.\n"
             "2. 2-color each component and record (x_c, y_c).\n3. Decide the winner from the parity.\n")


def main():
    import multiagent as ma
    assert ma.FIXES and ma.SPLIT and ma.FULL_NEXT and ma.AUTO_IMPLEMENT, "run with the four flags in the docstring"
    fails = 0

    def check(name, cond):
        nonlocal fails
        print(("PASS " if cond else "FAIL ") + name)
        fails += 0 if cond else 1

    t = ma._next_task(REAL_NEXT)
    check("multi-line NEXT keeps its header and every step", t.startswith("IMPLEMENT:") and "3. Decide the winner" in t)
    old = (ma._bullets(REAL_NEXT)[:1] or [REAL_NEXT.strip()])[0]
    check("the old parser kept only step 1 (the bug, for the record)", old.startswith("Read N, M") and "IMPLEMENT" not in old)
    check("a single bullet is still unwrapped", ma._next_task("- Write the brute force.") == "Write the brute force.")
    check("a single plain line is kept", ma._next_task("Derive the recurrence.") == "Derive the recurrence.")
    check("long NEXT is bounded", len(ma._next_task("x" * 20000)) == ma.MAX_NEXT_CHARS)

    for desc, want in (("Worker A: Write `sol.py` implementing the interval model", True),
                       ("Implement the Fenwick sweep", True), ("Task 2: implement it", True),
                       ("Rewrite the solution as clean Python", True), ("code the DP", True),
                       ("Derive the recurrence for f(k, d)", False), ("Prove that the greedy is optimal", False),
                       ("Hand-verify the greedy on sample 2", False), ("Writer's note: think first", False)):
        check(f"code task detection: {desc[:40]!r} -> {want}", ma._is_code_task(desc) == want)

    calls = []

    def fake_chat(ws, role, messages, temperature, meta=None):
        calls.append(role)
        if meta is not None:
            meta["finish_reason"] = "stop"
        return "### CODE\n```python\nprint(1)\n```\n### NOTES\n- **x:** y\n### NEXT\nnone\n### STATUS\nsolved\n"

    ma._chat = fake_chat
    ws = tempfile.mkdtemp()
    for f in ("plan.md", "notes.md", "solution.py", "answer.md", "task.md"):
        ma._write(ws, f, "")
    spec, log = {"kind": "code", "solver_system": "an expert."}, (lambda *a, **k: None)
    ma._worker("P", spec, ws, {"id": 2, "desc": "Worker A: Write `sol.py` implementing the interval model"}, log)
    ma._worker("P", spec, ws, {"id": 3, "desc": "Prove that the greedy is optimal"}, log)
    ma._worker("P", spec, ws, {"id": 4, "desc": ma._next_task(REAL_NEXT)}, log)
    check("untagged code task runs as implement", calls[0] == "implement:2")
    check("analysis task keeps full thinking", calls[1] == "worker:3")
    check("the real multi-line IMPLEMENT task now reaches the implement role", calls[2] == "implement:4")

    off = subprocess.run([sys.executable, "-c", "import multiagent as m; print(m.FULL_NEXT, m.AUTO_IMPLEMENT)"], cwd=HERE,
                         capture_output=True, text=True,
                         env={**os.environ, "MULTIAGENT_FULL_NEXT": "", "MULTIAGENT_AUTO_IMPLEMENT": ""})
    check("off by default", off.stdout.strip() == "False False")

    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
