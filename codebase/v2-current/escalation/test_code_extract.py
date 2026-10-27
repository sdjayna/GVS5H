"""Tests for strict code extraction (MULTIAGENT_STRICT_CODE=1), on the real brute-first reply from arc196_d
(version 4.6, 27 Oct). No dataset or server needed.

  MULTIAGENT_HARNESS_FIXES=1 MULTIAGENT_EXEC_CHECKS=1 MULTIAGENT_IMPLEMENT_SPLIT=1 MULTIAGENT_IMPLEMENT_PREFILL=1 MULTIAGENT_STRICT_CODE=1 python escalation/test_code_extract.py
"""
import json, os, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MULTIAGENT_MODEL", "openai:test")


def main():
    import multiagent as ma
    assert ma.FIXES and ma.STRICT_CODE and ma.IMPLEMENT_PREFILL, "run with the flags in the docstring"
    fails = 0

    def check(name, cond):
        nonlocal fails
        print(("PASS " if cond else "FAIL ") + name)
        fails += 0 if cond else 1

    cont = json.load(open(os.path.join(HERE, "fixtures", "prefill_brute_first_arc196_d.json")))["continuation"]

    def fake_chat(model, messages, temperature=0.2, meta=None):
        meta["finish_reason"] = "length"
        return fake_chat.reply

    ma.chat = fake_chat
    ma._summarize_cutoff = lambda ws, reply, desc: "summary"
    spec, log = {"kind": "code", "solver_system": "an expert."}, (lambda *a, **k: None)

    def run(continuation, previous="print('previous good solution')\n"):
        fake_chat.reply = continuation
        ws = tempfile.mkdtemp()
        for f in ("plan.md", "notes.md", "answer.md", "task.md"):
            ma._write(ws, f, "")
        ma._write(ws, "solution.py", previous)
        ma._worker("P", spec, ws, {"id": 1, "desc": "IMPLEMENT: brute force first"}, log)
        return ws

    ws = run(cont)
    check("real brute-first reply: the previous solution is not replaced by '### BRUTE'",
          ma._read(ws, "solution.py") == "print('previous good solution')\n")
    check("real brute-first reply: the brute force and generator are saved",
          len(ma._read(ws, "brute.py")) > 1000 and len(ma._read(ws, "gen.py")) > 300)

    ws = run("print(int(input()) * 2)\n```\n### BRUTE\n```python\nprint(2)\n```\n")
    check("normal prefilled reply: the program is saved", ma._read(ws, "solution.py").strip() == "print(int(input()) * 2)")
    check("normal prefilled reply: the brute force is saved", ma._read(ws, "brute.py").strip() == "print(2)")

    check("a heading alone has no statements", not ma._has_statements("### BRUTE"))
    check("a real program has statements", ma._has_statements("print(1)"))
    check("CODE section wins over earlier blocks",
          ma._worker_code({"CODE": "```python\nprint(1)\n```"}, "```python\nx = 0\n```") == "print(1)")
    check("the anywhere-fallback still works when the reply ignored the format",
          ma._worker_code({}, "Here it is:\n```python\nprint(3)\n```") == "print(3)")
    check("no fallback when the reply has sections but no CODE",
          ma._worker_code({"BRUTE": "```python\nprint(2)\n```"}, "### BRUTE\n```python\nprint(2)\n```") == "")

    off = subprocess.run([sys.executable, "-c", "import multiagent as m; print(m.STRICT_CODE)"], cwd=HERE,
                         capture_output=True, text=True, env={**os.environ, "MULTIAGENT_STRICT_CODE": ""})
    check("off by default", off.stdout.strip() == "False")

    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
