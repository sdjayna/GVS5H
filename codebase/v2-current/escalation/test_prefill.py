"""Tests for the implement prefill (MULTIAGENT_IMPLEMENT_PREFILL=1). No dataset or server needed.

  MULTIAGENT_HARNESS_FIXES=1 MULTIAGENT_EXEC_CHECKS=1 MULTIAGENT_IMPLEMENT_SPLIT=1 MULTIAGENT_IMPLEMENT_PREFILL=1 python escalation/test_prefill.py
"""
import json, os, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MULTIAGENT_MODEL", "openai:test")


def main():
    import multiagent as ma
    assert ma.IMPLEMENT_PREFILL and ma.SPLIT, "run with the flags in the docstring"
    fails = 0

    def check(name, cond):
        nonlocal fails
        print(("PASS " if cond else "FAIL ") + name)
        fails += 0 if cond else 1

    seen = []

    def fake_chat(model, messages, temperature=0.2, meta=None):
        seen.append(messages)
        meta["finish_reason"] = "stop"
        if messages[-1]["role"] == "assistant":   # a server continuing the prefill
            return "print(int(input()) * 2)\n```\n### NOTES\n- **Method:** double.\n### NEXT\nnone\n### STATUS\nsolved\n"
        return "### CODE\n```python\nprint(1)\n```\n### NOTES\n- **x:** y\n### NEXT\nnone\n### STATUS\nsolved\n"

    ma.chat = fake_chat
    ws = tempfile.mkdtemp()
    for f in ("plan.md", "notes.md", "solution.py", "answer.md", "task.md"):
        ma._write(ws, f, "")
    spec, log = {"kind": "code", "solver_system": "an expert."}, (lambda *a, **k: None)

    ma._worker("P", spec, ws, {"id": 1, "desc": "IMPLEMENT: double the input."}, log)
    check("implement request ends with the prefilled assistant message",
          seen[-1][-1] == {"role": "assistant", "content": ma.PREFILL_TEXT})
    check("the reply is parsed with the prefix restored: the program is extracted",
          ma._read(ws, "solution.py").strip() == "print(int(input()) * 2)")
    rec = [json.loads(l) for l in open(os.path.join(ws, "transcript.jsonl"))][-1]
    check("the transcript records what was sent and the full reply",
          rec["request"][-1]["role"] == "assistant" and rec["response"].startswith(ma.PREFILL_TEXT))

    ma._worker("P", spec, ws, {"id": 2, "desc": "Prove the greedy is optimal."}, log)
    check("full-thinking workers get no prefill", seen[-1][-1]["role"] != "assistant")
    ma._chat(ws, "primary_manage", [{"role": "user", "content": "x"}], 0.2)
    check("bookkeeping calls get no prefill", seen[-1][-1]["role"] != "assistant")

    off = subprocess.run([sys.executable, "-c", "import multiagent as m; print(m.IMPLEMENT_PREFILL)"], cwd=HERE,
                         capture_output=True, text=True, env={**os.environ, "MULTIAGENT_IMPLEMENT_PREFILL": ""})
    check("off by default", off.stdout.strip() == "False")

    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
