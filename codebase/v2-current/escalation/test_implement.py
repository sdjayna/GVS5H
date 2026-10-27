"""Tests for the implement split (MULTIAGENT_IMPLEMENT_SPLIT=1). No dataset or server needed.

  MULTIAGENT_HARNESS_FIXES=1 MULTIAGENT_EXEC_CHECKS=1 MULTIAGENT_IMPLEMENT_SPLIT=1 python escalation/test_implement.py
"""
import os, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MULTIAGENT_MODEL", "openai:test")
os.environ["MULTIAGENT_ROLE_EFFORT"] = "primary_plan=none,implement=none"
os.environ["MULTIAGENT_ROLE_MAX_TOKENS"] = "implement=48000"

CODE_REPLY = ("### CODE\n```python\nprint(int(input()) * 2)\n```\n### NOTES\n- **Method:** double it.\n"
              "### NEXT\nnone\n### STATUS\nsolved\n")


def main():
    import multiagent as ma
    assert ma.FIXES and ma.SPLIT, "run with MULTIAGENT_HARNESS_FIXES=1 MULTIAGENT_IMPLEMENT_SPLIT=1"
    fails = 0

    def check(name, cond):
        nonlocal fails
        print(("PASS " if cond else "FAIL ") + name)
        fails += 0 if cond else 1

    calls = []

    def fake_chat(ws, role, messages, temperature, meta=None):
        calls.append({"role": role, "system": messages[0]["content"]})
        if role == "cutoff_summary":
            return "Method found: answer is 2n. Edge case n = 0."
        if meta is not None:
            meta["finish_reason"] = fake_chat.finish
        return fake_chat.reply

    ma._chat = fake_chat
    spec = {"kind": "code", "solver_system": "an expert programmer."}
    log = lambda *a, **k: None

    def run_worker(desc, finish="stop", reply=CODE_REPLY, finalize=False):
        calls.clear()
        fake_chat.finish, fake_chat.reply = finish, reply
        ws = tempfile.mkdtemp()
        for f in ("plan.md", "notes.md", "solution.py", "answer.md", "task.md"):
            ma._write(ws, f, "")
        ma._write(ws, "notes.md", "- **Earlier:** something.\n")
        ma._worker("PROBLEM TEXT", spec, ws, {"id": 3, "desc": desc}, log, finalize=finalize)
        return ws

    run_worker("IMPLEMENT: double the input; O(1).")
    check("IMPLEMENT task runs as role implement:3", calls and calls[0]["role"] == "implement:3")
    check("IMPLEMENT task is told not to re-derive", calls and ma.IMPLEMENT_SELF in calls[0]["system"])
    run_worker("  implement: lower-case tag also counts")
    check("tag is case-insensitive and ignores leading spaces", calls and calls[0]["role"] == "implement:3")
    run_worker("Derive the recurrence for f(k, d).")
    check("normal task stays role worker:3", calls and calls[0]["role"] == "worker:3")
    check("normal task gets no implement instruction", calls and ma.IMPLEMENT_SELF not in calls[0]["system"])
    run_worker("IMPLEMENT: anything", finalize=True)
    check("finalize keeps its own role", calls and calls[0]["role"] == "finalize")

    check("role settings resolve for implement (effort none, 48000 tokens)",
          ma.ROLE_EFFORT.get("implement") == "none" and ma.ROLE_MAX_TOKENS.get("implement") == "48000")

    ws = run_worker("Derive it.", finish="length", reply="I was thinking and thinking")
    notes = ma._read(ws, "notes.md")
    check("cut-off summary is kept in the notes", "Cut-off worker summary (task 3)" in notes and "answer is 2n" in notes)
    check("earlier notes are preserved", "**Earlier:** something." in notes)

    calls.clear()
    ws = tempfile.mkdtemp()
    for f in ("notes.md", "solution.py", "answer.md"):
        ma._write(ws, f, "")
    fake_chat.finish, fake_chat.reply = "stop", "### STATUS\ncontinue\n### NEXT\nIMPLEMENT: x\n### TASKS\n- [todo] x\n"
    ma._primary_manage("PROBLEM TEXT", spec, ws, [], [], "start", log)
    check("manager is told about IMPLEMENT tasks", calls and ma.MANAGE_SPLIT in calls[0]["system"])

    off = subprocess.run([sys.executable, "-c", "import multiagent as m; print(m.SPLIT)"], cwd=HERE,
                         capture_output=True, text=True, env={**os.environ, "MULTIAGENT_IMPLEMENT_SPLIT": ""})
    check("off by default", off.stdout.strip() == "False")

    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
