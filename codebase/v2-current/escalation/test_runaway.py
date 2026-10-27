"""Tests for the runaway retry (MULTIAGENT_RUNAWAY_RETRY). No dataset or server needed.

  MULTIAGENT_RUNAWAY_RETRY=primary_plan,primary_manage python escalation/test_runaway.py
"""
import json, os, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MULTIAGENT_MODEL", "openai:test")


def main():
    import multiagent as ma
    assert ma.RUNAWAY_RETRY == {"primary_plan", "primary_manage"}, "run with MULTIAGENT_RUNAWAY_RETRY=primary_plan,primary_manage"
    fails = 0

    def check(name, cond):
        nonlocal fails
        print(("PASS " if cond else "FAIL ") + name)
        fails += 0 if cond else 1

    script = []

    def fake_chat(model, messages, temperature=0.2, meta=None):
        finish, text = script.pop(0)
        meta["finish_reason"] = finish
        return text

    ma.chat = fake_chat

    def run(role, outcomes, meta=None):
        script[:] = list(outcomes)
        ws = tempfile.mkdtemp()
        out = ma._chat(ws, role, [{"role": "user", "content": "x"}], 0.2, meta=meta)
        recs = [json.loads(l) for l in open(os.path.join(ws, "transcript.jsonl"))]
        return out, recs, len(script)

    out, recs, left = run("primary_plan", [("length", "analysis..."), ("stop", "### PLAN\nok")])
    check("runaway plan is re-sampled and the second reply returned", out == "### PLAN\nok")
    check("both attempts are recorded", [r["finish_reason"] for r in recs] == ["length", "stop"])
    out, recs, left = run("primary_plan", [("stop", "### PLAN\nfine"), ("stop", "unused")])
    check("normal plan is not re-sent", out == "### PLAN\nfine" and len(recs) == 1 and left == 1)
    out, recs, left = run("primary_manage", [("length", "a"), ("length", "b"), ("stop", "unused")])
    check("at most one retry, even if it runs away again", out == "b" and len(recs) == 2 and left == 1)
    out, recs, left = run("worker:3", [("length", "long"), ("stop", "unused")])
    check("worker calls are never re-sent (the cut-off summary handles them)", out == "long" and len(recs) == 1)
    out, recs, left = run("ideation", [("length", "long"), ("stop", "unused")])
    check("roles not listed are never re-sent", len(recs) == 1)
    wmeta = {}
    run("primary_plan", [("length", "a"), ("stop", "b")], meta=wmeta)
    check("caller's meta describes the call that was kept", wmeta.get("finish_reason") == "stop")

    off = subprocess.run([sys.executable, "-c", "import multiagent as m; print(m.RUNAWAY_RETRY)"], cwd=HERE,
                         capture_output=True, text=True, env={**os.environ, "MULTIAGENT_RUNAWAY_RETRY": ""})
    check("off by default", off.stdout.strip() == "set()")

    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
