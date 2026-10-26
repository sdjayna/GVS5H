"""Tests for harvesting programs from bookkeeping steps and per-role token limits. No dataset needed.

  MULTIAGENT_HARNESS_FIXES=1 MULTIAGENT_EXEC_CHECKS=1 python escalation/test_harvest.py
"""
import json, os, subprocess, sys, tempfile, types
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MULTIAGENT_MODEL", "openai:test")
os.environ["MULTIAGENT_ROLE_MAX_TOKENS"] = "primary_plan=48000,cutoff_summary=16000"

GOOD = "import sys\nn = int(sys.stdin.readline())\nprint(n * (n + 1) // 2)\n# a complete program, long enough to count\n"
WRONG = "import sys\nn = int(sys.stdin.readline())\nprint(n * n)\n# a complete program, long enough to count, but wrong\n"
TESTS = [types.SimpleNamespace(testtype="stdin", input="3\n", output="6"),
         types.SimpleNamespace(testtype="stdin", input="10\n", output="55")]


def main():
    import multiagent as ma
    import orchestrator as orch
    assert ma.FIXES and ma.EXEC, "run with MULTIAGENT_HARNESS_FIXES=1 MULTIAGENT_EXEC_CHECKS=1"
    fails = 0

    def check(name, cond):
        nonlocal fails
        print(("PASS " if cond else "FAIL ") + name)
        fails += 0 if cond else 1

    def ws_with(role, response, reasoning="", solution=""):
        ws = tempfile.mkdtemp()
        ma._write(ws, "solution.py", solution)
        with open(os.path.join(ws, "transcript.jsonl"), "w") as f:
            f.write(json.dumps({"_meta": True, "t": 0}) + "\n")
            f.write(json.dumps({"role": role, "response": response, "reasoning": reasoning}) + "\n")
        return ws

    spec, log = {"kind": "code"}, (lambda *a, **k: None)
    fence = lambda c: "```python\n" + c + "```\n"

    ws = ws_with("primary_plan", "### PLAN\nparity\n" + fence(GOOD))
    check("planner's complete passing program is harvested", ma._harvest_code(ws, spec, TESTS, log, "plan") == GOOD.strip()
          and ma._read(ws, "solution.py").strip() == GOOD.strip())

    ws = ws_with("primary_plan", "### PLAN\n" + "```python\n" + GOOD[:40])
    check("program cut off at the limit (no closing fence) is skipped", ma._harvest_code(ws, spec, TESTS, log, "plan") is None
          and ma._read(ws, "solution.py") == "")

    ws = ws_with("ideation", fence("def f(:\n    pass\n" + "#" * 80))
    check("program that does not parse is skipped", ma._harvest_code(ws, spec, TESTS, log, "ideation") is None)

    ws = ws_with("primary_manage", fence(WRONG), solution="")
    check("failing program is not kept and the old solution is restored", ma._harvest_code(ws, spec, TESTS, log, "manage") is None
          and ma._read(ws, "solution.py") == "")

    ws = ws_with("primary_plan", fence(WRONG), reasoning=fence(GOOD))
    check("reply tried first, then the reasoning", ma._harvest_code(ws, spec, TESTS, log, "plan") == GOOD.strip())

    ws = ws_with("primary_plan", fence(GOOD), solution=GOOD.replace("count", "keep"))
    check("existing passing solution is left alone", ma._harvest_code(ws, spec, TESTS, log, "plan") is None
          and "keep" in ma._read(ws, "solution.py"))

    ws = ws_with("worker:1", fence(GOOD))
    check("worker replies are never harvested (their BRUTE could pass the samples)", ma._harvest_code(ws, spec, TESTS, log, "w") is None)

    ws = ws_with("primary_plan", fence(GOOD))
    check("non-code problems are untouched", ma._harvest_code(ws, {"kind": "answer"}, TESTS, log, "plan") is None)

    check("per-role limits parsed", ma.ROLE_MAX_TOKENS == {"primary_plan": "48000", "cutoff_summary": "16000"})
    sent = {}
    orig = orch.urllib.request.urlopen if hasattr(orch, "urllib") else None

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                               "usage": {"completion_tokens": 1, "prompt_tokens": 1}}).encode()

    def fake_urlopen(req, *a, **k):
        sent.update(json.loads(req.data))
        return FakeResponse()

    if orig:
        orch.urllib.request.urlopen = fake_urlopen
        for meta, label in (({"max_tokens_override": 48000}, "override"), ({}, "default")):
            sent.clear()
            try:
                orch.openai_chat("http://x", "k", "m", [{"role": "user", "content": "hi"}], meta=meta)
            except BaseException:
                pass
            check(f"max_tokens sent with {label}: {sent.get('max_tokens')}",
                  sent.get("max_tokens") == (48000 if label == "override" else orch.CLOUD_MAX_TOKENS))
        orch.urllib.request.urlopen = orig
    else:
        check("orchestrator uses urllib (needed for the request check)", False)

    off = subprocess.run([sys.executable, "-c",
                          "import multiagent as m; print(m._harvest_code('/nonexistent', {'kind': 'code'}, [1], print, 'x'))"],
                         cwd=HERE, capture_output=True, text=True, env={**os.environ, "MULTIAGENT_EXEC_CHECKS": ""})
    check("off without MULTIAGENT_EXEC_CHECKS", off.stdout.strip() == "None")

    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
