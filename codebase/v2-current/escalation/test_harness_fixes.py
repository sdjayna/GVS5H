"""Tests for the gated harness fixes (MULTIAGENT_HARNESS_FIXES=1).

Run from codebase/v2-current with the LiveCodeBench dataset available:
  MULTIAGENT_HARNESS_FIXES=1 python escalation/test_harness_fixes.py
"""
import json, os, sys, tempfile, types
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "codebase", "livecodebench"))
os.environ.setdefault("MULTIAGENT_MODEL", "openai:test")


def main():
    import multiagent as ma
    assert ma.FIXES, "run with MULTIAGENT_HARNESS_FIXES=1"
    fails = 0

    def check(name, cond):
        nonlocal fails
        print(("PASS " if cond else "FAIL ") + name)
        fails += 0 if cond else 1

    # 2. detector against the real prompts the benchmark builds, for all 100 problems
    import run_bench
    from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
    by = {p.question_id: p for p in load_code_generation_dataset(release_version="release_v6")}
    ids = json.load(open(os.path.join(HERE, "lcb100_hardest_v6.json")))[:100]
    expected = {"abc396_e", "abc397_d", "abc398_f", "arc191_c", "arc190_a", "arc192_b"}
    matched = {q for q in ids if ma.MULTI_OK.search(run_bench.build_code_prompt(by[q]))}
    check(f"detector matches exactly the 6 known problems (matched {sorted(matched)})", matched == expected)

    # 3. sample check and feedback
    Test = lambda i, o: types.SimpleNamespace(testtype="stdin", input=i, output=o)
    with tempfile.TemporaryDirectory() as ws:
        tests = [Test("1\n", "2\n"), Test("5\n", "6\n")]
        ma._write(ws, "solution.py", "print(int(input()) + 1)\n")
        res = ma._run_samples(ws, tests)
        check("exact pass reported as passed", res["passed"] == 2 and res["errors"] == 0)
        check("exact pass feedback says PASSED", "PASSED all 2" in ma._sample_feedback(res))

        ma._write(ws, "solution.py", "print(int(input()) + 100)\n")
        res = ma._run_samples(ws, tests)
        res["multi"] = True
        fb = ma._sample_feedback(res)
        check("valid-alternative answer on a multi-answer problem reported as DIFFERS", "OUTPUT DIFFERS" in fb and "FAILED" not in fb)
        res["multi"] = False
        check("same mismatch on a single-answer problem still reported as FAILED", "FAILED" in ma._sample_feedback(res))

        ma._write(ws, "solution.py", "raise SystemExit(1)\n")
        res = ma._run_samples(ws, tests)
        res["multi"] = True
        check("runtime error counted as an error", res["errors"] == 2)
        check("runtime error on a multi-answer problem still reported as FAILED", "FAILED" in ma._sample_feedback(res))

        # 4. keep-best
        good = "print(int(input()) + 1)\n"
        ma._write(ws, "solution.py", "print('broken')\n")
        restored = ma._restore_best(ws, tests, good, lambda *a, **k: None)
        check("keep-best restores the passing solution when the final one fails", restored and ma._read(ws, "solution.py") == good)
        ma._write(ws, "solution.py", good)
        check("keep-best leaves a passing final solution alone", not ma._restore_best(ws, tests, "print(0)\n", lambda *a, **k: None))
        check("keep-best does nothing without a passing solution on record", not ma._restore_best(ws, tests, None, lambda *a, **k: None))

    # 1. the prompts carry the new rule only when the fixes are on
    check("worker-limits rule is defined", "cannot run code" in ma.WORKER_LIMITS)
    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
