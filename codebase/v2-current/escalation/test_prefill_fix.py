"""Tests for dropping a repeated CODE heading or fence after the prefill, on the three real continuations
from arc196_d (version 4.4, 27 Oct). No dataset or server needed.

  MULTIAGENT_HARNESS_FIXES=1 MULTIAGENT_EXEC_CHECKS=1 MULTIAGENT_IMPLEMENT_SPLIT=1 MULTIAGENT_IMPLEMENT_PREFILL=1 python escalation/test_prefill_fix.py
"""
import ast, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MULTIAGENT_MODEL", "openai:test")


def main():
    import multiagent as ma
    fails = 0

    def check(name, cond):
        nonlocal fails
        print(("PASS " if cond else "FAIL ") + name)
        fails += 0 if cond else 1

    real = json.load(open(os.path.join(HERE, "fixtures", "prefill_continuations_arc196_d.json")))

    def code_of(continuation, fixed):
        reply = ma.PREFILL_TEXT + (ma._strip_repeated_opening(continuation) if fixed else continuation)
        return ma._extract_py(ma._sections(reply).get("CODE", "")) or ma._extract_py(reply)

    before = [len(code_of(c, False)) for c in real]
    after = [code_of(c, True) for c in real]
    check(f"before the fix: round 3's program is lost (extracted {before[2]} chars)", before[2] < 100)
    check(f"after the fix: round 3's program is kept ({len(after[2])} chars) and parses",
          len(after[2]) > 1000 and bool(ast.parse(after[2])))
    check(f"after the fix: round 2's program is still kept ({len(after[1])} chars) and parses",
          len(after[1]) > 1000 and bool(ast.parse(after[1])))
    check("round 1 wrote its brute force first: the BRUTE section is intact",
          len(ma._extract_py(ma._sections(ma.PREFILL_TEXT + ma._strip_repeated_opening(real[0])).get("BRUTE", ""))) > 200)
    check("a normal continuation is untouched", ma._strip_repeated_opening("import sys\nprint(1)\n```") == "import sys\nprint(1)\n```")
    check("only one repeated opening is removed", ma._strip_repeated_opening("```python\n```python\nx") == "```python\nx")
    for variant in ("```py\nx", "  ```\nx", "## CODE\n```python\nx", "### CODE\n\n```python\nx"):
        check(f"variant {variant[:18]!r} is normalised", ma._strip_repeated_opening(variant) == "x")

    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
