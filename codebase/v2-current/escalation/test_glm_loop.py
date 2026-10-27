"""Tests for the loop engine (glm_loop.py) against a scripted fake server. No dataset or server needed.

  python escalation/test_glm_loop.py
"""
import json, os, sys, tempfile, types
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["LOOP_WS"] = tempfile.mkdtemp()
import glm_loop as gl

GOOD = "n = int(input())\nprint(n * (n + 1) // 2)\n"
SQUARE = "n = int(input())\nprint(n * n)\n"
OFF_BY_ONE = "n = int(input())\nprint(n * (n + 1) // 2 + (1 if n > 20 else 0))\n"
BRUTE = "n = int(input())\nprint(sum(range(n + 1)))\n"
GEN = "import random, sys\nrandom.seed(int(sys.argv[1]))\nprint(random.randint(1, 30) if sys.argv[2] == 'small' else 10 ** 6)\n"
TESTS = [types.SimpleNamespace(testtype="stdin", input="3\n", output="6"),
         types.SimpleNamespace(testtype="stdin", input="4\n", output="10")]
PROBLEM = "### Question\nPrint the n-th triangular number.\n"

fails = 0


def check(name, cond):
    global fails
    print(("PASS " if cond else "FAIL ") + name)
    fails += 0 if cond else 1


class Server:
    """Answers each step from a script: {step: [replies...]}; a reply is text or (text, finish, reasoning)."""
    def __init__(self, script):
        self.script, self.bodies = {k: list(v) for k, v in script.items()}, []

    @staticmethod
    def step_of(body):
        u = body["messages"][-2 if body["messages"][-1]["role"] == "assistant" else -1]["content"]
        for key, step in (("exactly one correct output", "classify"), ("Write a generator", "gen"),
                          ("different method from the most obvious", "brute2"),
                          ("simplest obviously correct", "brute"), ("Write a checker", "checker"),
                          ("was cut off", "summary"), ("Work out a correct algorithm", "derive"),
                          ("Implement this method", "implement"), ("Fix it", "implement")):
            if key in u:
                return step
        raise AssertionError("unknown step: " + u[:80])

    def __call__(self, body):
        step = self.step_of(body)
        self.bodies.append((step, body))
        r = self.script[step].pop(0)
        text, finish, reasoning = (r, "stop", "") if isinstance(r, str) else r
        return {"choices": [{"message": {"content": text, "reasoning_content": reasoning}, "finish_reason": finish}],
                "usage": {"completion_tokens": 100}}


def solve(script, tests=TESTS, problem=PROBLEM):
    srv = Server(script)
    gl._post = srv
    lines, status = [], {}
    out = gl.loop_solve(problem, {"kind": "code"}, log=lines.append, status_out=status, tests=tests)
    return srv, lines, status, out


def code_of(out):
    return out.split("```python\n", 1)[1].rsplit("```", 1)[0] if out else ""


def main():
    # unit checks
    check("clean_code drops a repeated opening fence", gl.clean_code("```python\nprint(1)") == "print(1)\n")
    check("clean_code drops a closing fence and what follows", gl.clean_code("print(1)\n```\ntext") == "print(1)\n")
    check("clean_code rejects a heading", gl.clean_code("### BRUTE") == "")
    check("clean_code rejects broken code", gl.clean_code("def f(:") == "")
    check("functional outputs compare as JSON", gl.same_output("[1, 2]\n", "[1,2]", "f") and not gl.same_output("[2, 1]", "[1,2]", "f"))
    check("stdin outputs compare token by token", gl.same_output("6 \n", "6", None))

    # 1. a wrong program is caught by the samples, then fixed
    srv, lines, status, out = solve({"classify": ["UNIQUE"], "gen": [GEN], "brute": [BRUTE],
                                     "derive": ["Reasoning.\nMETHOD:\nUse n(n+1)/2."], "implement": [SQUARE, GOOD]})
    check("samples: wrong program rejected, fix accepted", code_of(out) == GOOD and any("accepted" in l for l in lines))
    check("samples: the failing case is quoted to the fix", any("fails public sample 1" in b["messages"][1]["content"]
                                                                for s, b in srv.bodies if s == "implement"))
    impl = [b for s, b in srv.bodies if s == "implement"][0]
    check("implement requests: prefilled fence, stop at the closing fence, thinking off",
          impl["messages"][-1] == {"role": "assistant", "content": gl.FENCE_OPEN} and impl["stop"] == [gl.FENCE_STOP]
          and impl["reasoning_effort"] == "none")
    der = [b for s, b in srv.bodies if s == "derive"][0]
    check("derive requests: thinking on, no prefill, no stop", "reasoning_effort" not in der and "stop" not in der
          and der["messages"][-1]["role"] == "user")
    check("status reports calls and the outcome", status["n_calls"] == 6 and status["finish_reason"] == "stop")

    # 2. a sample-passing program is caught by the brute force
    srv, lines, status, out = solve({"classify": ["UNIQUE"], "gen": [GEN], "brute": [BRUTE],
                                     "derive": ["METHOD:\nformula"], "implement": [OFF_BY_ONE, GOOD]})
    check("stress: the off-by-one program is caught by the brute force",
          any("disagrees with the brute force" in b["messages"][1]["content"] for s, b in srv.bodies if s == "implement"))
    check("stress: the fixed program is accepted", code_of(out) == GOOD)

    # 3. a cut-off derive is recovered from its reasoning
    srv, lines, status, out = solve({"classify": ["UNIQUE"], "gen": [GEN], "brute": [BRUTE],
                                     "derive": [("", "length", "long reasoning ... use n(n+1)/2")],
                                     "summary": ["Use n(n+1)/2."], "implement": [GOOD]})
    check("cut-off derive: the summary becomes the method", any("from a cut-off summary" in l for l in lines)
          and code_of(out) == GOOD)

    # 4. two failed fixes in a row: the method is re-derived with the history
    srv, lines, status, out = solve({"classify": ["UNIQUE"], "gen": [GEN], "brute": [BRUTE],
                                     "derive": ["METHOD:\nsquare it", "METHOD:\ntriangular"],
                                     "implement": [SQUARE, SQUARE, GOOD]})
    derives = [b for s, b in srv.bodies if s == "derive"]
    check("re-derive after two failed fixes", len(derives) == 2 and code_of(out) == GOOD)
    check("re-derive: the history of failures is in the request", "Attempt 1" in derives[1]["messages"][0]["content"])

    # 5. a reply with no program counts as a failed round
    srv, lines, status, out = solve({"classify": ["UNIQUE"], "gen": [GEN], "brute": [BRUTE],
                                     "derive": ["METHOD:\nx"], "implement": ["### BRUTE", GOOD]})
    check("no program: logged and retried", any("no program" in l for l in lines) and code_of(out) == GOOD)

    # 6. a functional (LeetCode-style) problem runs through the wrapper
    fproblem = PROBLEM + "```python\nclass Solution:\n    def tri(self, n: int) -> int:\n        \n```\n"
    ftests = [types.SimpleNamespace(testtype="functional", input="3", output="6"),
              types.SimpleNamespace(testtype="functional", input="4", output="10")]
    fgood = "class Solution:\n    def tri(self, n: int) -> int:\n        return n * (n + 1) // 2\n"
    fbrute = "class Solution:\n    def tri(self, n: int) -> int:\n        return sum(range(n + 1))\n"
    srv, lines, status, out = solve({"classify": ["UNIQUE"], "gen": [GEN], "brute": [fbrute],
                                     "derive": ["METHOD:\nformula"], "implement": [fgood]}, tests=ftests, problem=fproblem)
    check("functional: samples and stress run through the wrapper, accepted",
          code_of(out) == fgood and any("[samples] 2/2" in l and "40 agree" in l for l in lines))

    # 7. more than one correct output: a checker judges, not exact match
    mproblem = "### Question\nPrint any integer whose absolute value is n.\n"
    mtests = [types.SimpleNamespace(testtype="stdin", input="3\n", output="3")]
    checker = ("import sys\nn = int(open(sys.argv[1]).read())\nx = open(sys.argv[2]).read().split()\n"
               "print('OK' if len(x) == 1 and abs(int(x[0])) == n else 'WRONG')\n")
    neg = "n = int(input())\nprint(-n)\n"
    srv, lines, status, out = solve({"classify": ["MULTIPLE"], "gen": [GEN], "checker": [checker],
                                     "derive": ["METHOD:\nprint -n"], "implement": [neg]}, tests=mtests, problem=mproblem)
    check("multiple outputs: a valid different answer passes the checker", code_of(out) == neg and
          any("accepted" in l for l in lines))
    check("multiple outputs: no brute force is built", not any(s == "brute" for s, _ in srv.bodies))

    # 8. the brute force is the one that is wrong (3697, 27 Oct): a second brute force outvotes it
    BAD_BRUTE = "n = int(input())\nprint(sum(range(n + 1)) + (1 if n > 20 else 0))\n"
    srv, lines, status, out = solve({"classify": ["UNIQUE"], "gen": [GEN], "brute": [BAD_BRUTE], "brute2": [BRUTE],
                                     "derive": ["METHOD:\nformula"], "implement": [GOOD]})
    check("dispute: a wrong brute force is replaced and the correct program accepted unchanged",
          code_of(out) == GOOD and any("brute force replaced" in l for l in lines)
          and sum(s == "implement" for s, _ in srv.bodies) == 1)

    # 9. the program is the one that is wrong: two independent brute forces agree
    srv, lines, status, out = solve({"classify": ["UNIQUE"], "gen": [GEN], "brute": [BRUTE], "brute2": [BRUTE],
                                     "derive": ["METHOD:\nformula"], "implement": [OFF_BY_ONE, GOOD]})
    check("dispute: two brute forces agree, the program is fixed",
          code_of(out) == GOOD and any("two independent brute forces agree" in l for l in lines))

    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
