# Running the escalation harness against a local model

This branch adds what was needed to run GVS5H's `multiagent` engine against GLM-5.3-Flash served locally through an OpenAI-compatible API, and to score the results fairly. Every change is opt-in: with none of the variables below set, the harness behaves as upstream at `707e2129`.

## Settings

| Variable | Effect |
| --- | --- |
| `ESCALATION_OPENAI_BASE` | Chat-completions URL for the `openai:` provider, for example `http://127.0.0.1:8000/v1/chat/completions`. Unset: api.openai.com |
| `MULTIAGENT_ROLE_EFFORT` | Per-role `reasoning_effort`, for example `primary_plan=none,ideation=none,primary_manage=none,cutoff_summary=none`. Roles not listed use the server default |
| `MULTIAGENT_HARNESS_FIXES=1` | Workers are known not to execute anything, the best passing solution is kept, and the statement decides whether a sample mismatch is a failure |
| `MULTIAGENT_EXEC_CHECKS=1` | On top of the fixes: workers may supply a brute force and input generators, which the harness runs (counterexamples and timing), and a complete program written by a planning, approach or manager step is kept if it passes every public sample |
| `MULTIAGENT_IMPLEMENT_SPLIT=1` | The manager may start a task with `IMPLEMENT:` and the full method when only the program remains. That task runs as role `implement`, so the two per-role settings can give it thinking off and a short limit. Cut-off workers' summaries are kept in the notes |
| `MULTIAGENT_ROLE_MAX_TOKENS` | Per-role output limit, for example `primary_plan=48000,ideation=48000,primary_manage=48000,cutoff_summary=16000`. Roles not listed use `ESCALATION_CLOUD_MAX_TOKENS` |

Example run of the 100 hardest LiveCodeBench problems:

```sh
export OPENAI_API_KEY=... ESCALATION_OPENAI_BASE=http://127.0.0.1:8000/v1/chat/completions
export MULTIAGENT_MODEL=openai:<served model name> LCB_RELEASE=release_v6
export ESCALATION_CLOUD_MAX_TOKENS=128000 ESCALATION_CLOUD_TIMEOUT=21600 ESCALATION_TIMEOUT=21600
export MULTIAGENT_ROLE_EFFORT=primary_plan=none,ideation=none,primary_manage=none,cutoff_summary=none,implement=none
export MULTIAGENT_ROLE_MAX_TOKENS=primary_plan=48000,ideation=48000,primary_manage=48000,cutoff_summary=16000,implement=48000
export MULTIAGENT_HARNESS_FIXES=1 MULTIAGENT_EXEC_CHECKS=1 MULTIAGENT_IMPLEMENT_SPLIT=1 BENCH_CHECKPOINT=runs/multi.jsonl
python escalation/run_bench.py --engine multiagent --only lcb --lcb 100 --parallel 2
```

## Why these settings

- **Thinking off for bookkeeping.** At reasoning effort high, GLM-5.3-Flash spent the whole 64,000-token allowance thinking on the planning call and returned no plan. Effort low and medium did the same. With thinking off for the four bookkeeping roles, the same request returned a plan with six tasks in 18 seconds. Workers keep full thinking.
- **Think, or write.** On hard problems GLM-5.3-Flash thinks until any budget runs out: a worker given 200,000 tokens on `abc398_g` used them all in 1.7 hours and wrote no code, as four workers had at 128,000. With thinking off and the method in the task, the same model wrote a complete program in 101 seconds and 2,426 tokens. The implement split lets the manager choose which a task needs.
- **Fewer in parallel.** With 33,000-token prompts, 2 requests in parallel produced the same total output as 1 (21.6 against 21.5 tokens per second): reading long prompts does not overlap. At 8 in parallel each request ran at about 10 tokens per second, at 2 at about 30, so results arrive about three times sooner for the same total work.
- **Timeouts longer than the slowest call.** `ESCALATION_CLOUD_TIMEOUT` must exceed the longest generation. At 8 parallel streams of about 10 tokens per second each, a 128,000-token call takes about 3.6 hours. A call that exceeds the timeout is abandoned without being cancelled and then retried, so the server keeps generating the abandoned answer and every retry gets slower.

## Grading and diagnosis

- `grade_interim.py`: grade a checkpoint at any point with LiveCodeBench's own grader.
- `judge_special.py`: AtCoder-style judges for the 7 problems in `lcb100_hardest_v6` that accept more than one correct output (`abc396_e`, `abc397_d`, `abc398_f`, `arc190_a`, `arc191_c`, `arc192_b`, and the construction problem `arc195_c`, whose statement asks to "show one example"). LiveCodeBench compares output text exactly and can reject correct answers on these. The paper's appendix already corrects `abc396_e`, `arc190_a`, `arc195_c` and `abc385_f` in its reported figures; `abc397_d`, `abc398_f`, `arc191_c` and `arc192_b` are additions. In the paper's stored runs, `abc397_d` fails every pass of Qwen3.8-27B (single and with the manager) and Fable 5 because of test 41, and passes under this judge. Its `abc397_d` test 41 expects `3 0` for N = 27, which the statement rules out: no pair of positive integers exists, so the answer is `-1`. `--selftest` checks each judge against the official answers.
- `analyze_harness.py`: per problem, every call (cut-off calls marked), the sample verdicts the manager saw and its latest decision.
- `calibrate_effort.py`: re-send a recorded call at different reasoning efforts, or with thinking off.
- `test_harness_fixes.py`: tests for `MULTIAGENT_HARNESS_FIXES`.
