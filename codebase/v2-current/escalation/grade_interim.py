"""Interim grader: grade every problem in a GVS5H checkpoint with the harness's own LCB grader.

Run from codebase/v2-current (prints pass@1 and writes <checkpoint>.graded.json):
  LCB_RELEASE=release_v6 python escalation/grade_interim.py runs/ckpt.jsonl
"""
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "codebase", "livecodebench"))


def passed(res0):
    import numpy as np
    if isinstance(res0, (list, tuple)):
        return len(res0) > 0 and bool(np.all(np.array(res0) > 0))
    return bool(res0)


def main():
    from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

    ckpt = sys.argv[1]
    recs = {}
    with open(ckpt) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                recs[r["question_id"]] = r

    dataset = load_code_generation_dataset(release_version=os.environ.get("LCB_RELEASE", "release_v6"))
    by_id = {p.question_id: p for p in dataset}
    ids = [q for q in recs if q in by_id]
    samples = [by_id[q].get_evaluation_sample() for q in ids]
    generations = [[recs[q]["code"]] for q in ids]
    _, results, _ = codegen_metrics(samples, generations, k_list=[1], num_process_evaluate=4)

    rows = []
    for idx, q in enumerate(ids):
        s = recs[q].get("status") or {}
        rows.append({
            "question_id": q,
            "passed": passed(results[idx][0]),
            "finish": s.get("finish_reason"),
            "tokens": s.get("completion_tokens") or 0,
            "class": s.get("class"),
        })

    n = len(rows)
    npass = sum(r["passed"] for r in rows)
    stop = [r for r in rows if r["finish"] == "stop"]
    capped = [r for r in rows if r["finish"] == "length"]
    toks = sorted(r["tokens"] for r in rows)
    print(f"graded {n} problems: pass@1 = {npass}/{n} = {100.0 * npass / max(1, n):.1f}%")
    print(f"  finished naturally: {sum(r['passed'] for r in stop)}/{len(stop)} pass")
    print(f"  hit the cap:        {sum(r['passed'] for r in capped)}/{len(capped)} pass")
    print(f"  tokens/problem: median {toks[n // 2]:,} | mean {st.mean(toks):,.0f} | max {toks[-1]:,}")
    out = ckpt.replace(".jsonl", ".graded.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"  per-problem results: {out}")


if __name__ == "__main__":
    main()
