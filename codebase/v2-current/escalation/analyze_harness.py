"""Per-problem picture of a multiagent run: every call (cut-off calls marked), every sample-test
verdict the manager saw, and the manager's latest decision.

Run from codebase/v2-current while a run is in progress or after it:
  LCB_RELEASE=release_v6 python escalation/analyze_harness.py --checkpoint runs/multi.jsonl
Workspaces are read from MULTIAGENT_WS (default: runs/ws), as multiagent.py writes them.
"""
import argparse, glob, json, os, re, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "codebase", "livecodebench"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", default=os.path.join(ROOT, "runs", "ckpt-multi.jsonl"))
    args = ap.parse_args()
    from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
    probs = load_code_generation_dataset(release_version=os.environ.get("LCB_RELEASE", "release_v6"))
    ws_root = os.environ.get("MULTIAGENT_WS", os.path.join(ROOT, "runs", "ws"))
    done = set()
    if os.path.exists(args.checkpoint):
        done = {json.loads(l)["question_id"] for l in open(args.checkpoint) if l.strip()}
    now = time.time()
    for t in sorted(glob.glob(os.path.join(ws_root, "*", "transcript.jsonl"))):
        ws = os.path.dirname(t)
        task = open(os.path.join(ws, "task.md")).read()
        qid = next((p.question_id for p in probs if p.question_content.strip()[:150] in task), "?")
        recs = [json.loads(l) for l in open(t) if l.strip()]
        calls = [r for r in recs if not r.get("_meta")]
        roles = []
        for r in calls:
            tag = (r.get("role") or "?").replace("primary_", "").replace("worker:", "w")
            if r.get("finish_reason") == "length":
                tag += "(CUT)"
            roles.append(tag)
        print(f"{qid:10s} {'DONE' if qid in done else 'running':7s} {(now - recs[0]['t']) / 3600:4.1f} h | "
              f"calls {len(calls):2d}: {' '.join(roles)}")
        manages = [r for r in calls if r.get("role") == "primary_manage"]
        for r in manages[-1:]:
            body = " ".join(m.get("content", "") for m in r.get("request", []))
            for m in list(re.finditer(r"\[SAMPLE TESTS: [^\]]{0,80}", body))[-1:]:
                print(f"{'':12s}  manager saw: {m.group(0)[15:95]}")
            resp = r.get("response") or ""
            status = re.search(r"### STATUS\s*\n\s*(\S+)", resp)
            nxt = re.search(r"### NEXT\s*\n(.*?)(\n###|\Z)", resp, re.S)
            print(f"{'':12s}  last decision: {status.group(1) if status else '?'} | "
                  f"next: {(nxt.group(1).strip() if nxt else '-')[:170]!r}")


if __name__ == "__main__":
    main()
