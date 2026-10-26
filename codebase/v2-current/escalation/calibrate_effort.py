"""Calibrate reasoning effort for one harness call, by re-sending the exact recorded request.

Usage (from codebase/v2-current):
  python escalation/calibrate_effort.py <transcript.jsonl> <role> <effort> [<effort> ...]
Effort values are sent as reasoning_effort ("none", "low", "medium", "high"), except "nothink",
which sends chat_template_kwargs {"enable_thinking": false} instead. Each reply is parsed with the
harness's own section and bullet parsers, so the output shows whether the call produced a usable
plan or task list. Requests go to ESCALATION_OPENAI_BASE with OPENAI_API_KEY, for the model named
in MULTIAGENT_MODEL (the "openai:" prefix is dropped).
"""
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multiagent  # noqa: E402

URL = os.environ.get("ESCALATION_OPENAI_BASE", "http://127.0.0.1:8000/v1/chat/completions")
KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = os.environ.get("MULTIAGENT_MODEL", "").split(":", 1)[-1]


def call(request, effort, max_tokens):
    body = {"model": MODEL, "messages": request, "max_tokens": max_tokens,
            "temperature": 0.3, "stream": False}
    if effort == "nothink":
        body["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        body["reasoning_effort"] = effort
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=7200) as r:
        out = json.loads(r.read())
    msg = out["choices"][0]["message"]
    content = msg.get("content") or ""
    sec = multiagent._sections(content)
    tasks = multiagent._bullets(sec.get("TASKS", ""))
    return {"effort": effort, "seconds": round(time.time() - t0),
            "finish": out["choices"][0].get("finish_reason"),
            "completion_tokens": out["usage"].get("completion_tokens"),
            "reasoning_tokens": (out["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "content_chars": len(content), "has_plan": bool(sec.get("PLAN")), "n_tasks": len(tasks),
            "tasks": [t[:90] for t in tasks[:6]]}


def main():
    if len(sys.argv) < 4 or not MODEL:
        sys.exit(__doc__)
    transcript, role, efforts = sys.argv[1], sys.argv[2], sys.argv[3:]
    request = next(json.loads(l)["request"] for l in open(transcript) if json.loads(l).get("role") == role)
    max_tokens = int(os.environ.get("ESCALATION_CLOUD_MAX_TOKENS", "64000"))
    with cf.ThreadPoolExecutor(len(efforts)) as ex:
        for res in ex.map(lambda e: call(request, e, max_tokens), efforts):
            print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
