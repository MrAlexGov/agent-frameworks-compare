"""Прогон всех фреймворков × моделей по TASKS: точность, вызовы инструментов, задержка.

EVAL_BASE_URL/EVAL_API_KEY — OpenAI-совместимый endpoint. Запуск:
python run_bench.py --models google/gemini-3-flash-preview,openai/gpt-5.4-nano-2026-03-17 --repeats 2
"""
import argparse, asyncio, importlib, json, statistics, time, traceback

from common import TASKS, check

FRAMEWORKS = {"langgraph": "impl.langgraph_agent", "pydantic-ai": "impl.pydanticai_agent",
              "openai-agents-sdk": "impl.openai_agents_sdk"}


async def bench(fw, model, repeats):
    mod = importlib.import_module(FRAMEWORKS[fw])
    agent = mod.build(model)
    rows = []
    for rep in range(repeats):
        for t in TASKS:
            t0 = time.perf_counter()
            try:
                out, calls = await asyncio.wait_for(mod.run(agent, t[0]), 240)
                ok, err = check(t, out), None
                ans = out.model_dump()
            except Exception as e:
                ok, calls, err, ans = False, 0, f"{type(e).__name__}: {e}"[:300], None
            rows.append({"q": t[0], "rep": rep, "ok": ok, "tool_calls": calls, "enough_tools": calls >= t[3],
                         "latency_s": round(time.perf_counter() - t0, 2), "error": err, "answer": ans})
            print(fw, model.split("/")[-1], "ok" if ok else "FAIL", calls, rows[-1]["latency_s"], (err or "")[:120], flush=True)
    n = len(rows)
    return {"framework": fw, "model": model, "n": n, "accuracy": round(sum(r["ok"] for r in rows) / n, 3),
            "errors": sum(bool(r["error"]) for r in rows),
            "enough_tool_calls": round(sum(r["enough_tools"] for r in rows) / n, 3),
            "avg_tool_calls": round(sum(r["tool_calls"] for r in rows) / n, 2),
            "latency_p50_s": round(statistics.median(r["latency_s"] for r in rows), 2), "rows": rows}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="google/gemini-3-flash-preview")
    ap.add_argument("--frameworks", default=",".join(FRAMEWORKS))
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out", default="results/results.json")
    a = ap.parse_args()
    import os
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    res = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r["framework"], r["model"]) for r in res}
    for m in a.models.split(","):
        for fw in a.frameworks.split(","):
            if (fw, m) in done:
                continue
            try:
                res.append(await bench(fw, m, a.repeats))
            except Exception:
                traceback.print_exc()
            json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=2)
    for r in res:
        print({k: v for k, v in r.items() if k != "rows"})


if __name__ == "__main__":
    asyncio.run(main())
