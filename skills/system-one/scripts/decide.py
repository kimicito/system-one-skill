#!/usr/bin/env python3
"""System One decision CLI: выбор из кандидатов вместо генерации.

Режимы:
  embed  — смысловой скоринг через локальный OpenAI-совместимый embed-сервер
  matrix — взвешенная матрица критериев (json-спека, без внешних сервисов)

Пороги по умолчанию: CONF_MIN=0.55, GAP_MIN=0.05 — ниже/меньше = escalate.
"""
import argparse, json, math, sys, urllib.request

EMBED_URL = "http://127.0.0.1:7997/v1/embeddings"
CONF_MIN = 0.55
GAP_MIN = 0.05


def embed(texts):
    payload = json.dumps({"input": texts}).encode()
    req = urllib.request.Request(EMBED_URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read())
    return [d["embedding"] for d in data["data"]]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def cmd_embed(args):
    options = [o.strip() for o in args.options.split(";") if o.strip()]
    if not options:
        sys.exit("нет кандидатов")
    try:
        vectors = embed([args.query] + options)
    except Exception as e:
        sys.exit(f"embed-сервер недоступен ({e}); используй matrix-режим")
    q, cands = vectors[0], vectors[1:]
    scored = sorted(((cosine(q, c), o) for c, o in zip(cands, options)),
                    reverse=True)
    top, second = scored[0], scored[1] if len(scored) > 1 else (0.0, "")
    decision = {
        "mode": "embed",
        "query": args.query,
        "choice": top[1] if (top[0] >= CONF_MIN and top[0] - second[0] >= GAP_MIN) else None,
        "escalate": not (top[0] >= CONF_MIN and top[0] - second[0] >= GAP_MIN),
        "confidence": round(top[0], 4),
        "margin": round(top[0] - second[0], 4),
        "scores": [{"option": o, "score": round(s, 4)} for s, o in scored[:args.n]],
    }
    print(json.dumps(decision, ensure_ascii=False, indent=2))


def cmd_matrix(args):
    with open(args.spec, encoding="utf-8") as f:
        spec = json.load(f)
    criteria = spec["criteria"]          # [{name, weight}]
    options = spec["options"]            # [{name, scores: {crit: 0..1}}]
    results = []
    for opt in options:
        total = sum(c["weight"] * opt["scores"].get(c["name"], 0.0)
                    for c in criteria)
        results.append((total, opt["name"]))
    results.sort(reverse=True)
    wsum = sum(c["weight"] for c in criteria)
    top, second = results[0], results[1] if len(results) > 1 else (0.0, "")
    decision = {
        "mode": "matrix",
        "choice": top[1] if top[0] / wsum >= CONF_MIN and top[0] - second[0] >= GAP_MIN else None,
        "escalate": not (top[0] / wsum >= CONF_MIN and top[0] - second[0] >= GAP_MIN),
        "confidence": round(top[0] / wsum, 4),
        "margin": round((top[0] - second[0]) / wsum, 4),
        "scores": [{"option": n, "score": round(s / wsum, 4)} for s, n in results],
    }
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pe = sub.add_parser("embed", help="смысловой скоринг через embed-сервер")
    pe.add_argument("--query", required=True)
    pe.add_argument("--options", required=True, help="кандидаты через ;")
    pe.add_argument("-n", type=int, default=5)
    pe.set_defaults(fn=cmd_embed)
    pm = sub.add_parser("matrix", help="матрица критериев")
    pm.add_argument("--spec", required=True, help="json: {criteria, options}")
    pm.set_defaults(fn=cmd_matrix)
    args = p.parse_args()
    args.fn(args)
