import json
import sys
import requests

API = "http://localhost:8000"
doc_id = sys.argv[1]
questions = json.load(open("tests/eval_questions.json", encoding="utf-8"))
score = {"baseline": 0, "hybrid": 0}
for item in questions:
    body = {"doc_id": doc_id, "question": item["q"]}
    for mode in ("baseline", "hybrid"):
        r = requests.post(f"{API}/ask/{mode}", json=body, timeout=300)
        if r.status_code != 200:
            print(f"[{mode:8}] ERROR {r.status_code}: {r.text[:300]}")
            continue
        ans = r.json()["answer"]
        ok = all(e.lower() in ans.lower() for e in item["expect"])
        score[mode] += ok
        print(f"[{mode:8}] {'PASS' if ok else 'FAIL'} ({item['type']}) {item['q']}")
n = len(questions)
print(f"\nBaseline: {score['baseline']}/{n}   Hybrid: {score['hybrid']}/{n}")
