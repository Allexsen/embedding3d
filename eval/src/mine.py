"""Case miner: pulls real questions (title = prompt, body = the asker's own
elaboration = intent evidence) from public APIs into cases/.

Sources:
  - StackExchange threads the community labeled "XY problem" (severe end)
  - General StackExchange questions across casual + technical sites
  - ELI5 via HuggingFace datasets-server (optional; skipped if the API is down)

Usage:  python src/mine.py [--xy N] [--general N] [--eli5 N]
Everything is KB-scale JSON API traffic — no dataset downloads.
Attribution (CC BY-SA) is stored per case, as StackExchange requires.
"""
import html
import random
import re
import sys
import time

import requests

from common import CASES_DIR, read_json, write_json

SE_API = "https://api.stackexchange.com/2.3"
XY_SITES = ["superuser", "serverfault", "unix", "ux"]
GENERAL_SITES = ["ux", "superuser", "workplace", "cooking", "diy", "gardening"]
HEADERS = {"User-Agent": "embedding3d-eval-miner (research; contact via repo)"}

TAG_RE = re.compile(r"<[^>]+>")
CODE_RE = re.compile(r"<(pre|code)[^>]*>.*?</\1>", re.S)


def body_to_text(body_html: str) -> str:
    no_code = CODE_RE.sub(" ", body_html)
    text = TAG_RE.sub(" ", no_code)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def acceptable(title: str, body_text: str, body_html: str) -> bool:
    if not (25 <= len(title) <= 140):
        return False
    if not (150 <= len(body_text) <= 1400):
        return False
    if "?" not in title and not re.match(
            r"(?i)^(how|why|what|when|which|where|can|should|is|are|do|does)\b", title):
        return False
    code_share = sum(len(m.group(0)) for m in CODE_RE.finditer(body_html)) / max(1, len(body_html))
    if code_share > 0.25:
        return False
    ascii_share = sum(c.isascii() for c in body_text) / len(body_text)
    return ascii_share > 0.95


def se_get(path: str, **params) -> dict:
    params.setdefault("pagesize", 100)
    resp = requests.get(f"{SE_API}/{path}", params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if data.get("backoff"):
        time.sleep(data["backoff"] + 1)
    time.sleep(0.3)  # be polite regardless
    return data


def fetch_bodies(site: str, ids: list[int]) -> list[dict]:
    out = []
    for i in range(0, len(ids), 90):
        chunk = ";".join(str(x) for x in ids[i:i + 90])
        data = se_get(f"questions/{chunk}", site=site, filter="withbody")
        out.extend(data.get("items", []))
    return out


def make_case(prefix: str, seq: int, source: str, tags: list, item: dict, site: str) -> dict:
    title = html.unescape(item["title"])
    return {
        "id": f"case-{prefix}{seq:03d}-{site}-{item['question_id']}",
        "source": source,
        "tags": tags,
        "prompt": title,
        "evidence": body_to_text(item["body"]),
        "attribution": {
            "url": item.get("link", ""),
            "author": html.unescape(item.get("owner", {}).get("display_name", "unknown")),
            "license": "CC BY-SA 4.0",
        },
    }


def mine_xy(target: int) -> list[dict]:
    """Threads where someone literally wrote 'XY problem' — community-labeled framing capture."""
    cases = []
    for site in XY_SITES:
        if len(cases) >= target:
            break
        found = se_get("search/excerpts", q='"XY problem"', site=site,
                       order="desc", sort="relevance")
        qids = list({it["question_id"] for it in found.get("items", [])})[:60]
        for item in fetch_bodies(site, qids):
            text = body_to_text(item["body"])
            if acceptable(html.unescape(item["title"]), text, item["body"]):
                cases.append(make_case("1", len(cases) + 1, "stackexchange-xy",
                                       ["xy-labeled"], item, site))
            if len(cases) >= target:
                break
    return cases


def mine_general(target: int) -> list[dict]:
    cases = []
    per_site = max(3, target // len(GENERAL_SITES) + 1)
    for site in GENERAL_SITES:
        if len(cases) >= target:
            break
        picked = 0
        data = se_get("questions", site=site, order="desc", sort="votes",
                      filter="withbody")
        for item in data.get("items", []):
            text = body_to_text(item["body"])
            if acceptable(html.unescape(item["title"]), text, item["body"]):
                cases.append(make_case("2", len(cases) + 1, f"stackexchange-{site}",
                                       ["general"], item, site))
                picked += 1
            if picked >= per_site or len(cases) >= target:
                break
    return cases


def mine_eli5(target: int) -> list[dict]:
    """Optional: HF datasets-server rows API; skipped gracefully when unavailable."""
    cases, offset = [], 0
    try:
        while len(cases) < target and offset < 2000:
            resp = requests.get(
                "https://datasets-server.huggingface.co/rows",
                params={"dataset": "eli5_category", "config": "default",
                        "split": "train", "offset": offset, "length": 100},
                headers=HEADERS, timeout=30)
            if resp.status_code != 200:
                print(f"  eli5: datasets-server unavailable ({resp.status_code}) — skipping")
                return cases
            for row in resp.json().get("rows", []):
                r = row["row"]
                title, body = r.get("title", ""), r.get("selftext", "") or ""
                if len(body) >= 150 and acceptable(title, body, body):
                    cases.append({
                        "id": f"case-3{len(cases) + 1:03d}-eli5-{r.get('q_id', offset)}",
                        "source": "eli5", "tags": ["casual"],
                        "prompt": re.sub(r"(?i)^eli5:?\s*", "", title).strip(),
                        "evidence": body,
                        "attribution": {"url": "", "author": "r/explainlikeimfive user",
                                        "license": "reddit content, research excerpt"},
                    })
                if len(cases) >= target:
                    break
            offset += 100
    except requests.RequestException as err:
        print(f"  eli5: {err} — skipping")
    return cases


def main() -> None:
    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    want_xy = int(args.get("--xy", 20))
    want_general = int(args.get("--general", 45))
    want_eli5 = int(args.get("--eli5", 25))

    existing = {read_json(p)["prompt"].lower() for p in CASES_DIR.glob("*.json")}
    written = []
    for batch in (mine_xy(want_xy), mine_general(want_general), mine_eli5(want_eli5)):
        for case in batch:
            if case["prompt"].lower() in existing:
                continue
            existing.add(case["prompt"].lower())
            write_json(CASES_DIR / f"{case['id']}.json", case)
            written.append(case["id"])

    print(f"\nWrote {len(written)} new cases.")
    by_src = {}
    for cid in written:
        src = cid.split("-")[1][0]
        by_src[src] = by_src.get(src, 0) + 1
    print("  by source prefix (1=xy, 2=se-general, 3=eli5):", by_src)
    sample = random.sample(written, min(8, len(written)))
    print("\nSpot-read sample:")
    for cid in sample:
        print(f"  cases/{cid}.json")


if __name__ == "__main__":
    main()
