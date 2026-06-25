#!/usr/bin/env python3
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request


PROMPT = (
    "海滩上有一堆桃子，五只猴子来分。第一只猴子把这堆桃子平均分为五份，多了一个，"
    "这只猴子把多的一个扔入海中，拿走了一份。第二只猴子把剩下的桃子又平均分成五份，"
    "又多了一个，它同样把多的一个扔入海中，拿走了一份，第三、第四、第五只猴子都是这样做的，"
    "问海滩上原来最少有多少个桃子？"
)


def http_get(url: str, timeout: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def http_post_json(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_spec_metrics(text: str) -> dict | None:
    metrics = {
        "drafts": 0,
        "draft_tokens": 0,
        "accepted_tokens": 0,
        "accepted_per_pos": {},
    }
    found = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith("vllm:spec_decode"):
            continue

        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        metric_name = parts[0].split("{", 1)[0]
        if not metric_name.endswith("_total"):
            continue

        try:
            value = int(float(parts[1]))
        except ValueError:
            continue

        found = True
        if "num_drafts" in metric_name:
            metrics["drafts"] += value
        elif "num_draft_tokens" in metric_name:
            metrics["draft_tokens"] += value
        elif "num_accepted_tokens_per_pos" in metric_name:
            match = re.search(r'position="(\d+)"', line)
            if match:
                pos = int(match.group(1))
                metrics["accepted_per_pos"][pos] = (
                    metrics["accepted_per_pos"].get(pos, 0) + value
                )
        elif "num_accepted_tokens" in metric_name:
            metrics["accepted_tokens"] += value

    return metrics if found else None


def fetch_spec_metrics(base_url: str, timeout: float) -> dict | None:
    try:
        return parse_spec_metrics(http_get(f"{base_url}/metrics", timeout))
    except (urllib.error.URLError, TimeoutError):
        return None


def diff_metrics(after: dict, before: dict) -> dict:
    per_pos = {}
    positions = set(before["accepted_per_pos"]) | set(after["accepted_per_pos"])
    for pos in positions:
        per_pos[pos] = (
            after["accepted_per_pos"].get(pos, 0)
            - before["accepted_per_pos"].get(pos, 0)
        )
    return {
        "drafts": after["drafts"] - before["drafts"],
        "draft_tokens": after["draft_tokens"] - before["draft_tokens"],
        "accepted_tokens": after["accepted_tokens"] - before["accepted_tokens"],
        "accepted_per_pos": per_pos,
    }


def run_chat_once(args: argparse.Namespace) -> dict:
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": False,
    }
    if args.top_p is not None:
        payload["top_p"] = args.top_p

    return http_post_json(
        f"{args.base_url}/v1/chat/completions",
        payload,
        timeout=args.timeout,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send the peach-monkey prompt through chat API and measure MTP acceptance rate."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8101")
    parser.add_argument("--model", default="pangu")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--n", type=int, default=1, help="number of chat requests")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--sleep-after",
        type=float,
        default=1.0,
        help="seconds to wait after requests before scraping /metrics",
    )
    args = parser.parse_args()
    args.base_url = args.base_url.rstrip("/")

    before = fetch_spec_metrics(args.base_url, args.timeout)
    if before is None:
        print("WARNING: /metrics does not expose vllm:spec_decode metrics before the request.")

    first_text = None
    total_completion_tokens = 0
    started = time.time()
    for idx in range(args.n):
        try:
            resp = run_chat_once(args)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            print(f"HTTP {exc.code} from chat API:\n{body}", file=sys.stderr)
            return 1
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"Failed to call chat API: {exc}", file=sys.stderr)
            return 1

        choice = resp.get("choices", [{}])[0]
        message = choice.get("message", {})
        if first_text is None:
            first_text = message.get("content", "")
        usage = resp.get("usage", {})
        total_completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        print(f"request {idx + 1}/{args.n}: finish_reason={choice.get('finish_reason')}, usage={usage}")

    if args.sleep_after > 0:
        time.sleep(args.sleep_after)

    after = fetch_spec_metrics(args.base_url, args.timeout)
    elapsed = time.time() - started

    print("\n=== first response ===")
    print(first_text or "")

    print("\n=== summary ===")
    print(f"requests: {args.n}")
    print(f"elapsed_sec: {elapsed:.2f}")
    print(f"completion_tokens_from_api: {total_completion_tokens}")

    if before is None or after is None:
        print("\nCould not compute acceptance rate from /metrics.")
        print("Please check the server log for lines like:")
        print("SpecDecoding metrics: ... Accepted: X tokens, Drafted: Y tokens, ... Avg Draft acceptance rate: Z%")
        return 2

    delta = diff_metrics(after, before)
    accepted = delta["accepted_tokens"]
    drafted = delta["draft_tokens"]
    drafts = delta["drafts"]

    print(f"spec_decode_num_drafts: {drafts}")
    print(f"spec_decode_accepted_tokens: {accepted}")
    print(f"spec_decode_draft_tokens: {drafted}")
    if drafted > 0:
        print(f"overall_acceptance_rate: {accepted / drafted:.6f} ({accepted / drafted * 100:.2f}%)")
    else:
        print("overall_acceptance_rate: N/A (no draft tokens recorded)")

    if drafts > 0 and delta["accepted_per_pos"]:
        rates = []
        for pos in sorted(delta["accepted_per_pos"]):
            rates.append(f"pos{pos}={delta['accepted_per_pos'][pos] / drafts:.6f}")
        print("per_position_acceptance_rate: " + ", ".join(rates))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
