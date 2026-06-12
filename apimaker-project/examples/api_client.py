"""Call the local FastAPI wrapper."""

from __future__ import annotations

import argparse

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["claude", "codex", "gemini"], default="claude")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    with httpx.Client(base_url=args.base_url, timeout=120.0) as client:
        session_response = client.post(
            "/sessions",
            json={"provider": args.provider},
        )
        session_response.raise_for_status()
        session = session_response.json()
        session_id = session["session_id"]

        try:
            first = client.post(
                f"/sessions/{session_id}/messages",
                json={"prompt": "내가 좋아하는 숫자는 42야. 알겠다고만 답해."},
            )
            first.raise_for_status()
            print(f"[1턴] {first.json()['response']}")

            second = client.post(
                f"/sessions/{session_id}/messages",
                json={"prompt": "그 숫자에 8을 더하면? 숫자만."},
            )
            second.raise_for_status()
            print(f"[2턴] {second.json()['response']}")
        finally:
            client.delete(f"/sessions/{session_id}")


if __name__ == "__main__":
    main()
