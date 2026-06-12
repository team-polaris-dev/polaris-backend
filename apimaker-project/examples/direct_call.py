"""Call Claude or Codex directly from Python."""

from __future__ import annotations

import argparse
import asyncio

from apimaker import AgentOptions, AgentService


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["claude", "codex", "gemini"], default="claude")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    service = AgentService()
    session = await service.start_session(
        args.provider,
        AgentOptions(model=args.model),
    )
    try:
        first = await service.send_message(
            session.session_id,
            "내가 좋아하는 숫자는 42야. 알겠다고만 답해.",
        )
        print(f"[1턴] {first.response}")

        second = await service.send_message(
            session.session_id,
            "그 숫자에 8을 더하면? 숫자만.",
        )
        print(f"[2턴] {second.response}")
    finally:
        await service.close_all()


if __name__ == "__main__":
    asyncio.run(main())
