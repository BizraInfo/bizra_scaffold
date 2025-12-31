from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional


def _parse_tz_offset(value: str) -> Optional[timezone]:
    if not value:
        return None
    try:
        hours = float(value)
    except ValueError:
        return None
    return timezone(timedelta(hours=hours))


def _now() -> datetime:
    tz = _parse_tz_offset(os.getenv("BIZRA_MIDNIGHT_TZ_OFFSET_HOURS", ""))
    if tz is None:
        return datetime.now()
    return datetime.now(tz)


def _next_midnight(now: datetime) -> datetime:
    return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)


def _sleep_until(target: datetime) -> None:
    while True:
        current = datetime.now(target.tzinfo) if target.tzinfo else datetime.now()
        remaining = (target - current).total_seconds()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 30))


def main() -> int:
    message = os.getenv("BIZRA_MIDNIGHT_MESSAGE", "Hello World")
    now = _now()
    target = _next_midnight(now)
    remaining = max((target - now).total_seconds(), 0.0)
    print("[MIDNIGHT] Waiting {:.1f}s until {}...".format(remaining, target.isoformat()))
    _sleep_until(target)
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
