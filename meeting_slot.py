"""meeting-slot: multi-turn tool-using meeting scheduler with a deterministic grader.

The model must find the earliest UTC start that fits every attendee. Busy
intervals, working hours, and timezones are hidden behind tools — stuffing
the calendar into the prompt would erase the tool-use point.

Final answer is a UTC ISO-8601 start inside ``<answer>`` tags, e.g.
``2024-03-11T15:00:00Z``. If no slot exists, ``NONE``.

Gold is interval intersection in UTC via ``zoneinfo`` — the same solver
builds the dataset. Reward is exact match plus a small XML-format bonus
and partial credit for a valid-but-not-earliest start.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Literal
from zoneinfo import ZoneInfo

Difficulty = Literal["easy", "medium", "hard"]

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
HM_RE = re.compile(r"^(\d{1,2}):(\d{2})$")
# Accept Z / +00:00 / missing seconds. Minutes required.
UTC_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::(\d{2}))?(?:Z|\+00:00| UTC)?$",
    re.IGNORECASE,
)

DIFFICULTIES: tuple[Difficulty, ...] = ("easy", "medium", "hard")
DIFFICULTY_WEIGHTS = {"easy": 0.40, "medium": 0.35, "hard": 0.25}

WEEKDAY_NAMES = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)

PEOPLE: tuple[tuple[str, str], ...] = (
    ("Ava", "America/New_York"),
    ("Ben", "America/Los_Angeles"),
    ("Chen", "Asia/Shanghai"),
    ("Diya", "Asia/Kolkata"),
    ("Elena", "Europe/Berlin"),
    ("Farid", "Europe/London"),
    ("Gita", "Asia/Tokyo"),
    ("Hugo", "Australia/Sydney"),
    ("Ines", "America/Sao_Paulo"),
    ("Jules", "UTC"),
)

CLUSTERS: dict[str, tuple[tuple[str, str], ...]] = {
    "us": (
        ("Ava", "America/New_York"),
        ("Ben", "America/Los_Angeles"),
        ("Ines", "America/Sao_Paulo"),
    ),
    "eu": (
        ("Elena", "Europe/Berlin"),
        ("Farid", "Europe/London"),
        ("Jules", "UTC"),
    ),
    "asia": (
        ("Chen", "Asia/Shanghai"),
        ("Gita", "Asia/Tokyo"),
        ("Diya", "Asia/Kolkata"),
    ),
    "atlantic": (
        ("Ava", "America/New_York"),
        ("Farid", "Europe/London"),
        ("Elena", "Europe/Berlin"),
        ("Jules", "UTC"),
    ),
}

NONE_PAIRS: tuple[tuple[tuple[str, str], tuple[str, str]], ...] = (
    (("Ava", "America/New_York"), ("Diya", "Asia/Kolkata")),
    (("Ava", "America/New_York"), ("Gita", "Asia/Tokyo")),
    (("Hugo", "Australia/Sydney"), ("Ava", "America/New_York")),
    (("Ben", "America/Los_Angeles"), ("Gita", "Asia/Tokyo")),
    (("Hugo", "Australia/Sydney"), ("Ben", "America/Los_Angeles")),
)

HOURS_CHOICES: tuple[tuple[str, str], ...] = (
    ("09:00", "17:00"),
    ("08:00", "16:00"),
    ("10:00", "18:00"),
    ("09:00", "18:00"),
)

# Anchors that sit on or next to DST transitions (always a Friday).
DST_FRIDAYS: tuple[str, ...] = (
    "2024-03-08",  # US spring-forward Sunday Mar 10
    "2024-03-29",  # EU spring-forward Sunday Mar 31
    "2024-10-25",  # EU fall-back Sunday Oct 27
    "2024-11-01",  # US fall-back Sunday Nov 3
)

SYSTEM_PROMPT = """You are scheduling a meeting across timezones.

The prompt gives the duration and the UTC search window. Everything else is
hidden: attendee names, timezones, working hours, and busy calendars. You
must call tools to inspect them.

Tools:
- list_attendees() — names
- get_timezone(name) — IANA timezone
- get_working_hours(name) — local hours and weekdays (0=Monday … 6=Sunday)
- get_busy(name, date) — local busy intervals for that attendee's local date (YYYY-MM-DD)

Rules:
- A valid start is the earliest UTC instant such that the whole [start, start+duration)
  sits inside every attendee's working hours and overlaps no busy interval.
- Intervals are half-open [start, end). A busy block ending at 11:00 means 11:00 is free.
- Working hours do not wrap past local midnight. Weekends count only when listed in weekdays.
- The start must fall inside the search window (UTC dates, inclusive).
- Do not guess calendars or offsets. Query tools.
- Put the final answer alone inside <answer>...</answer>.
- Format: YYYY-MM-DDTHH:MM:SSZ (zero-padded, Z suffix). If no slot exists, NONE.

Respond in this format when you have the answer (no tool calls on that turn):
<answer>
YOUR_ANSWER
</answer>
"""


def _att(
    name: str,
    tz: str,
    start: str = "09:00",
    end: str = "17:00",
    weekdays: list[int] | None = None,
    busy: dict[str, list[dict[str, str]]] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "timezone": tz,
        "working_hours": {
            "start": start,
            "end": end,
            "weekdays": list(weekdays if weekdays is not None else [0, 1, 2, 3, 4]),
        },
        "busy": busy or {},
    }


# Curated eval rows. Gold is filled by the solver in example_from_spec —
# the strings below are the worlds, not the answers.
EDGE_CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "same_tz_empty_busy",
        "duration_minutes": 30,
        "window_start": "2024-03-11",
        "window_days": 1,
        "difficulty": "easy",
        "world": {
            "attendees": [
                _att("Ava", "America/New_York"),
                _att("Ben", "America/New_York"),
            ]
        },
    },
    {
        "id": "overlapping_busy",
        "duration_minutes": 60,
        "window_start": "2024-03-11",
        "window_days": 1,
        "difficulty": "medium",
        "world": {
            "attendees": [
                _att(
                    "Ava",
                    "America/New_York",
                    busy={"2024-03-11": [{"start": "09:00", "end": "12:00"}]},
                ),
                _att(
                    "Ben",
                    "America/New_York",
                    busy={"2024-03-11": [{"start": "11:00", "end": "14:00"}]},
                ),
            ]
        },
    },
    {
        "id": "no_overlap_ny_kolkata",
        "duration_minutes": 30,
        "window_start": "2024-03-11",
        "window_days": 1,
        "difficulty": "hard",
        "world": {
            "attendees": [
                _att("Ava", "America/New_York"),
                _att("Diya", "Asia/Kolkata"),
            ]
        },
    },
    {
        "id": "weekend_skip_dst",
        "duration_minutes": 30,
        "window_start": "2024-03-08",
        "window_days": 4,
        "difficulty": "hard",
        "world": {
            "attendees": [
                _att(
                    "Ava",
                    "America/New_York",
                    busy={"2024-03-08": [{"start": "09:00", "end": "17:00"}]},
                ),
                _att(
                    "Farid",
                    "Europe/London",
                    busy={"2024-03-08": [{"start": "09:00", "end": "17:00"}]},
                ),
            ]
        },
    },
    {
        "id": "dst_us_spring_monday",
        "duration_minutes": 30,
        "window_start": "2024-03-11",
        "window_days": 1,
        "difficulty": "hard",
        "world": {
            "attendees": [
                _att("Ava", "America/New_York"),
                _att("Farid", "Europe/London"),
            ]
        },
    },
    {
        "id": "inclusive_busy_utc",
        "duration_minutes": 30,
        "window_start": "2024-03-11",
        "window_days": 1,
        "difficulty": "medium",
        "world": {
            "attendees": [
                _att(
                    "Jules",
                    "UTC",
                    busy={"2024-03-11": [{"start": "09:00", "end": "11:00"}]},
                ),
                _att(
                    "Kit",
                    "UTC",
                    busy={"2024-03-11": [{"start": "09:00", "end": "11:00"}]},
                ),
            ]
        },
    },
    {
        "id": "gap_30min",
        "duration_minutes": 30,
        "window_start": "2024-03-11",
        "window_days": 1,
        "difficulty": "medium",
        "world": {
            "attendees": [
                _att(
                    "Jules",
                    "UTC",
                    busy={
                        "2024-03-11": [
                            {"start": "09:00", "end": "11:00"},
                            {"start": "11:30", "end": "16:00"},
                        ]
                    },
                ),
                _att(
                    "Kit",
                    "UTC",
                    busy={
                        "2024-03-11": [
                            {"start": "09:00", "end": "11:00"},
                            {"start": "11:30", "end": "16:00"},
                        ]
                    },
                ),
            ]
        },
    },
    {
        "id": "gap_60min",
        "duration_minutes": 60,
        "window_start": "2024-03-11",
        "window_days": 1,
        "difficulty": "medium",
        "world": {
            "attendees": [
                _att(
                    "Jules",
                    "UTC",
                    busy={
                        "2024-03-11": [
                            {"start": "09:00", "end": "11:00"},
                            {"start": "11:30", "end": "16:00"},
                        ]
                    },
                ),
                _att(
                    "Kit",
                    "UTC",
                    busy={
                        "2024-03-11": [
                            {"start": "09:00", "end": "11:00"},
                            {"start": "11:30", "end": "16:00"},
                        ]
                    },
                ),
            ]
        },
    },
    {
        "id": "four_person_summer",
        "duration_minutes": 30,
        "window_start": "2024-06-10",
        "window_days": 1,
        "difficulty": "hard",
        "world": {
            "attendees": [
                _att("Ava", "America/New_York"),
                _att("Farid", "Europe/London"),
                _att("Elena", "Europe/Berlin"),
                _att("Jules", "UTC"),
            ]
        },
    },
    {
        "id": "sydney_ny_none",
        "duration_minutes": 30,
        "window_start": "2024-06-10",
        "window_days": 1,
        "difficulty": "hard",
        "world": {
            "attendees": [
                _att("Hugo", "Australia/Sydney"),
                _att("Ava", "America/New_York"),
            ]
        },
    },
    {
        "id": "shanghai_tokyo",
        "duration_minutes": 30,
        "window_start": "2024-03-11",
        "window_days": 1,
        "difficulty": "medium",
        "world": {
            "attendees": [
                _att("Chen", "Asia/Shanghai", start="09:00", end="18:00"),
                _att("Gita", "Asia/Tokyo", start="09:00", end="18:00"),
            ]
        },
    },
    {
        "id": "last_friday_slot",
        "duration_minutes": 30,
        "window_start": "2024-03-08",
        "window_days": 4,
        "difficulty": "hard",
        "world": {
            "attendees": [
                _att("Ava", "America/New_York"),
                _att(
                    "Farid",
                    "Europe/London",
                    busy={"2024-03-08": [{"start": "09:00", "end": "16:30"}]},
                ),
            ]
        },
    },
    {
        "id": "dst_eu_monday",
        "duration_minutes": 30,
        "window_start": "2024-03-29",
        "window_days": 4,
        "difficulty": "hard",
        "world": {
            "attendees": [
                _att(
                    "Farid",
                    "Europe/London",
                    busy={"2024-03-29": [{"start": "09:00", "end": "17:00"}]},
                ),
                _att(
                    "Elena",
                    "Europe/Berlin",
                    busy={"2024-03-29": [{"start": "09:00", "end": "17:00"}]},
                ),
            ]
        },
    },
    {
        "id": "three_person_busy",
        "duration_minutes": 30,
        "window_start": "2024-06-10",
        "window_days": 1,
        "difficulty": "hard",
        "world": {
            "attendees": [
                _att("Ava", "America/New_York"),
                _att("Farid", "Europe/London"),
                _att(
                    "Jules",
                    "UTC",
                    busy={"2024-06-10": [{"start": "09:00", "end": "14:00"}]},
                ),
            ]
        },
    },
    {
        "id": "no_slot_fully_booked",
        "duration_minutes": 30,
        "window_start": "2024-03-11",
        "window_days": 1,
        "difficulty": "easy",
        "world": {
            "attendees": [
                _att(
                    "Ava",
                    "America/New_York",
                    busy={"2024-03-11": [{"start": "09:00", "end": "17:00"}]},
                ),
                _att(
                    "Ben",
                    "America/New_York",
                    busy={"2024-03-11": [{"start": "09:00", "end": "17:00"}]},
                ),
            ]
        },
    },
)


def decode_world(world: Any) -> dict[str, Any]:
    """HF datasets unifies nested dict keys and fills holes with None.

    Dataset rows store ``info["world"]`` as a JSON string so the schema stays
    stable. In-memory tests still pass a dict.
    """
    if isinstance(world, str):
        world = json.loads(world)
    if not isinstance(world, dict):
        return {"attendees": []}
    return world


def busy_blocks(attendee: dict[str, Any], day: str) -> list[dict[str, str]]:
    raw = (attendee.get("busy") or {}).get(day)
    return list(raw) if isinstance(raw, list) else []


def rows_for_dataset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        cloned = dict(row)
        info = dict(cloned["info"])
        world = info.get("world")
        if not isinstance(world, str):
            info["world"] = json.dumps(world, sort_keys=True)
        cloned["info"] = info
        out.append(cloned)
    return out


def parse_iso_date(value: str) -> date:
    match = ISO_DATE_RE.match(value.strip())
    if not match:
        raise ValueError(f"not an ISO date: {value!r}")
    year, month, day = (int(part) for part in match.groups())
    return date(year, month, day)


def parse_hm(value: str) -> tuple[int, int]:
    match = HM_RE.match(value.strip())
    if not match:
        raise ValueError(f"not HH:MM: {value!r}")
    hour, minute = int(match.group(1)), int(match.group(2))
    if hour > 24 or minute > 59 or hour == 24 and minute != 0:
        raise ValueError(f"not HH:MM: {value!r}")
    return hour, minute


def hm_to_min(value: str) -> int:
    hour, minute = parse_hm(value)
    return hour * 60 + minute


def min_to_hm(value: int) -> str:
    hour, minute = divmod(int(value), 60)
    return f"{hour:02d}:{minute:02d}"


def format_utc(dt: datetime) -> str:
    utc = dt.astimezone(timezone.utc).replace(microsecond=0, tzinfo=timezone.utc)
    return utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(value: str) -> datetime | None:
    text = value.strip()
    if not text:
        return None
    match = UTC_RE.match(text)
    if not match:
        return None
    year, month, day, hour, minute = (int(match.group(i)) for i in range(1, 6))
    second = int(match.group(6) or 0)
    try:
        return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    except ValueError:
        return None


def window_bounds(window_start: str, window_days: int) -> tuple[datetime, datetime]:
    start = datetime.combine(parse_iso_date(window_start), datetime.min.time(), tzinfo=timezone.utc)
    return start, start + timedelta(days=int(window_days))


def window_end_date(window_start: str, window_days: int) -> str:
    return (parse_iso_date(window_start) + timedelta(days=int(window_days) - 1)).isoformat()


def _iter_days(start: date, end: date) -> Iterable[date]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def iter_local_minutes_as_utc(
    tz_name: str, day: date, start_min: int, end_min: int
) -> list[datetime]:
    """Convert half-open local [start_min, end_min) on ``day`` to UTC instants.

    Both DST folds are included; nonexistent spring-forward minutes are skipped.
    """
    tz = ZoneInfo(tz_name)
    out: list[datetime] = []
    seen: set[int] = set()
    for m in range(int(start_min), int(end_min)):
        hour, minute = divmod(m, 60)
        if hour > 23:
            break
        naive = datetime(day.year, day.month, day.day, hour, minute, 0)
        for fold in (0, 1):
            aware = naive.replace(tzinfo=tz, fold=fold)
            back = aware.astimezone(tz)
            if back.replace(tzinfo=None) != naive:
                continue
            utc = aware.astimezone(timezone.utc).replace(microsecond=0)
            key = int(utc.timestamp())
            if key not in seen:
                seen.add(key)
                out.append(utc)
    out.sort()
    return out


def merge_consecutive_datetimes(times: list[datetime]) -> list[tuple[datetime, datetime]]:
    if not times:
        return []
    unique = sorted({t.replace(microsecond=0) for t in times})
    runs: list[tuple[datetime, datetime]] = []
    run_s = prev = unique[0]
    for t in unique[1:]:
        if t - prev == timedelta(minutes=1):
            prev = t
            continue
        runs.append((run_s, prev + timedelta(minutes=1)))
        run_s = prev = t
    runs.append((run_s, prev + timedelta(minutes=1)))
    return runs


def subtract_busy(
    work: tuple[int, int], busy: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    segments = [work]
    for bs, be in sorted(busy):
        nxt: list[tuple[int, int]] = []
        for s, e in segments:
            if be <= s or bs >= e:
                nxt.append((s, e))
                continue
            if s < bs:
                nxt.append((s, min(bs, e)))
            if be < e:
                nxt.append((max(be, s), e))
        segments = [(s, e) for s, e in nxt if e > s]
    return segments


def merge_intervals(
    intervals: list[tuple[datetime, datetime]],
) -> list[tuple[datetime, datetime]]:
    if not intervals:
        return []
    ordered = sorted((s, e) for s, e in intervals if e > s)
    if not ordered:
        return []
    out = [ordered[0]]
    for s, e in ordered[1:]:
        last_s, last_e = out[-1]
        if s <= last_e:
            out[-1] = (last_s, max(last_e, e))
        else:
            out.append((s, e))
    return out


def intersect_two(
    a: list[tuple[datetime, datetime]], b: list[tuple[datetime, datetime]]
) -> list[tuple[datetime, datetime]]:
    i = j = 0
    a = sorted(a)
    b = sorted(b)
    out: list[tuple[datetime, datetime]] = []
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if s < e:
            out.append((s, e))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def intersect_all(
    groups: list[list[tuple[datetime, datetime]]],
) -> list[tuple[datetime, datetime]]:
    if not groups:
        return []
    acc = merge_intervals(groups[0])
    for other in groups[1:]:
        acc = intersect_two(acc, merge_intervals(other))
        if not acc:
            return []
    return acc


# ---------------------------------------------------------------------------
# Gold solver
# ---------------------------------------------------------------------------


def attendee_free_utc(
    attendee: dict[str, Any], window_0: datetime, window_1: datetime
) -> list[tuple[datetime, datetime]]:
    hours = attendee["working_hours"]
    weekdays = set(int(d) for d in hours["weekdays"])
    work = (hm_to_min(hours["start"]), hm_to_min(hours["end"]))
    tz_name = attendee["timezone"]

    pad_start = (window_0 - timedelta(days=2)).date()
    pad_end = (window_1 + timedelta(days=2)).date()
    free: list[tuple[datetime, datetime]] = []
    for day in _iter_days(pad_start, pad_end):
        if day.weekday() not in weekdays:
            continue
        busy_local = []
        for block in busy_blocks(attendee, day.isoformat()):
            busy_local.append((hm_to_min(block["start"]), hm_to_min(block["end"])))
        for fs, fe in subtract_busy(work, busy_local):
            utc_minutes = iter_local_minutes_as_utc(tz_name, day, fs, fe)
            free.extend(merge_consecutive_datetimes(utc_minutes))
    return merge_intervals(free)


def gold_earliest(
    world: dict[str, Any],
    duration_minutes: int,
    window_start: str,
    window_days: int,
) -> str:
    """Earliest valid UTC start, or ``NONE``."""
    window_0, window_1 = window_bounds(window_start, window_days)
    duration = timedelta(minutes=int(duration_minutes))
    groups = [
        attendee_free_utc(att, window_0, window_1) for att in world["attendees"]
    ]
    common = intersect_all(groups)
    best: datetime | None = None
    for s, e in common:
        start = max(s, window_0)
        if start >= window_1:
            continue
        if start + duration <= e:
            if best is None or start < best:
                best = start
    return format_utc(best) if best is not None else "NONE"


def gold_answer(example: dict[str, Any]) -> str:
    info = example.get("info", example)
    return gold_earliest(
        decode_world(info["world"]),
        int(info["duration_minutes"]),
        info["window_start"],
        int(info["window_days"]),
    )


def is_valid_slot(
    world: dict[str, Any],
    start_str: str,
    duration_minutes: int,
    window_start: str,
    window_days: int,
) -> bool:
    """True iff ``start_str`` is a conflict-free in-hours start inside the window."""
    start = parse_utc(start_str)
    if start is None:
        return False
    window_0, window_1 = window_bounds(window_start, window_days)
    if not (window_0 <= start < window_1):
        return False
    end = start + timedelta(minutes=int(duration_minutes))
    for att in world["attendees"]:
        tz = ZoneInfo(att["timezone"])
        local_s = start.astimezone(tz)
        local_e = end.astimezone(tz)
        if local_e <= local_s:
            return False
        if local_s.date() != (local_e - timedelta(microseconds=1)).date():
            return False
        hours = att["working_hours"]
        if local_s.weekday() not in set(int(d) for d in hours["weekdays"]):
            return False
        if local_s.second or local_s.microsecond or local_e.second or local_e.microsecond:
            return False
        start_min = local_s.hour * 60 + local_s.minute
        end_min = local_e.hour * 60 + local_e.minute
        work_s, work_e = hm_to_min(hours["start"]), hm_to_min(hours["end"])
        if start_min < work_s or end_min > work_e:
            return False
        day_key = local_s.date().isoformat()
        for block in busy_blocks(att, day_key):
            bs, be = hm_to_min(block["start"]), hm_to_min(block["end"])
            if start_min < be and end_min > bs:
                return False
    return True


def naive_earliest(
    world: dict[str, Any],
    duration_minutes: int,
    window_start: str,
    window_days: int,
) -> str:
    """Discrimination policy: treat local clocks as UTC, busy end inclusive."""
    window_0, window_1 = window_bounds(window_start, window_days)
    duration = timedelta(minutes=int(duration_minutes))
    groups: list[list[tuple[datetime, datetime]]] = []
    for att in world["attendees"]:
        hours = att["working_hours"]
        weekdays = set(int(d) for d in hours["weekdays"])
        work = (hm_to_min(hours["start"]), hm_to_min(hours["end"]))
        free: list[tuple[datetime, datetime]] = []
        pad_start = (window_0 - timedelta(days=2)).date()
        pad_end = (window_1 + timedelta(days=2)).date()
        for day in _iter_days(pad_start, pad_end):
            if day.weekday() not in weekdays:
                continue
            # Inclusive busy: convert [s, e) to [s, e+1min) so the end minute is blocked.
            busy_local = []
            for block in busy_blocks(att, day.isoformat()):
                busy_local.append(
                    (hm_to_min(block["start"]), hm_to_min(block["end"]) + 1)
                )
            for fs, fe in subtract_busy(work, busy_local):
                s = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc) + timedelta(
                    minutes=fs
                )
                e = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc) + timedelta(
                    minutes=fe
                )
                free.append((s, e))
        groups.append(merge_intervals(free))
    common = intersect_all(groups)
    best: datetime | None = None
    for s, e in common:
        start = max(s, window_0)
        if start >= window_1:
            continue
        if start + duration <= e:
            if best is None or start < best:
                best = start
    return format_utc(best) if best is not None else "NONE"


# ---------------------------------------------------------------------------
# Tools (hidden ``world`` injected by StatefulToolEnv.update_tool_args)
# ---------------------------------------------------------------------------


def _attendee_by_name(world: dict[str, Any] | None, name: str) -> dict[str, Any] | None:
    if not world:
        return None
    for att in world.get("attendees") or []:
        if att.get("name") == name:
            return att
    lowered = name.strip().lower()
    for att in world.get("attendees") or []:
        if str(att.get("name", "")).lower() == lowered:
            return att
    return None


def list_attendees(world: dict) -> str:
    """List meeting attendee names.

    Returns a JSON array of names, e.g. ["Ava", "Ben"].
    """
    names = [att["name"] for att in (world or {}).get("attendees") or []]
    return json.dumps(names)


def get_timezone(name: str, world: dict) -> str:
    """Return the IANA timezone of an attendee, e.g. America/New_York."""
    att = _attendee_by_name(world, name)
    if att is None:
        return f"error: unknown attendee: {name}"
    return str(att["timezone"])


def get_working_hours(name: str, world: dict) -> str:
    """Return local working hours for an attendee.

    JSON object with start/end as HH:MM and weekdays as ints (0=Monday … 6=Sunday).
    """
    att = _attendee_by_name(world, name)
    if att is None:
        return f"error: unknown attendee: {name}"
    hours = att["working_hours"]
    weekdays = [int(d) for d in hours["weekdays"]]
    payload = {
        "start": hours["start"],
        "end": hours["end"],
        "weekdays": weekdays,
        "weekday_names": [WEEKDAY_NAMES[d] for d in weekdays],
    }
    return json.dumps(payload)


def get_busy(name: str, date: str, world: dict) -> str:
    """Return local busy intervals for an attendee on a local calendar date.

    `date` is YYYY-MM-DD in the attendee's local timezone, not UTC.
    Intervals are half-open [start, end) with HH:MM local times.
    """
    att = _attendee_by_name(world, name)
    if att is None:
        return f"error: unknown attendee: {name}"
    try:
        parse_iso_date(date)
    except ValueError:
        return f"error: invalid date: {date}"
    blocks = busy_blocks(att, date)
    return json.dumps(blocks)


def padded_local_dates(window_start: str, window_days: int) -> list[str]:
    start = parse_iso_date(window_start) - timedelta(days=1)
    end = parse_iso_date(window_start) + timedelta(days=int(window_days))
    return [d.isoformat() for d in _iter_days(start, end)]


def solve_from_tools(
    world: dict[str, Any],
    duration_minutes: int,
    window_start: str,
    window_days: int,
) -> str:
    """Gold policy: query tools, rebuild the world, run the sweep-line solver."""
    names = json.loads(list_attendees(world=world))
    rebuilt_attendees: list[dict[str, Any]] = []
    for name in names:
        tz = get_timezone(name=name, world=world)
        hours = json.loads(get_working_hours(name=name, world=world))
        busy: dict[str, list[dict[str, str]]] = {}
        for day in padded_local_dates(window_start, window_days):
            blocks = json.loads(get_busy(name=name, date=day, world=world))
            if blocks:
                busy[day] = blocks
        rebuilt_attendees.append(
            {
                "name": name,
                "timezone": tz,
                "working_hours": {
                    "start": hours["start"],
                    "end": hours["end"],
                    "weekdays": hours["weekdays"],
                },
                "busy": busy,
            }
        )
    return gold_earliest(
        {"attendees": rebuilt_attendees},
        duration_minutes,
        window_start,
        window_days,
    )


def gold_completion(answer: str) -> str:
    return f"<answer>{answer}</answer>"


# ---------------------------------------------------------------------------
# Parser + rubric (stdlib — tests run without verifiers installed)
# ---------------------------------------------------------------------------


def extract_completion_text(completion: Any) -> str:
    """Normalize verifiers completion objects down to a single string.

    Multi-turn rollouts are a list of messages; we prefer the last assistant
    message that actually contains ``<answer>`` tags.
    """
    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        texts: list[str] = []
        for message in completion:
            role, content = _message_role_content(message)
            if role not in (None, "assistant"):
                continue
            text = _content_to_text(content)
            if text:
                texts.append(text)
        for text in reversed(texts):
            if ANSWER_RE.search(text):
                return text
        return texts[-1] if texts else ""
    if isinstance(completion, dict):
        return _content_to_text(completion.get("content", ""))
    role, content = _message_role_content(completion)
    if content is not None:
        return _content_to_text(content)
    return str(completion)


def _message_role_content(message: Any) -> tuple[Any, Any]:
    if isinstance(message, dict):
        return message.get("role"), message.get("content")
    role = getattr(message, "role", None)
    content = getattr(message, "content", None)
    if role is None and content is None and hasattr(message, "model_dump"):
        dumped = message.model_dump()
        if isinstance(dumped, dict):
            return dumped.get("role"), dumped.get("content")
    return role, content


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
            else:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


def parse_answer(completion: Any) -> str | None:
    text = extract_completion_text(completion)
    match = ANSWER_RE.search(text)
    if not match:
        return None
    return match.group(1).strip()


def has_answer_tags(completion: Any) -> bool:
    return parse_answer(completion) is not None


def normalize_answer(value: str) -> str:
    text = " ".join(value.strip().split())
    if text.upper() == "NONE":
        return "NONE"
    parsed = parse_utc(text)
    if parsed is not None:
        return format_utc(parsed)
    return text


def exact_match_score(completion: Any, answer: str) -> float:
    parsed = parse_answer(completion)
    if parsed is None:
        return 0.0
    return 1.0 if normalize_answer(parsed) == normalize_answer(str(answer)) else 0.0


def format_score(completion: Any) -> float:
    return 1.0 if has_answer_tags(completion) else 0.0


def partial_credit_score(
    completion: Any, answer: str, info: dict[str, Any] | None = None
) -> float:
    """0.5 if the parsed start is valid but not the earliest. 0 on exact match."""
    parsed = parse_answer(completion)
    if parsed is None:
        return 0.0
    if normalize_answer(parsed) == normalize_answer(str(answer)):
        return 0.0
    info = info or {}
    world = decode_world(info.get("world"))
    if not world.get("attendees"):
        return 0.0
    if normalize_answer(parsed) == "NONE":
        return 0.0
    if is_valid_slot(
        world,
        parsed,
        int(info["duration_minutes"]),
        info["window_start"],
        int(info["window_days"]),
    ):
        return 0.5
    return 0.0


def grade(completion: Any, answer: str, info: dict[str, Any] | None = None) -> dict[str, float]:
    exact = exact_match_score(completion, answer)
    fmt = format_score(completion)
    partial = partial_credit_score(completion, answer, info)
    weighted = 1.0 * exact + 0.2 * fmt + 0.2 * partial
    return {
        "exact_match": exact,
        "format": fmt,
        "partial_credit": partial,
        "reward": weighted,
    }


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def _question_for(info: dict[str, Any]) -> str:
    window_end = window_end_date(info["window_start"], info["window_days"])
    duration = int(info["duration_minutes"])
    req = info.get("request_id", "--------")
    return (
        f"Meeting request {req}: find the earliest UTC start time for a "
        f"{duration}-minute meeting.\n\n"
        f"Search window: {info['window_start']} to {window_end} "
        f"(inclusive calendar dates in UTC). A valid start must fall inside this window.\n\n"
        "Working hours, timezones, and busy calendars are not in this prompt — "
        "you must use the tools (list_attendees, get_timezone, get_working_hours, get_busy).\n\n"
        "Intervals are half-open [start, end). Put the earliest valid start in "
        "<answer>YYYY-MM-DDTHH:MM:SSZ</answer>. If no slot exists, answer NONE."
    )


def _request_id(info: dict[str, Any]) -> str:
    blob = json.dumps(
        {
            "world": info["world"],
            "duration_minutes": info["duration_minutes"],
            "window_start": info["window_start"],
            "window_days": info["window_days"],
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8].upper()


def example_from_spec(spec: dict[str, Any]) -> dict[str, Any]:
    info = {
        "duration_minutes": int(spec["duration_minutes"]),
        "window_start": spec["window_start"],
        "window_days": int(spec["window_days"]),
        "difficulty": spec.get("difficulty", "hard"),
        "world": spec["world"],
    }
    if spec.get("id"):
        info["edge_id"] = spec["id"]
    info["request_id"] = _request_id(info)
    answer = gold_earliest(
        info["world"],
        info["duration_minutes"],
        info["window_start"],
        info["window_days"],
    )
    info["gold"] = answer
    return {
        "question": _question_for(info),
        "answer": answer,
        "difficulty": info["difficulty"],
        "info": info,
    }


def _random_monday(rng: random.Random) -> date:
    # 2024-01-01 was a Monday; stay in 2024–2025, skip obvious DST weeks here.
    start = date(2024, 1, 8)
    weeks = rng.randint(0, 90)
    return start + timedelta(weeks=weeks)


def _hours_only_world(attendees: list[dict[str, Any]]) -> dict[str, Any]:
    stripped = []
    for att in attendees:
        stripped.append(
            {
                "name": att["name"],
                "timezone": att["timezone"],
                "working_hours": att["working_hours"],
                "busy": {},
            }
        )
    return {"attendees": stripped}


def _grid_valid_starts(
    world: dict[str, Any],
    duration_minutes: int,
    window_start: str,
    window_days: int,
    step: int = 15,
) -> list[datetime]:
    window_0, window_1 = window_bounds(window_start, window_days)
    out: list[datetime] = []
    t = window_0
    while t < window_1:
        if is_valid_slot(world, format_utc(t), duration_minutes, window_start, window_days):
            out.append(t)
        t += timedelta(minutes=step)
    return out


def _fully_busy(attendees: list[dict[str, Any]], window_start: str, window_days: int) -> None:
    window_0, window_1 = window_bounds(window_start, window_days)
    pad_start = (window_0 - timedelta(days=2)).date()
    pad_end = (window_1 + timedelta(days=2)).date()
    for att in attendees:
        hours = att["working_hours"]
        weekdays = set(int(d) for d in hours["weekdays"])
        busy: dict[str, list[dict[str, str]]] = {}
        for day in _iter_days(pad_start, pad_end):
            if day.weekday() in weekdays:
                busy[day.isoformat()] = [{"start": hours["start"], "end": hours["end"]}]
        att["busy"] = busy


def _plant_slot(
    attendees: list[dict[str, Any]],
    target: datetime,
    duration_minutes: int,
    window_start: str,
    window_days: int,
) -> bool:
    """Busy-out every work minute except [target, target+duration)."""
    window_0, window_1 = window_bounds(window_start, window_days)
    duration = timedelta(minutes=int(duration_minutes))
    end = target + duration
    pad_start = (window_0 - timedelta(days=2)).date()
    pad_end = (window_1 + timedelta(days=2)).date()
    for att in attendees:
        tz = ZoneInfo(att["timezone"])
        local_s = target.astimezone(tz)
        local_e = end.astimezone(tz)
        if local_s.date() != (local_e - timedelta(microseconds=1)).date():
            return False
        hours = att["working_hours"]
        work_s, work_e = hm_to_min(hours["start"]), hm_to_min(hours["end"])
        slot_s = local_s.hour * 60 + local_s.minute
        slot_e = local_e.hour * 60 + local_e.minute
        if slot_s < work_s or slot_e > work_e:
            return False
        busy: dict[str, list[dict[str, str]]] = {}
        for day in _iter_days(pad_start, pad_end):
            if day.weekday() not in set(int(d) for d in hours["weekdays"]):
                continue
            if day != local_s.date():
                busy[day.isoformat()] = [{"start": hours["start"], "end": hours["end"]}]
                continue
            blocks: list[dict[str, str]] = []
            if work_s < slot_s:
                blocks.append({"start": min_to_hm(work_s), "end": min_to_hm(slot_s)})
            if slot_e < work_e:
                blocks.append({"start": min_to_hm(slot_e), "end": min_to_hm(work_e)})
            if blocks:
                busy[day.isoformat()] = blocks
        att["busy"] = busy
    return True


def generate_example(
    rng: random.Random,
    *,
    difficulty: Difficulty | None = None,
) -> dict[str, Any]:
    difficulty = difficulty or rng.choices(
        population=list(DIFFICULTY_WEIGHTS),
        weights=list(DIFFICULTY_WEIGHTS.values()),
        k=1,
    )[0]
    duration = {
        "easy": rng.choice((30, 60)),
        "medium": rng.choice((30, 45, 60)),
        "hard": rng.choice((15, 30, 60)),
    }[difficulty]
    if difficulty == "hard" and rng.random() < 0.45:
        window_start = rng.choice(DST_FRIDAYS)
        window_days = rng.choice((3, 4, 5))
    else:
        window_start = _random_monday(rng).isoformat()
        window_days = {
            "easy": 1,
            "medium": rng.choice((1, 2, 3)),
            "hard": rng.choice((2, 3, 4)),
        }[difficulty]

    want_none = rng.random() < {"easy": 0.08, "medium": 0.10, "hard": 0.14}[difficulty]
    if want_none and rng.random() < 0.55:
        pair = rng.choice(NONE_PAIRS)
        attendees = [
            _att(pair[0][0], pair[0][1]),
            _att(pair[1][0], pair[1][1]),
        ]
        if rng.random() < 0.4:
            _fully_busy(attendees, window_start, window_days)
    elif want_none:
        cluster = rng.choice(list(CLUSTERS.values()))
        n_att = min(len(cluster), {"easy": 2, "medium": rng.choice((2, 3)), "hard": rng.choice((3, 4))}[difficulty])
        people = rng.sample(list(cluster), min(n_att, len(cluster)))
        attendees = [_att(name, tz) for name, tz in people]
        _fully_busy(attendees, window_start, window_days)
    else:
        attendees = _generate_valid_attendees(
            rng, difficulty, duration, window_start, window_days
        )

    world = {"attendees": attendees}
    spec = {
        "duration_minutes": duration,
        "window_start": window_start,
        "window_days": window_days,
        "difficulty": difficulty,
        "world": world,
    }
    return example_from_spec(spec)


def _generate_valid_attendees(
    rng: random.Random,
    difficulty: Difficulty,
    duration: int,
    window_start: str,
    window_days: int,
) -> list[dict[str, Any]]:
    cluster_name = {
        "easy": rng.choice(("us", "eu", "asia")),
        "medium": rng.choice(("us", "eu", "asia", "atlantic")),
        "hard": rng.choice(("atlantic", "eu", "us", "asia")),
    }[difficulty]
    cluster = list(CLUSTERS[cluster_name])
    n_att = {
        "easy": 2,
        "medium": min(len(cluster), rng.choice((2, 3))),
        "hard": min(len(cluster), rng.choice((3, 4))) if len(cluster) >= 3 else len(cluster),
    }[difficulty]
    n_att = max(2, min(n_att, len(cluster)))

    for _ in range(24):
        people = rng.sample(cluster, n_att)
        attendees: list[dict[str, Any]] = []
        for name, tz in people:
            hs, he = ("09:00", "17:00") if difficulty == "easy" else rng.choice(HOURS_CHOICES)
            attendees.append(_att(name, tz, start=hs, end=he))
        hours_world = _hours_only_world(attendees)
        candidates = _grid_valid_starts(
            hours_world, duration, window_start, window_days, step=15
        )
        if not candidates:
            continue
        if difficulty == "easy" and rng.random() < 0.55:
            return attendees
        cap = 12 if difficulty == "hard" else 6
        target = candidates[rng.randint(0, min(len(candidates) - 1, cap))]
        if _plant_slot(attendees, target, duration, window_start, window_days):
            planted = gold_earliest(
                {"attendees": attendees}, duration, window_start, window_days
            )
            if planted != "NONE":
                return attendees
        # plant failed; try again with empty hours-only world
        if difficulty == "easy":
            return [
                _att(p[0], p[1]) for p in people
            ]

    # Guaranteed overlap fallback: two New Yorkers, standard hours, empty busy.
    return [
        _att("Ava", "America/New_York"),
        _att("Ben", "America/New_York"),
    ]


def build_rows(
    *,
    n: int,
    seed: int,
    include_edge_cases: bool = False,
    difficulty: Difficulty | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    if include_edge_cases:
        for spec in EDGE_CASES:
            if difficulty is None or spec.get("difficulty") == difficulty:
                rows.append(example_from_spec(spec))
    seen = {row["question"] for row in rows}
    guard = 0
    while len(rows) < n and guard < n * 40:
        guard += 1
        example = generate_example(rng, difficulty=difficulty)
        if example["question"] in seen:
            continue
        seen.add(example["question"])
        rows.append(example)
    if len(rows) < n:
        raise RuntimeError(f"could only generate {len(rows)} unique rows (wanted {n})")
    return rows[:n]


# ---------------------------------------------------------------------------
# load_environment — Hub / verifiers v0 entrypoint
# ---------------------------------------------------------------------------


def load_environment(
    num_train_examples: int = 500,
    num_eval_examples: int = 100,
    seed: int = 42,
    max_turns: int = 40,
    difficulty: Difficulty | None = None,
    **kwargs: Any,
) -> Any:
    """Build a ``vf.StatefulToolEnv`` for earliest-meeting search.

    Dataset rows never use a top-level ``task`` string — verifiers ≥0.1
    treats ``info["task"]`` as a nested rollout payload.

    Parameters
    ----------
    num_train_examples:
        Size of the train split. Default 500.
    num_eval_examples:
        Size of the eval split. Curated DST / TZ / no-slot / busy-gap rows
        are always prepended, then unique random rows fill the rest.
    seed:
        Dataset RNG seed. Train uses ``seed``; eval uses ``seed + 1``.
    max_turns:
        Tool-call turns before stop. 4 attendees × ~5 local dates needs
        headroom; default 40.
    difficulty:
        Pin generation to one band, or ``None`` for a weighted mix.
    """
    import verifiers as vf
    from datasets import Dataset

    train_rows = build_rows(
        n=num_train_examples, seed=seed, difficulty=difficulty
    )
    eval_rows = build_rows(
        n=num_eval_examples,
        seed=seed + 1,
        include_edge_cases=True,
        difficulty=difficulty,
    )
    train_ds = Dataset.from_list(rows_for_dataset(train_rows))
    eval_ds = Dataset.from_list(rows_for_dataset(eval_rows))

    parser = vf.XMLParser(["answer"], answer_field="answer")

    def exact_match_reward(completion, answer, **_kwargs) -> float:
        parsed = parser.parse_answer(completion)
        if parsed is None:
            return exact_match_score(completion, answer)
        return 1.0 if normalize_answer(parsed) == normalize_answer(str(answer)) else 0.0

    def format_reward(completion, **_kwargs) -> float:
        return format_score(completion)

    def partial_credit_reward(completion, answer, info=None, **_kwargs) -> float:
        return partial_credit_score(completion, answer, info)

    rubric = vf.Rubric(
        funcs=[exact_match_reward, format_reward, partial_credit_reward],
        weights=[1.0, 0.2, 0.2],
        parser=parser,
    )

    class MeetingSlotEnv(vf.StatefulToolEnv):
        async def setup_state(self, state, **_kwargs):
            info = state.get("info") or {}
            state["world"] = decode_world(info.get("world"))
            parent_setup = getattr(super(), "setup_state", None)
            if parent_setup is not None:
                result = parent_setup(state)
                if hasattr(result, "__await__"):
                    await result

        def update_tool_args(self, tool_name, tool_args, messages, state, **_kwargs):
            updated = dict(tool_args or {})
            world = state.get("world")
            if world is None:
                world = decode_world((state.get("info") or {}).get("world"))
            updated["world"] = world
            return updated

    env = MeetingSlotEnv(
        tools=[],
        dataset=train_ds,
        eval_dataset=eval_ds,
        system_prompt=SYSTEM_PROMPT,
        parser=parser,
        rubric=rubric,
        max_turns=max_turns,
        **{k: v for k, v in kwargs.items() if k in ("max_concurrent",)},
    )
    env.add_tool(list_attendees, args_to_skip=["world"])
    env.add_tool(get_timezone, args_to_skip=["world"])
    env.add_tool(get_working_hours, args_to_skip=["world"])
    env.add_tool(get_busy, args_to_skip=["world"])
    return env
