# meeting-slot

Multi-turn **tool-using meeting scheduler** for RLVR / evals on the [Prime Intellect Environments Hub](https://app.primeintellect.ai/dashboard/environments).

Source: [github.com/devtechedge/meeting-slot](https://github.com/devtechedge/meeting-slot)

A sequel to [calendar-math](https://github.com/devtechedge/calendar-math): same calendar domain, but the model has to **query tools** instead of reading the calendar out of the prompt.

The task is to find the **earliest valid UTC start** that works for every attendee. Busy intervals, working hours, and timezones are hidden. The final answer goes in `<answer>` tags as a UTC ISO-8601 timestamp, e.g. `2024-03-11T15:00:00Z`. If no slot exists, `NONE`.

The grader is a UTC sweep-line over `zoneinfo` — no LLM-as-judge, no fuzzy string match on the main reward.

| Tool | Returns |
| --- | --- |
| `list_attendees()` | JSON names |
| `get_timezone(name)` | IANA timezone |
| `get_working_hours(name)` | local `HH:MM` hours + weekdays (`0=Monday`) |
| `get_busy(name, date)` | local half-open busy intervals for that **local** date |

## Why this design

- **Verifiable.** Gold answers are produced by interval intersection in UTC. The same functions are the reference solver.
- **Tool use is load-bearing.** The world is not in the prompt. A model that does not call tools cannot solve the task.
- **Hard where it matters.** Eval always includes curated edge cases: US/EU DST, NY↔Kolkata no-overlap, Friday 16:30 vs Monday 09:00, 30 vs 60 minute gaps, four-person summer overlap, inclusive-busy traps.
- **Not gameable by format alone.** Format is a 0.2 bonus. Exact match is the 1.0 term.
- **Shaping, not noise.** A conflict-free in-hours start that is not the earliest scores 0.5 partial credit. Invalid overlap / wrong duration / outside hours score 0 on the main terms.
- **Configurable.** `num_train_examples`, `num_eval_examples`, `seed`, `max_turns`, optional `difficulty` pin.

## Reward

```
reward = 1.0 * exact_match + 0.2 * format + 0.2 * partial_credit
```

| Term | 1.0 when | Notes |
| --- | --- | --- |
| `exact_match` | parsed `<answer>` equals gold | UTC ISO with `Z`, or `NONE` |
| `format` | `<answer>...</answer>` present | extra prose outside the tags is ignored |
| `partial_credit` | valid-but-not-earliest | 0.5; 0 when exact match already fired, so a perfect answer is **1.2** not 1.4 |

Intervals are half-open `[start, end)`. A busy block ending at 11:00 means 11:00 is free.

## Eval

15 curated edge cases (`num_eval_examples=15`, 1 rollout each).

| Policy | avg reward | exact | format | partial |
| --- | --- | --- | --- | --- |
| Gold sweep-line via tools (ceiling) | **1.200** | 1.000 | 1.000 | 0.000 |
| Naive (ignore TZ; busy end inclusive) | **0.293** | 0.067 | 1.000 | 0.133 |
| `minimax/minimax-m2.7` (OpenRouter, T=0, 2048 tok) | **0.993** | 0.800 | 0.933 | 0.033 |

The gold policy is a harness check: tools leak enough to rebuild the world, and the rubric fires 1.2. The naive policy is a discrimination check: DST offsets, NY/Kolkata non-overlap, and inclusive busy do not rubber-stamp 1.2. Naive is exact on only the fully-booked `NONE` row.

MiniMax is exact on **12/15**. The three misses are the ones the env is supposed to catch:

- `sydney_ny_none` — format only (0.2). Claimed a 13:00Z overlap between Sydney and New York that does not exist.
- `dst_eu_monday` — valid-not-earliest (0.3). Answered 10:00Z after the EU spring-forward; gold is 08:00Z.
- `no_slot_fully_booked` — 0.0. Truncated before `</answer>` at 2048 tokens.

```bash
uv run vf-eval meeting-slot -n 15 -r 1 -p openrouter \
  -m minimax/minimax-m2.7 --max-tokens 2048 \
  --temperature 0 --max-concurrent 1 --disable-tui --disable-env-server
```

`--max-concurrent 1` keeps OpenRouter in-flight budget from aborting mid-rollout. Budget ≥2048 max tokens. Reasoning models spend the window on tool calls before `</answer>`; truncation looks like a grader bug.

## Installation

```bash
uv pip install -e .
python -m pytest tests/test_meeting_slot.py -q
```

From the Hub (after push):

```bash
prime env install devtechedge/meeting-slot
```

```python
import verifiers as vf

env = vf.load_environment("meeting-slot")
```

Requires `verifiers>=0.1.14,<0.2`.

## `load_environment` arguments

| Arg | Default | Meaning |
| --- | --- | --- |
| `num_train_examples` | `500` | train split size |
| `num_eval_examples` | `100` | eval split size (edge cases prepended) |
| `seed` | `42` | train RNG; eval uses `seed + 1` |
| `max_turns` | `40` | tool-call turns before stop |
| `difficulty` | `None` | `"easy"` \| `"medium"` \| `"hard"` \| mixed |

```bash
uv run vf-eval meeting-slot -n 20
uv run vf-eval meeting-slot -a '{"difficulty": "hard", "num_eval_examples": 40}'
```

Dataset rows never use a column named `task`. Verifiers ≥0.1 treats `info["task"]` as a nested rollout payload.

## Gold solution

Dataset construction **is** the gold solver. For each attendee, working hours minus busy are converted to UTC with `zoneinfo` (both DST folds, skipping spring-forward gaps). The UTC free intervals are intersected; the earliest start `s` with `s + duration` inside the intersection and inside the search window is gold.

```python
from meeting_slot import gold_earliest, solve_from_tools

gold_earliest(world, duration_minutes, window_start, window_days)
# identical, but only using the four public tools:
solve_from_tools(world, duration_minutes, window_start, window_days)
```

`tests/test_meeting_slot.py` asserts the solver against a hand-checked fixture list (US/EU DST, 30 vs 60 minute gaps, NY–Kolkata NONE, inclusive busy).

## Files

```
meeting_slot.py                 # generator, gold, tools, grader, load_environment
pyproject.toml
README.md
LICENSE
tests/test_meeting_slot.py
```

## What this is not

- Not a wrap of a public calendar dataset.
- Not LLM-judged.
- Not single-turn. The calendars are behind tools on purpose.
- Not a dump of the whole calendar into the prompt.

## License

MIT
