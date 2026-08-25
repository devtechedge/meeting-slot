"""Stdlib-only tests for generator, gold solver, tools, and grader.

Run from this directory:

    python3 -m pytest tests/test_meeting_slot.py -q
    python3 tests/test_meeting_slot.py
"""

from __future__ import annotations

import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import meeting_slot as ms  # noqa: E402


EXPECTED_GOLD = {
    "same_tz_empty_busy": "2024-03-11T13:00:00Z",
    "overlapping_busy": "2024-03-11T18:00:00Z",
    "no_overlap_ny_kolkata": "NONE",
    "weekend_skip_dst": "2024-03-11T13:00:00Z",
    "dst_us_spring_monday": "2024-03-11T13:00:00Z",
    "inclusive_busy_utc": "2024-03-11T11:00:00Z",
    "gap_30min": "2024-03-11T11:00:00Z",
    "gap_60min": "2024-03-11T16:00:00Z",
    "four_person_summer": "2024-06-10T13:00:00Z",
    "sydney_ny_none": "NONE",
    "shanghai_tokyo": "2024-03-11T01:00:00Z",
    "last_friday_slot": "2024-03-08T16:30:00Z",
    "dst_eu_monday": "2024-04-01T08:00:00Z",
    "three_person_busy": "2024-06-10T14:00:00Z",
    "no_slot_fully_booked": "NONE",
}


def _row(edge_id: str) -> dict:
    spec = next(s for s in ms.EDGE_CASES if s["id"] == edge_id)
    return ms.example_from_spec(spec)


def _msg(text: str):
    return [{"role": "assistant", "content": text}]


class GoldSolverTests(unittest.TestCase):
    def test_edge_golds_match_hand_checked_utc(self) -> None:
        for spec in ms.EDGE_CASES:
            row = ms.example_from_spec(spec)
            self.assertEqual(row["answer"], EXPECTED_GOLD[spec["id"]], spec["id"])

    def test_us_dst_spring_offset(self) -> None:
        # 2024-03-10 02:00 America/New_York springs forward.
        tz = ZoneInfo("America/New_York")
        friday = datetime(2024, 3, 8, 9, 0, tzinfo=tz).astimezone(timezone.utc)
        monday = datetime(2024, 3, 11, 9, 0, tzinfo=tz).astimezone(timezone.utc)
        self.assertEqual(ms.format_utc(friday), "2024-03-08T14:00:00Z")
        self.assertEqual(ms.format_utc(monday), "2024-03-11T13:00:00Z")
        self.assertEqual(_row("dst_us_spring_monday")["answer"], "2024-03-11T13:00:00Z")

    def test_eu_dst_spring_offset(self) -> None:
        london = ZoneInfo("Europe/London")
        berlin = ZoneInfo("Europe/Berlin")
        monday_london = datetime(2024, 4, 1, 9, 0, tzinfo=london).astimezone(timezone.utc)
        monday_berlin = datetime(2024, 4, 1, 9, 0, tzinfo=berlin).astimezone(timezone.utc)
        self.assertEqual(ms.format_utc(monday_london), "2024-04-01T08:00:00Z")
        self.assertEqual(ms.format_utc(monday_berlin), "2024-04-01T07:00:00Z")
        self.assertEqual(_row("dst_eu_monday")["answer"], "2024-04-01T08:00:00Z")

    def test_30_vs_60_on_same_calendar(self) -> None:
        self.assertEqual(_row("gap_30min")["answer"], "2024-03-11T11:00:00Z")
        self.assertEqual(_row("gap_60min")["answer"], "2024-03-11T16:00:00Z")

    def test_ny_kolkata_hours_do_not_overlap(self) -> None:
        self.assertEqual(_row("no_overlap_ny_kolkata")["answer"], "NONE")

    def test_inclusive_busy_half_open(self) -> None:
        # Busy [09:00, 11:00) means 11:00 is free.
        self.assertEqual(_row("inclusive_busy_utc")["answer"], "2024-03-11T11:00:00Z")

    def test_gold_is_valid_or_none(self) -> None:
        for spec in ms.EDGE_CASES:
            row = ms.example_from_spec(spec)
            info = row["info"]
            if row["answer"] == "NONE":
                self.assertFalse(
                    ms.is_valid_slot(
                        info["world"],
                        "2024-03-11T13:00:00Z",
                        info["duration_minutes"],
                        info["window_start"],
                        info["window_days"],
                    )
                    and info["window_start"] == "2024-03-11"
                )
                continue
            self.assertTrue(
                ms.is_valid_slot(
                    info["world"],
                    row["answer"],
                    info["duration_minutes"],
                    info["window_start"],
                    info["window_days"],
                ),
                spec["id"],
            )

    def test_example_from_spec_fills_answer(self) -> None:
        row = _row("same_tz_empty_busy")
        self.assertEqual(row["answer"], "2024-03-11T13:00:00Z")
        self.assertIn("30-minute", row["question"])
        self.assertIn("2024-03-11", row["question"])


class ToolTests(unittest.TestCase):
    def test_tools_hide_and_rebuild_the_world(self) -> None:
        for spec in ms.EDGE_CASES:
            row = ms.example_from_spec(spec)
            info = row["info"]
            via = ms.solve_from_tools(
                info["world"],
                info["duration_minutes"],
                info["window_start"],
                info["window_days"],
            )
            self.assertEqual(via, row["answer"], spec["id"])

    def test_list_attendees_and_unknown(self) -> None:
        world = _row("same_tz_empty_busy")["info"]["world"]
        names = json.loads(ms.list_attendees(world=world))
        self.assertEqual(names, ["Ava", "Ben"])
        self.assertEqual(ms.get_timezone(name="Ava", world=world), "America/New_York")
        self.assertIn("unknown attendee", ms.get_timezone(name="Zed", world=world))
        hours = json.loads(ms.get_working_hours(name="Ava", world=world))
        self.assertEqual(hours["start"], "09:00")
        self.assertEqual(hours["weekdays"], [0, 1, 2, 3, 4])
        self.assertEqual(json.loads(ms.get_busy(name="Ava", date="2024-03-11", world=world)), [])

    def test_get_busy_is_local_date(self) -> None:
        world = _row("overlapping_busy")["info"]["world"]
        blocks = json.loads(ms.get_busy(name="Ava", date="2024-03-11", world=world))
        self.assertEqual(blocks, [{"start": "09:00", "end": "12:00"}])
        self.assertEqual(json.loads(ms.get_busy(name="Ava", date="2024-03-12", world=world)), [])


class ParserAndGraderTests(unittest.TestCase):
    def test_gold_scores_perfect(self) -> None:
        row = _row("same_tz_empty_busy")
        scores = ms.grade(
            _msg(ms.gold_completion(row["answer"])),
            row["answer"],
            row["info"],
        )
        self.assertEqual(scores["exact_match"], 1.0)
        self.assertEqual(scores["format"], 1.0)
        self.assertEqual(scores["partial_credit"], 0.0)
        self.assertAlmostEqual(scores["reward"], 1.2)

    def test_all_edge_golds_score_1_2(self) -> None:
        for spec in ms.EDGE_CASES:
            row = ms.example_from_spec(spec)
            scores = ms.grade(_msg(f"<answer>{row['answer']}</answer>"), row["answer"], row["info"])
            self.assertAlmostEqual(scores["reward"], 1.2, msg=spec["id"])

    def test_conflict_scores_zero_on_main_terms(self) -> None:
        row = _row("overlapping_busy")
        # 12:00 EDT = 16:00Z, still inside Ben's 11:00-14:00 busy.
        scores = ms.grade(_msg("<answer>2024-03-11T16:00:00Z</answer>"), row["answer"], row["info"])
        self.assertEqual(scores["exact_match"], 0.0)
        self.assertEqual(scores["partial_credit"], 0.0)
        self.assertEqual(scores["format"], 1.0)
        self.assertAlmostEqual(scores["reward"], 0.2)

    def test_valid_not_earliest(self) -> None:
        row = _row("same_tz_empty_busy")
        scores = ms.grade(_msg("<answer>2024-03-11T13:15:00Z</answer>"), row["answer"], row["info"])
        self.assertEqual(scores["exact_match"], 0.0)
        self.assertEqual(scores["format"], 1.0)
        self.assertEqual(scores["partial_credit"], 0.5)
        self.assertAlmostEqual(scores["reward"], 0.2 + 0.2 * 0.5)

    def test_later_friday_skip_is_partial(self) -> None:
        row = _row("last_friday_slot")
        scores = ms.grade(_msg("<answer>2024-03-11T13:00:00Z</answer>"), row["answer"], row["info"])
        self.assertEqual(scores["partial_credit"], 0.5)
        self.assertEqual(scores["format"], 1.0)
        self.assertAlmostEqual(scores["reward"], 0.2 + 0.2 * 0.5)

    def test_format_only(self) -> None:
        row = _row("same_tz_empty_busy")
        scores = ms.grade(_msg("<answer>not-a-time</answer>"), row["answer"], row["info"])
        self.assertEqual(scores["exact_match"], 0.0)
        self.assertEqual(scores["format"], 1.0)
        self.assertEqual(scores["partial_credit"], 0.0)
        self.assertAlmostEqual(scores["reward"], 0.2)

    def test_wrong_answer_no_tags(self) -> None:
        row = _row("same_tz_empty_busy")
        scores = ms.grade(_msg("I think 1pm UTC"), row["answer"], row["info"])
        self.assertEqual(scores["reward"], 0.0)

    def test_none_wrongly_guessed_slot(self) -> None:
        row = _row("no_overlap_ny_kolkata")
        scores = ms.grade(_msg("<answer>2024-03-11T13:00:00Z</answer>"), row["answer"], row["info"])
        self.assertEqual(scores["exact_match"], 0.0)
        self.assertEqual(scores["partial_credit"], 0.0)

    def test_plus_zero_offset_normalizes(self) -> None:
        self.assertEqual(ms.exact_match_score("<answer>2024-03-11T13:00:00+00:00</answer>", "2024-03-11T13:00:00Z"), 1.0)

    def test_pydantic_style_message_objects(self) -> None:
        class Msg:
            def __init__(self) -> None:
                self.role = "assistant"
                self.content = "<answer>2024-03-11T13:00:00Z</answer>"

        row = _row("same_tz_empty_busy")
        scores = ms.grade([Msg()], row["answer"], row["info"])
        self.assertAlmostEqual(scores["reward"], 1.2)

    def test_multi_turn_uses_last_answer_tags(self) -> None:
        row = _row("same_tz_empty_busy")
        completion = [
            {"role": "assistant", "content": "", "tool_calls": [{"name": "list_attendees"}]},
            {"role": "tool", "content": '["Ava","Ben"]'},
            {"role": "assistant", "content": "<answer>2024-03-11T13:00:00Z</answer>"},
        ]
        scores = ms.grade(completion, row["answer"], row["info"])
        self.assertAlmostEqual(scores["reward"], 1.2)

    def test_grader_does_not_rubber_stamp(self) -> None:
        row = _row("four_person_summer")
        scores = ms.grade(_msg("<answer>2024-06-10T09:00:00Z</answer>"), row["answer"], row["info"])
        self.assertLess(scores["reward"], 1.2)
        self.assertEqual(scores["exact_match"], 0.0)


class NaivePolicyTests(unittest.TestCase):
    def test_naive_is_wrong_on_timezone_edges(self) -> None:
        for edge_id in (
            "same_tz_empty_busy",
            "dst_us_spring_monday",
            "shanghai_tokyo",
            "weekend_skip_dst",
        ):
            row = _row(edge_id)
            info = row["info"]
            naive = ms.naive_earliest(
                info["world"], info["duration_minutes"], info["window_start"], info["window_days"]
            )
            self.assertNotEqual(naive, row["answer"], edge_id)

    def test_naive_inclusive_busy_misses_the_boundary(self) -> None:
        row = _row("inclusive_busy_utc")
        info = row["info"]
        naive = ms.naive_earliest(
            info["world"], info["duration_minutes"], info["window_start"], info["window_days"]
        )
        self.assertEqual(naive, "2024-03-11T11:01:00Z")
        scores = ms.grade(_msg(f"<answer>{naive}</answer>"), row["answer"], info)
        self.assertEqual(scores["exact_match"], 0.0)
        self.assertEqual(scores["partial_credit"], 0.5)

    def test_naive_mean_reward_below_gold(self) -> None:
        rewards = []
        exacts = []
        for spec in ms.EDGE_CASES:
            row = ms.example_from_spec(spec)
            info = row["info"]
            naive = ms.naive_earliest(
                info["world"], info["duration_minutes"], info["window_start"], info["window_days"]
            )
            scores = ms.grade(_msg(f"<answer>{naive}</answer>"), row["answer"], info)
            rewards.append(scores["reward"])
            exacts.append(scores["exact_match"])
        self.assertAlmostEqual(sum(rewards) / len(rewards), 0.293333, places=3)
        self.assertEqual(sum(exacts), 1.0)  # only the fully-booked NONE case


class DatasetTests(unittest.TestCase):
    def test_train_size_and_uniqueness(self) -> None:
        rows = ms.build_rows(n=80, seed=0)
        self.assertEqual(len(rows), 80)
        questions = [r["question"] for r in rows]
        self.assertEqual(len(questions), len(set(questions)))
        none_count = 0
        for row in rows:
            self.assertEqual(row["answer"], ms.gold_answer(row))
            self.assertNotIn("task", row)
            self.assertNotIn("task", row["info"])
            self.assertTrue(row["question"])
            info = row["info"]
            self.assertIn("world", info)
            self.assertGreaterEqual(len(info["world"]["attendees"]), 2)
            if row["answer"] == "NONE":
                none_count += 1
            else:
                self.assertTrue(
                    ms.is_valid_slot(
                        info["world"],
                        row["answer"],
                        info["duration_minutes"],
                        info["window_start"],
                        info["window_days"],
                    )
                )
        self.assertLess(none_count / len(rows), 0.25)

    def test_eval_includes_edge_cases(self) -> None:
        rows = ms.build_rows(n=20, seed=1, include_edge_cases=True)
        ids = [r["info"].get("edge_id") for r in rows[:15]]
        self.assertEqual(ids, [spec["id"] for spec in ms.EDGE_CASES])
        questions = " ".join(r["question"] for r in rows)
        self.assertIn("2024-03-11", questions)

    def test_info_is_json_serializable(self) -> None:
        rows = ms.build_rows(n=12, seed=4, include_edge_cases=True)
        for row in rows:
            json.dumps(row["info"])

    def test_none_busy_does_not_crash_grader(self) -> None:
        row = _row("dst_us_spring_monday")
        poisoned = json.loads(json.dumps(row["info"]))
        for att in poisoned["world"]["attendees"]:
            att["busy"] = {"2024-03-11": None, "2024-03-08": None}
        # 14:00Z is valid-but-not-earliest on this row (winter offset).
        scores = ms.grade(
            _msg("<answer>2024-03-11T14:00:00Z</answer>"),
            row["answer"],
            poisoned,
        )
        self.assertEqual(scores["partial_credit"], 0.5)
        self.assertEqual(scores["format"], 1.0)

    def test_json_string_world_roundtrip(self) -> None:
        row = _row("same_tz_empty_busy")
        info = dict(row["info"])
        info["world"] = json.dumps(info["world"])
        scores = ms.grade(_msg(ms.gold_completion(row["answer"])), row["answer"], info)
        self.assertAlmostEqual(scores["reward"], 1.2)
        self.assertEqual(json.loads(ms.get_busy(name="Ava", date="2024-03-12", world=ms.decode_world(info["world"]))), [])


class GoldPolicyHarnessTests(unittest.TestCase):
    def test_gold_with_tools_scores_1_2_on_eval_edges(self) -> None:
        for spec in ms.EDGE_CASES:
            row = ms.example_from_spec(spec)
            info = row["info"]
            answer = ms.solve_from_tools(
                info["world"],
                info["duration_minutes"],
                info["window_start"],
                info["window_days"],
            )
            scores = ms.grade(_msg(ms.gold_completion(answer)), row["answer"], info)
            self.assertAlmostEqual(scores["reward"], 1.2, msg=spec["id"])


if __name__ == "__main__":
    unittest.main()
