#!/usr/bin/env python3
"""Integration test for Progressive Disclosure training pipeline."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_io.io_utils import read_jsonl, write_jsonl
from disclosure.answer_rollout import run_answer_rollouts
from disclosure.budget_allocator import allocate_budget
from disclosure.metrics import evaluate_disclosure_predictions
from disclosure.on_policy_buffer import (
    OnPolicyReplayBuffer,
    ReplayRecord,
    build_replay_records_from_targets,
    mix_replay_with_onpolicy,
)
from disclosure.renderer import render_prompt, render_span
from disclosure.review_sft_builder import build_review_sft_targets
from disclosure.selection_prompt import build_selection_prompt, parse_selection_response
from disclosure.selection_rollout import run_selection_rollouts
from disclosure.schemas import ContextSpan, DisclosureDecision, DisclosureRollout
from disclosure.slime_export import export_review_sft_records
from disclosure.slime_sft_builder import build_slime_sft_record
from disclosure.summary_builder import audit_summaries, span_from_raw
from disclosure.teacher_review import heuristic_teacher_target


def create_test_spans() -> list[ContextSpan]:
    """Create test spans for integration testing."""
    return [
        span_from_raw(
            task_id="test_task_001",
            span_id="span_001",
            source_id="doc1",
            question="What is the capital of France?",
            raw_text="France is a country in Western Europe. Its capital and largest city is Paris, a global center for art and culture.",
            split="train",
            max_summary_words=10,
        ),
        span_from_raw(
            task_id="test_task_001",
            span_id="span_002",
            source_id="doc2",
            question="What is the capital of France?",
            raw_text="Germany is a country in Central Europe. Berlin is its capital and largest city.",
            split="train",
            max_summary_words=10,
        ),
        span_from_raw(
            task_id="test_task_001",
            span_id="span_003",
            source_id="doc3",
            question="What is the capital of France?",
            raw_text="Paris is known for the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral.",
            split="train",
            max_summary_words=10,
        ),
    ]


def test_span_creation_and_validation():
    """Test 1: Span creation and validation."""
    print("\n" + "="*60)
    print("TEST 1: Span Creation and Validation")
    print("="*60)
    
    spans = create_test_spans()
    print(f"Created {len(spans)} test spans")
    
    for span in spans:
        span.validate()
        print(f"  ✓ Span {span.span_id}: {span.summary_text[:50]}...")
        print(f"    Raw tokens: {span.raw_tokens}, Summary tokens: {span.summary_tokens}")
    
    # Audit summaries
    audit = audit_summaries(spans)
    print(f"\nSummary audit: {json.dumps(audit.to_dict(), indent=2)}")
    
    assert len(spans) == 3
    assert all(s.summary_cues for s in spans)
    print("✅ Test 1 PASSED")
    return spans


def test_selection_prompt_and_parsing(spans: list[ContextSpan]):
    """Test 2: Selection prompt building and response parsing."""
    print("\n" + "="*60)
    print("TEST 2: Selection Prompt and Response Parsing")
    print("="*60)
    
    # Build selection prompt
    prompt = build_selection_prompt(spans, token_budget=256)
    print(f"Selection prompt length: {len(prompt)} chars")
    print(f"Prompt preview:\n{prompt[:300]}...\n")
    
    # Test parsing valid responses - must include ALL span_ids
    valid_responses = [
        '{"span_001": "EXPAND", "span_002": "HIDDEN", "span_003": "SUMMARY"}',
        '{"span_001": "SUMMARY", "span_002": "SUMMARY", "span_003": "SUMMARY"}',
    ]
    
    for resp in valid_responses:
        decisions = parse_selection_response(resp, spans)
        print(f"  Parsed response: {len(decisions)} decisions")
        for d in decisions:
            print(f"    - {d.span_id}: {d.action}")
    
    # Test invalid response
    try:
        parse_selection_response('{"span_001": "INVALID", "span_002": "SUMMARY", "span_003": "SUMMARY"}', spans)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  ✓ Correctly rejected invalid action: {e}")
    
    # Test incomplete response (missing span)
    try:
        parse_selection_response('{"span_001": "EXPAND"}', spans)
        assert False, "Should have raised ValueError for missing span"
    except ValueError as e:
        print(f"  ✓ Correctly rejected incomplete response: {e}")
    
    print("✅ Test 2 PASSED")
    return prompt


def test_budget_allocation(spans: list[ContextSpan]):
    """Test 3: Budget-aware allocation."""
    print("\n" + "="*60)
    print("TEST 3: Budget Allocation")
    print("="*60)
    
    # Create decisions with scores
    decisions = [
        DisclosureDecision("span_001", "EXPAND", {"EXPAND": 3.0, "SUMMARY": 1.0, "HIDDEN": 0.0}),
        DisclosureDecision("span_002", "HIDDEN", {"EXPAND": 0.5, "SUMMARY": 1.0, "HIDDEN": 2.0}),
        DisclosureDecision("span_003", "SUMMARY", {"EXPAND": 1.5, "SUMMARY": 2.0, "HIDDEN": 0.5}),
    ]
    
    # Test greedy allocation
    record = allocate_budget(
        spans, decisions,
        rollout_id="test_rollout_001",
        token_budget=50,
        reserved_prompt_tokens=10,
        allocator="greedy_margin_per_cost",
    )
    
    print(f"Budget allocation result:")
    print(f"  Allocator: {record.allocator}")
    print(f"  Budget used: {record.budget_used}/{record.token_budget}")
    print(f"  Budget violation: {record.budget_violation}")
    print(f"  Final actions: {record.final_actions}")
    
    assert not record.budget_violation
    assert record.budget_used <= record.token_budget
    
    # Test knapsack allocation
    record2 = allocate_budget(
        spans, decisions,
        rollout_id="test_rollout_002",
        token_budget=100,
        allocator="knapsack_analysis",
    )
    print(f"\nKnapsack allocation:")
    print(f"  Budget used: {record2.budget_used}")
    
    print("✅ Test 3 PASSED")
    return record


def test_selection_and_answer_rollout(spans: list[ContextSpan]):
    """Test 4: Selection and answer rollouts."""
    print("\n" + "="*60)
    print("TEST 4: Selection and Answer Rollouts")
    print("="*60)
    
    # Run selection rollouts
    selections = run_selection_rollouts(
        spans,
        policy_name="rule_expand_omission",
        token_budget=256,
        reserved_prompt_tokens=50,
        allocator="greedy_margin_per_cost",
    )
    
    print(f"Selection rollouts: {len(selections)}")
    for sel in selections:
        print(f"  Selection {sel.selection_id}:")
        print(f"    Policy: {sel.policy_name}")
        print(f"    Budget: {sel.token_budget}")
        for state in sel.states:
            print(f"      {state.span_id}: {state.action}")
    
    # Run answer rollouts
    answers = run_answer_rollouts(
        spans,
        [s.to_dict() for s in selections],
        token_budget=256,
        reserved_prompt_tokens=50,
    )
    
    print(f"\nAnswer rollouts: {len(answers)}")
    for ans in answers:
        print(f"  Answer {ans.answer_id}:")
        print(f"    Selection: {ans.selection_id}")
        print(f"    Answer preview: {ans.answer[:100]}...")
    
    print("✅ Test 4 PASSED")
    return selections, answers


def test_review_sft_targets(spans: list[ContextSpan], selections: list, answers: list):
    """Test 5: Review-SFT target building."""
    print("\n" + "="*60)
    print("TEST 5: Review-SFT Target Building")
    print("="*60)
    
    targets = build_review_sft_targets(
        spans,
        [s.to_dict() for s in selections],
        [a.to_dict() for a in answers],
    )
    
    print(f"Review-SFT targets: {len(targets)}")
    for target in targets[:3]:  # Show first 3
        print(f"  Target {target.target_id}:")
        print(f"    Student state: {target.student_state}")
        print(f"    Target state: {target.target_state}")
        print(f"    Reason: {target.verifiable_reason[:60]}...")
    
    # Export to Slime format
    records = export_review_sft_records(spans, targets)
    print(f"\nExported {len(records)} Slime records")
    
    for rec in records[:2]:
        print(f"  Record prompt preview: {rec['prompt'][:80]}...")
        print(f"  Record label: {rec['label']}")
    
    print("✅ Test 5 PASSED")
    return targets, records


def test_replay_buffer(targets: list):
    """Test 6: Replay buffer functionality."""
    print("\n" + "="*60)
    print("TEST 6: Replay Buffer")
    print("="*60)
    
    buffer = OnPolicyReplayBuffer(max_size=100, replay_ratio=0.3, seed=42)
    
    # Build replay records from targets
    replay_records = build_replay_records_from_targets(targets, iteration=0)
    print(f"Built {len(replay_records)} replay records")
    
    # Add to buffer
    added = buffer.add(replay_records)
    print(f"Added {added} records to buffer")
    
    # Get stats
    stats = buffer.stats()
    print(f"Buffer stats: {json.dumps(stats, indent=2)}")
    
    # Sample from buffer
    sampled = buffer.sample(batch_size=5, current_iteration=1)
    print(f"Sampled {len(sampled)} records for training")
    
    # Test save/load
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "replay_buffer.jsonl"
        buffer.save(path)
        
        new_buffer = OnPolicyReplayBuffer(max_size=100)
        loaded = new_buffer.load(path)
        print(f"Loaded {loaded} records into new buffer")
    
    print("✅ Test 6 PASSED")


def test_rendering_and_metrics(spans: list[ContextSpan]):
    """Test 7: Context rendering and metrics."""
    print("\n" + "="*60)
    print("TEST 7: Rendering and Metrics")
    print("="*60)
    
    # Render individual spans
    for span in spans:
        for action in ["EXPAND", "SUMMARY", "HIDDEN"]:
            rendered = render_span(span, action)
            if rendered:
                print(f"  {span.span_id} ({action}): {rendered[:60]}...")
            else:
                print(f"  {span.span_id} ({action}): [HIDDEN]")
    
    # Render full prompt
    decisions = [
        DisclosureDecision("span_001", "EXPAND"),
        DisclosureDecision("span_002", "HIDDEN"),
        DisclosureDecision("span_003", "SUMMARY"),
    ]
    
    prompt = render_prompt(
        spans[0].question,
        spans,
        decisions,
        system_prompt="Answer the question using disclosed evidence.",
    )
    
    print(f"\nRendered prompt length: {len(prompt)} chars")
    print(f"Prompt preview:\n{prompt[:400]}...")
    
    # Test metrics
    ground_truth = [
        {"decision_id": "d1", "target_action": "EXPAND"},
        {"decision_id": "d2", "target_action": "SUMMARY"},
    ]
    predictions = [
        {"decision_id": "d1", "pred_action": "EXPAND"},
        {"decision_id": "d2", "pred_action": "SUMMARY"},
    ]
    
    metrics = evaluate_disclosure_predictions(ground_truth, predictions)
    print(f"\nMetrics: {json.dumps(metrics, indent=2)}")
    
    assert metrics["action_accuracy"] == 1.0
    print("✅ Test 7 PASSED")


def test_end_to_end_pipeline():
    """Test 8: Complete end-to-end pipeline."""
    print("\n" + "="*60)
    print("TEST 8: End-to-End Pipeline")
    print("="*60)
    
    # Create spans
    spans = create_test_spans()
    
    # Run full pipeline
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        
        # 1. Save spans
        spans_file = tmp_path / "spans.jsonl"
        write_jsonl(spans_file, [s.to_dict() for s in spans])
        print(f"Step 1: Saved {len(spans)} spans to {spans_file}")
        
        # 2. Run selection rollouts
        selections = run_selection_rollouts(spans, policy_name="rule_expand_omission", token_budget=256)
        print(f"Step 2: Generated {len(selections)} selection rollouts")
        
        # 3. Run answer rollouts
        answers = run_answer_rollouts(spans, [s.to_dict() for s in selections], token_budget=256)
        print(f"Step 3: Generated {len(answers)} answer rollouts")
        
        # 4. Build review targets
        targets = build_review_sft_targets(spans, [s.to_dict() for s in selections], [a.to_dict() for a in answers])
        print(f"Step 4: Built {len(targets)} review-SFT targets")
        
        # 5. Export to Slime format
        records = export_review_sft_records(spans, targets)
        slime_file = tmp_path / "slime_train.jsonl"
        write_jsonl(slime_file, records)
        print(f"Step 5: Exported {len(records)} Slime training records")
        
        # 6. Add to replay buffer
        buffer = OnPolicyReplayBuffer(max_size=1000, replay_ratio=0.2)
        replay_records = build_replay_records_from_targets(targets, iteration=0)
        buffer.add(replay_records)
        print(f"Step 6: Added {len(replay_records)} records to replay buffer")
        
        # 7. Verify outputs
        loaded_records = read_jsonl(slime_file)
        print(f"Step 7: Verified {len(loaded_records)} records in output file")
        
        assert len(loaded_records) == len(records)
        assert all("prompt" in r and "label" in r for r in loaded_records)
        
    print("✅ Test 8 PASSED")


def main():
    """Run all integration tests."""
    print("\n" + "="*70)
    print("PROGRESSIVE DISCLOSURE INTEGRATION TEST SUITE")
    print("="*70)
    
    try:
        spans = test_span_creation_and_validation()
        test_selection_prompt_and_parsing(spans)
        test_budget_allocation(spans)
        selections, answers = test_selection_and_answer_rollout(spans)
        targets, records = test_review_sft_targets(spans, selections, answers)
        test_replay_buffer(targets)
        test_rendering_and_metrics(spans)
        test_end_to_end_pipeline()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED")
        print("="*70)
        return 0
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
