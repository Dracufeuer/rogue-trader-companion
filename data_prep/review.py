"""
Dataset Quality Review Tool
============================
Use this to review generated Q&A pairs and flag bad ones for removal.
Run after extract.py to clean up the generated dataset.

Usage:
    python data_prep/review.py                    # Review all datasets
    python data_prep/review.py --file rules_qa    # Review specific file
    python data_prep/review.py --stats            # Just show statistics
"""

import json
import argparse
from pathlib import Path

DATASET_DIR = Path("finetune/dataset")


def review_file(jsonl_path: Path):
    """
    Interactive review of a JSONL dataset file.
    Shows each example and lets you mark it as keep/remove/edit.
    """
    if not jsonl_path.exists():
        print(f"File not found: {jsonl_path}")
        return

    with open(jsonl_path, encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]

    print(f"\nReviewing: {jsonl_path.name} ({len(examples)} examples)")
    print("Commands: [k]eep, [d]elete, [s]kip, [q]uit\n")

    keep = []
    removed = 0

    for i, example in enumerate(examples):
        print(f"{'─'*60}")
        print(f"Example {i+1}/{len(examples)}")
        print(f"\nPROMPT:\n{example['prompt']}")
        print(f"\nCOMPLETION:\n{example['completion']}")
        print()

        while True:
            cmd = input("Action [k/d/s/q]: ").strip().lower()
            if cmd in ["k", ""]:
                keep.append(example)
                break
            elif cmd == "d":
                removed += 1
                break
            elif cmd == "s":
                keep.append(example)
                break
            elif cmd == "q":
                # Save what we have and quit
                keep.extend(examples[i+1:])
                _save_reviewed(jsonl_path, keep)
                print(f"\nSaved. Removed {removed} examples.")
                return
            else:
                print("Invalid command. Use k/d/s/q")

    _save_reviewed(jsonl_path, keep)
    print(f"\n✅ Review complete. Kept {len(keep)}, removed {removed} examples.")


def _save_reviewed(path: Path, examples: list[dict]):
    """Save reviewed examples back to the file."""
    backup = path.with_suffix(".jsonl.bak")
    path.rename(backup)

    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Saved to {path} (backup at {backup})")


def show_stats():
    """Show statistics for all dataset files."""
    print("\n📊 Dataset Statistics")
    print("─" * 40)

    total = 0
    for jsonl_file in sorted(DATASET_DIR.glob("*.jsonl")):
        with open(jsonl_file, encoding="utf-8") as f:
            examples = [json.loads(line) for line in f if line.strip()]

        count = len(examples)
        total += count

        # Calculate average lengths
        avg_prompt = sum(len(e["prompt"]) for e in examples) / max(count, 1)
        avg_completion = sum(len(e["completion"]) for e in examples) / max(count, 1)

        print(f"{jsonl_file.name}:")
        print(f"  Examples: {count}")
        print(f"  Avg prompt length: {avg_prompt:.0f} chars")
        print(f"  Avg completion length: {avg_completion:.0f} chars")

    print(f"─" * 40)
    print(f"Total examples: {total}")
    print(f"Target: 500-800")

    if total < 500:
        print(f"⚠ Need {500 - total} more examples to hit minimum target")
    elif total >= 500:
        print(f"✅ Minimum target reached!")


def check_quality(jsonl_path: Path) -> list[str]:
    """
    Automatically flag potential quality issues without manual review.
    Returns a list of warning messages.
    """
    warnings = []

    with open(jsonl_path, encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]

    for i, ex in enumerate(examples):
        prompt = ex.get("prompt", "")
        completion = ex.get("completion", "")

        # Too short
        if len(prompt) < 15:
            warnings.append(f"Line {i+1}: Prompt very short ({len(prompt)} chars)")
        if len(completion) < 30:
            warnings.append(f"Line {i+1}: Completion very short ({len(completion)} chars)")

        # Looks like the model refused or errored
        refuse_patterns = ["I cannot", "I'm sorry", "As an AI", "I don't have access"]
        if any(p in completion for p in refuse_patterns):
            warnings.append(f"Line {i+1}: Possible model refusal in completion")

        # Completion looks like it cut off mid-sentence
        if completion and completion[-1] not in ".!?\"'":
            if len(completion) > 100:
                warnings.append(f"Line {i+1}: Completion may be truncated (ends: ...{completion[-20:]})")

        # Duplicate prompts
        if sum(1 for e in examples if e.get("prompt") == prompt) > 1:
            warnings.append(f"Line {i+1}: Duplicate prompt detected")

    return warnings


def main():
    parser = argparse.ArgumentParser(description="Dataset Quality Review Tool")
    parser.add_argument("--file", type=str, help="Specific file to review (without .jsonl)")
    parser.add_argument("--stats", action="store_true", help="Show statistics only")
    parser.add_argument("--check", action="store_true", help="Auto-check quality issues")
    args = parser.parse_args()

    if args.stats:
        show_stats()
        return

    if args.check:
        print("\n🔍 Quality Check")
        for jsonl_file in sorted(DATASET_DIR.glob("*.jsonl")):
            warnings = check_quality(jsonl_file)
            if warnings:
                print(f"\n{jsonl_file.name}: {len(warnings)} issues found")
                for w in warnings[:10]:  # Show first 10
                    print(f"  ⚠ {w}")
                if len(warnings) > 10:
                    print(f"  ... and {len(warnings) - 10} more")
            else:
                print(f"\n{jsonl_file.name}: ✅ No issues found")
        return

    if args.file:
        path = DATASET_DIR / f"{args.file}.jsonl"
        review_file(path)
    else:
        # Review all files
        for jsonl_file in sorted(DATASET_DIR.glob("*.jsonl")):
            review_file(jsonl_file)
            print()


if __name__ == "__main__":
    main()