"""Export OfficeQA questions into temporary JSONL files and upload to Beaker."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from officeqa.data import load_split
from officeqa.data.dataset import EvalRecord


def build_row(sample: EvalRecord) -> dict[str, Any]:
    return {
        "id": sample.uid,
        "input": {
            "uid": sample.uid,
            "question": sample.question,
            "source_files": sample.source_files,
            "difficulty": sample.difficulty,
        },
        "expected": {
            "answer": sample.answer,
            "source_docs": sample.source_docs,
            "source_files": sample.source_files,
        },
        "metadata": {
            "uid": sample.uid,
            "difficulty": sample.difficulty,
        },
    }


def export_and_upload(
    *,
    dataset_name: str = "officeqa-pro",
    agent_key: str | None = None,
    train_limit: int | None = None,
    test_limit: int | None = None,
) -> dict[str, Any]:
    train_samples = load_split("train")
    test_samples = load_split("test")

    if train_limit is not None:
        train_samples = train_samples[:train_limit]
    if test_limit is not None:
        test_samples = test_samples[:test_limit]

    train_rows = [build_row(s) for s in train_samples]
    test_rows = [build_row(s) for s in test_samples]

    with tempfile.TemporaryDirectory(prefix="beaker-dataset-") as temp_dir:
        dataset_dir = Path(temp_dir)
        splits = {"train": train_rows, "test": test_rows}

        for split_name, rows in splits.items():
            split_path = dataset_dir / f"{split_name}.jsonl"
            with split_path.open("w", encoding="utf-8") as output:
                for row in rows:
                    output.write(json.dumps(row) + "\n")

        cmd = [
            "uv",
            "run",
            "beaker",
            "dataset",
            "upload",
            str(dataset_dir),
            "--name",
            dataset_name,
            "--total-count",
            str(sum(len(rows) for rows in splits.values())),
            "--split",
            f"train={len(train_rows)}",
            "--split",
            f"test={len(test_rows)}",
            "--json",
        ]
        if agent_key:
            cmd.extend(["--agent", agent_key])

        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        artifact = json.loads(result.stdout)
        dataset_ref = f"{artifact['artifact_key']}@{artifact['dataset_revision']}"
        print(f"Uploaded dataset: {dataset_ref}")
        return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and upload OfficeQA Pro dataset to Beaker")
    parser.add_argument("--name", default="officeqa-pro", help="Dataset name in Beaker")
    parser.add_argument("--agent", default=None, help="Agent key")
    parser.add_argument("--train-limit", type=int, default=None, help="Optional limit on train samples")
    parser.add_argument("--test-limit", type=int, default=None, help="Optional limit on test samples")
    args = parser.parse_args()

    export_and_upload(
        dataset_name=args.name,
        agent_key=args.agent,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
    )


if __name__ == "__main__":
    main()

