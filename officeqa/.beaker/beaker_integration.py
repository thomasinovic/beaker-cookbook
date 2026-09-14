"""Beaker repository integration for officeqa.

Keep evaluation tooling in .beaker/.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path

from beaker import (
    Case,
    CaseResult,
    CaseScore,
    Check,
    Integration,
    JsonValue,
    RepositoryRunSetup,
    RolloutRuntime,
    SetupRuntime,
    inference_target,
    repository,
)
from pydantic import BaseModel, Field

from officeqa import config
from officeqa.data.corpus import ensure_corpus, fetch_corpus
from officeqa.data.dataset import EvalRecord
from officeqa.evaluation.reward import score_answer
from officeqa.runner import run_one_async


class QuestionInput(BaseModel):
    uid: str = Field(min_length=1)
    question: str
    source_files: str = ""
    difficulty: str = ""


class QuestionExpected(BaseModel):
    answer: str
    source_docs: str = ""
    source_files: str = ""


class Row(BaseModel):
    id: str = Field(min_length=1)
    input: QuestionInput
    expected: QuestionExpected
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class Setup(RepositoryRunSetup[Row]):
    row_model = Row

    async def load_cases(self, row: Row, *, runtime: SetupRuntime) -> AsyncIterator[Case]:
        del runtime
        yield Case(
            id=row.id,
            input=row.input.model_dump(),
            expected=row.expected.model_dump(),
            metadata=row.metadata,
        )


logger = logging.getLogger(__name__)

_CORPUS_LOCK = asyncio.Lock()


async def _ensure_corpus_ready(representation: str = config.DEFAULT_CORPUS) -> Path:
    """Ensure the required corpus representation is present, fetching it if missing."""
    try:
        return ensure_corpus(representation)
    except FileNotFoundError:
        pass

    async with _CORPUS_LOCK:
        try:
            return ensure_corpus(representation)
        except FileNotFoundError:
            logger.info("Corpus representation %r not found; fetching from Hugging Face...", representation)
            return await asyncio.to_thread(fetch_corpus, (representation,))


async def run_case(*, case_input: JsonValue, runtime: RolloutRuntime) -> CaseResult:
    inp = case_input if isinstance(case_input, dict) else QuestionInput.model_validate(case_input).model_dump()
    uid = str(inp.get("uid") or "")
    question = str(inp.get("question") or "")
    source_files = str(inp.get("source_files") or "")
    difficulty = str(inp.get("difficulty") or "")

    sample = EvalRecord(
        uid=uid,
        question=question,
        answer="",  # Ground truth is kept in Case.expected and not exposed to the agent
        source_docs="",
        source_files=source_files,
        difficulty=difficulty,
    )

    model_name = config.DEFAULT_MODEL
    if runtime.model:
        try:
            target = inference_target(runtime)
            model_name = target.model.split(":", 1)[1] if ":" in target.model else target.model
        except Exception:
            model_name = runtime.model.split(":", 1)[1] if ":" in runtime.model else runtime.model

    corpus_root = await _ensure_corpus_ready(config.DEFAULT_CORPUS)

    with runtime.trace.stage("officeqa.run_case", inputs={"uid": uid, "question": question}) as stage:
        try:
            # Bound case runtime (480s / 40 steps) to prevent candidate hangs
            # and guarantee cases complete well before the 900s candidate watchdog.
            res = await asyncio.wait_for(
                run_one_async(
                    sample,
                    model=model_name,
                    corpus_root=corpus_root,
                    task_timeout_s=480.0,
                    max_steps=40,
                ),
                timeout=540.0,
            )
            output = {
                "final_answer": res.final_answer,
                "status": res.status,
                "steps": res.steps,
                "tool_calls": res.tool_calls,
                "cost_usd": res.cost_usd,
                "latency_s": res.latency_s,
                "error": res.error,
            }
        except asyncio.TimeoutError:
            output = {
                "final_answer": None,
                "status": "timeout",
                "steps": 0,
                "tool_calls": 0,
                "cost_usd": None,
                "latency_s": 540.0,
                "error": "Case timed out after 540s",
            }
        except Exception as exc:
            output = {
                "final_answer": None,
                "status": "error",
                "steps": 0,
                "tool_calls": 0,
                "cost_usd": None,
                "latency_s": 0.0,
                "error": f"{type(exc).__name__}: {exc}",
            }
        stage.output(output)

    return CaseResult(output=output, output_kind="record")


async def score_case(*, case: Case, result: CaseResult, case_files_dir: Path) -> CaseScore:
    del case_files_dir
    exp = case.expected if isinstance(case.expected, dict) else {}
    ground_truth = str(exp.get("answer") or "")

    out = result.output if isinstance(result.output, dict) else {}
    final_answer = out.get("final_answer")
    if final_answer is not None:
        final_answer = str(final_answer)

    scores: dict[str, float] = {}
    for t in config.TOLERANCES:
        if not final_answer or not final_answer.strip():
            scores[str(t)] = 0.0
        else:
            try:
                scores[str(t)] = float(score_answer(ground_truth, final_answer, t))
            except Exception:
                scores[str(t)] = 0.0

    headline = scores.get(str(config.HEADLINE_TOLERANCE), 0.0)

    checks = (
        Check(
            name="Headline correctness (0.0% tolerance)",
            verdict="pass" if headline == 1.0 else "fail",
            expected=ground_truth,
            predicted=final_answer or "(none)",
            message=(
                f"Answer evaluated against {ground_truth!r} at 0.0% tolerance"
                if headline == 1.0
                else f"Status: {out.get('status', 'unknown')}, error: {out.get('error')}"
            ),
        ),
        Check(
            name="Tolerance 0.1%",
            verdict="pass" if scores.get("0.001", 0.0) == 1.0 else "fail",
            informational=True,
            expected=ground_truth,
            predicted=final_answer or "(none)",
        ),
        Check(
            name="Tolerance 1.0%",
            verdict="pass" if scores.get("0.01", 0.0) == 1.0 else "fail",
            informational=True,
            expected=ground_truth,
            predicted=final_answer or "(none)",
        ),
        Check(
            name="Tolerance 5.0%",
            verdict="pass" if scores.get("0.05", 0.0) == 1.0 else "fail",
            informational=True,
            expected=ground_truth,
            predicted=final_answer or "(none)",
        ),
    )

    field_scores = {
        "correctness": headline,
        "tol_0.001": scores.get("0.001", 0.0),
        "tol_0.01": scores.get("0.01", 0.0),
        "tol_0.05": scores.get("0.05", 0.0),
    }

    return CaseScore(
        objective=headline,
        field_scores=field_scores,
        checks=checks,
    )


integration = Integration(
    targets=repository(("src/officeqa/agent",)),
    run_setup=Setup,
    run_case=run_case,
    score_case=score_case,
)
