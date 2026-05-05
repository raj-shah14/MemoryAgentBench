"""Compute wall-clock timings and a comparison report for two pipeline runs.

Given two AML pipeline job names (typically a sequential baseline and a
parallel variant), this script pulls the root + child job timings, prints
a side-by-side table, and writes the report to disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from azure.ai.ml import MLClient
from azure.identity import AzureCliCredential


DEFAULT_SUBSCRIPTION_ID = "c7d74d79-1ca2-4d95-a534-783b00cbf117"
DEFAULT_RESOURCE_GROUP = "shahra-rg"
DEFAULT_WORKSPACE = "shahra-workspace"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequential", required=True, help="Pipeline job name of the sequential baseline.")
    parser.add_argument("--parallel", required=True, help="Pipeline job name of the parallel variant.")
    parser.add_argument("--subscription_id", default=DEFAULT_SUBSCRIPTION_ID)
    parser.add_argument("--resource_group", default=DEFAULT_RESOURCE_GROUP)
    parser.add_argument("--workspace_name", default=DEFAULT_WORKSPACE)
    parser.add_argument(
        "--report",
        default="aml/runtime_comparison_report.json",
        help="Output path for the JSON report.",
    )
    return parser.parse_args()


def parse_iso(value) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


@dataclass
class JobTiming:
    name: str
    display_name: str
    status: str
    started: Optional[datetime] = None
    ended: Optional[datetime] = None
    children: list = field(default_factory=list)

    @property
    def duration_s(self) -> Optional[float]:
        if self.started and self.ended:
            return (self.ended - self.started).total_seconds()
        return None


def extract_execution_window(props: dict):
    """AML records StartTime/EndTime inside properties['azureml.pipelines.stages'] as a JSON blob."""
    stages_raw = (props or {}).get("azureml.pipelines.stages")
    if not stages_raw:
        return None, None
    try:
        stages = json.loads(stages_raw) if isinstance(stages_raw, str) else stages_raw
    except (TypeError, ValueError):
        return None, None
    execution = (stages or {}).get("Execution") or {}
    return parse_iso(execution.get("StartTime")), parse_iso(execution.get("EndTime"))


def extract_child_window(props: dict, child=None):
    """Child command jobs don't expose StartTimeUtc; fall back to creation_context fields."""
    props = props or {}
    started = (
        parse_iso(props.get("StartTimeUtc"))
        or parse_iso(props.get("startTimeUtc"))
        or parse_iso(props.get("startTime"))
    )
    ended = (
        parse_iso(props.get("EndTimeUtc"))
        or parse_iso(props.get("endTimeUtc"))
        or parse_iso(props.get("endTime"))
    )
    if started is None or ended is None:
        # Fall back to the pipeline-stage shape in case AML chose that representation.
        s2, e2 = extract_execution_window(props)
        started = started or s2
        ended = ended or e2
    if (started is None or ended is None) and child is not None:
        cc = getattr(child, "creation_context", None)
        if cc is not None:
            if started is None:
                started = parse_iso(getattr(cc, "created_at", None))
            if ended is None:
                ended = parse_iso(getattr(cc, "last_modified_at", None))
    return started, ended


def fetch_timings(ml_client: MLClient, job_name: str) -> JobTiming:
    job = ml_client.jobs.get(job_name)
    props = getattr(job, "properties", {}) or {}
    started, ended = extract_execution_window(props)
    timing = JobTiming(
        name=job_name,
        display_name=getattr(job, "display_name", "") or "",
        status=getattr(job, "status", "") or "",
        started=started,
        ended=ended,
    )

    for child in ml_client.jobs.list(parent_job_name=job_name):
        cprops = getattr(child, "properties", {}) or {}
        c_started, c_ended = extract_child_window(cprops, child=child)
        child_timing = JobTiming(
            name=child.name,
            display_name=getattr(child, "display_name", "") or "",
            status=getattr(child, "status", "") or "",
            started=c_started,
            ended=c_ended,
        )
        timing.children.append(child_timing)
    return timing


def fmt_dt(dt: Optional[datetime]) -> str:
    return dt.isoformat() if dt else "n/a"


def fmt_dur(seconds: Optional[float]) -> str:
    if seconds is None:
        return "n/a"
    return f"{seconds:7.1f}s"


def render_table(label: str, timing: JobTiming) -> str:
    lines = [f"=== {label}: {timing.name} ({timing.status}) {timing.display_name} ==="]
    lines.append(f"  root duration : {fmt_dur(timing.duration_s)}  [{fmt_dt(timing.started)} -> {fmt_dt(timing.ended)}]")
    lines.append("  children:")
    for child in sorted(timing.children, key=lambda c: c.started or datetime.min.replace(tzinfo=timezone.utc)):
        lines.append(
            f"    {child.display_name:<28} status={child.status:<10}"
            f" duration={fmt_dur(child.duration_s)}"
            f"  [{fmt_dt(child.started)} -> {fmt_dt(child.ended)}]"
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    ml_client = MLClient(
        credential=AzureCliCredential(),
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace_name,
    )

    seq = fetch_timings(ml_client, args.sequential)
    par = fetch_timings(ml_client, args.parallel)

    print(render_table("SEQUENTIAL", seq))
    print()
    print(render_table("PARALLEL  ", par))
    print()

    seq_dur = seq.duration_s
    par_dur = par.duration_s
    speedup = seq_dur / par_dur if seq_dur and par_dur else None

    print("=== Comparison ===")
    print(f"  sequential wall : {fmt_dur(seq_dur)}")
    print(f"  parallel   wall : {fmt_dur(par_dur)}")
    if speedup is not None:
        print(f"  speedup         : {speedup:.2f}x  (saved {seq_dur - par_dur:.1f}s)")

    payload = {
        "sequential": {
            "name": seq.name,
            "status": seq.status,
            "display_name": seq.display_name,
            "duration_seconds": seq_dur,
            "started": fmt_dt(seq.started),
            "ended": fmt_dt(seq.ended),
            "children": [
                {
                    "name": c.name,
                    "display_name": c.display_name,
                    "status": c.status,
                    "duration_seconds": c.duration_s,
                    "started": fmt_dt(c.started),
                    "ended": fmt_dt(c.ended),
                }
                for c in seq.children
            ],
        },
        "parallel": {
            "name": par.name,
            "status": par.status,
            "display_name": par.display_name,
            "duration_seconds": par_dur,
            "started": fmt_dt(par.started),
            "ended": fmt_dt(par.ended),
            "children": [
                {
                    "name": c.name,
                    "display_name": c.display_name,
                    "status": c.status,
                    "duration_seconds": c.duration_s,
                    "started": fmt_dt(c.started),
                    "ended": fmt_dt(c.ended),
                }
                for c in par.children
            ],
        },
        "comparison": {
            "speedup": speedup,
            "absolute_savings_seconds": (seq_dur - par_dur) if (seq_dur and par_dur) else None,
        },
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nreport written to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
