"""
evaluate.py — Self-evaluation script.

Compares agent predictions against the labelled sample_support_tickets.csv.

Two modes:
  1. --generate  (default): Runs main.py on the sample tickets first, writes
     support_tickets/sample_output.csv, then scores it against the sample labels.
  2. --no-generate: Skip the pipeline run; just score an already-generated
     sample_output.csv (useful for quick iteration without re-calling the API).

Usage (from repo root):
    python code/evaluate.py                # generate + score
    python code/evaluate.py --no-generate  # score only (must already have sample_output.csv)

    # Override paths:
    python code/evaluate.py \\
        --sample  support_tickets/sample_support_tickets.csv \\
        --output  support_tickets/sample_output.csv
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

# Repo root is one level above this file
_REPO_ROOT = pathlib.Path(__file__).parent.parent.resolve()
_CODE_DIR = pathlib.Path(__file__).parent.resolve()
_DEFAULT_SAMPLE = _REPO_ROOT / "support_tickets" / "sample_support_tickets.csv"
_DEFAULT_OUTPUT = _REPO_ROOT / "support_tickets" / "sample_output.csv"

console = Console()


def _normalise(value: str) -> str:
    """Lowercase and strip for comparison."""
    return str(value).strip().lower()


def _load_csv(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(str(path), encoding="utf-8", encoding_errors="replace")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def generate_predictions(sample_path: pathlib.Path, output_path: pathlib.Path) -> bool:
    """Run main.py on the sample tickets and write predictions to output_path."""
    console.print(f"[bold cyan]Running pipeline on sample tickets…[/bold cyan]")
    console.print(f"  Input:  {sample_path}")
    console.print(f"  Output: {output_path}")

    cmd = [
        sys.executable,
        str(_CODE_DIR / "main.py"),
        "--input", str(sample_path),
        "--output", str(output_path),
    ]
    result = subprocess.run(cmd, cwd=str(_REPO_ROOT))
    if result.returncode != 0:
        console.print("[red]Pipeline failed. Check errors above.[/red]")
        return False
    return True


def evaluate(output_path: pathlib.Path, sample_path: pathlib.Path) -> int:
    """
    Compare predictions in output_path against reference labels in sample_path.
    Returns exit code: 0 if ≥70% accuracy, 1 otherwise.
    Matching is done by row position (both files must cover the same tickets in order).
    """
    if not output_path.exists():
        console.print(f"[red]Output file not found: {output_path}[/red]")
        console.print("Run with --generate (or omit --no-generate) to create it first.")
        return 1

    if not sample_path.exists():
        console.print(f"[red]Sample file not found: {sample_path}[/red]")
        return 1

    output_df = _load_csv(output_path)
    sample_df = _load_csv(sample_path)

    n_sample = len(sample_df)
    n_output = len(output_df)

    if n_output < n_sample:
        console.print(
            f"[yellow]Warning: output has {n_output} rows but sample has "
            f"{n_sample}. Comparing only {n_output}.[/yellow]"
        )
        n_sample = n_output

    results = []
    status_correct = 0
    rt_correct = 0
    area_correct = 0

    for i in range(n_sample):
        out_row = output_df.iloc[i]
        smp_row = sample_df.iloc[i]

        out_status = _normalise(out_row.get("status", ""))
        smp_status = _normalise(smp_row.get("status", ""))

        out_rt = _normalise(out_row.get("request_type", ""))
        smp_rt = _normalise(smp_row.get("request_type", ""))

        out_area = _normalise(out_row.get("product_area", ""))
        smp_area = _normalise(smp_row.get("product_area", ""))

        s_match = out_status == smp_status
        r_match = out_rt == smp_rt
        # Product area: partial match (either direction) counts
        a_match = bool(
            out_area == smp_area
            or (out_area and smp_area and (out_area in smp_area or smp_area in out_area))
        )

        if s_match:
            status_correct += 1
        if r_match:
            rt_correct += 1
        if a_match:
            area_correct += 1

        subject = ""
        for col in ("subject", "issue"):
            if col in smp_row.index:
                subject = str(smp_row[col])[:40]
                break

        results.append({
            "row": i,
            "subject": subject,
            "out_status": out_status,
            "smp_status": smp_status,
            "status_ok": s_match,
            "out_rt": out_rt,
            "smp_rt": smp_rt,
            "rt_ok": r_match,
            "out_area": out_area[:25],
            "smp_area": smp_area[:25],
            "area_ok": a_match,
        })

    # ---- Summary table ----
    console.print()
    console.rule("[bold cyan]Evaluation Report[/bold cyan]")
    console.print(f"  Predictions: {output_path}")
    console.print(f"  Reference:   {sample_path}")
    console.print(f"  Rows compared: {n_sample}")
    console.print()

    table = Table(
        title="Row-by-Row Comparison",
        box=box.SIMPLE_HEAD,
        show_lines=True,
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Subject / Issue", max_width=35)
    table.add_column("Status (pred)", style="cyan")
    table.add_column("Status (ref)", style="cyan")
    table.add_column("✓", width=3)
    table.add_column("Type (pred)", style="magenta")
    table.add_column("Type (ref)", style="magenta")
    table.add_column("✓", width=3)
    table.add_column("Area (pred)", style="yellow")
    table.add_column("Area (ref)", style="yellow")
    table.add_column("✓", width=3)

    for r in results:
        table.add_row(
            str(r["row"]),
            r["subject"],
            r["out_status"],
            r["smp_status"],
            "[green]✓[/green]" if r["status_ok"] else "[red]✗[/red]",
            r["out_rt"],
            r["smp_rt"],
            "[green]✓[/green]" if r["rt_ok"] else "[red]✗[/red]",
            r["out_area"],
            r["smp_area"],
            "[green]✓[/green]" if r["area_ok"] else "[red]✗[/red]",
        )

    console.print(table)

    # ---- Score summary ----
    score_table = Table(box=box.SIMPLE, title="Accuracy Summary")
    score_table.add_column("Metric", style="bold")
    score_table.add_column("Correct", justify="right")
    score_table.add_column("Total", justify="right")
    score_table.add_column("Accuracy", justify="right")

    for label, correct in [
        ("status", status_correct),
        ("request_type", rt_correct),
        ("product_area (partial match)", area_correct),
    ]:
        pct = correct / n_sample * 100
        colour = "green" if pct >= 70 else ("yellow" if pct >= 50 else "red")
        score_table.add_row(label, str(correct), str(n_sample), f"[{colour}]{pct:.0f}%[/{colour}]")

    overall = (status_correct + rt_correct) / (n_sample * 2) * 100
    o_colour = "green" if overall >= 70 else ("yellow" if overall >= 50 else "red")
    score_table.add_row(
        "[bold]Overall (status + type)[/bold]",
        str(status_correct + rt_correct),
        str(n_sample * 2),
        f"[{o_colour}][bold]{overall:.0f}%[/bold][/{o_colour}]",
    )
    console.print(score_table)

    # ---- Mismatch detail ----
    mismatches = [r for r in results if not r["status_ok"] or not r["rt_ok"]]
    if mismatches:
        console.print("[bold yellow]Mismatch Detail[/bold yellow]")
        for r in mismatches:
            flags = []
            if not r["status_ok"]:
                flags.append(
                    f"status: pred=[cyan]{r['out_status']}[/cyan] ref=[cyan]{r['smp_status']}[/cyan]"
                )
            if not r["rt_ok"]:
                flags.append(
                    f"request_type: pred=[magenta]{r['out_rt']}[/magenta] ref=[magenta]{r['smp_rt']}[/magenta]"
                )
            console.print(f"  Row {r['row']} ({r['subject']}): {' | '.join(flags)}")
        console.print()
    else:
        console.print("[bold green]✓ All rows match on status and request_type![/bold green]\n")

    return 0 if overall >= 70 else 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate agent output against the labelled sample CSV."
    )
    p.add_argument(
        "--sample",
        default=str(_DEFAULT_SAMPLE),
        help="Path to sample_support_tickets.csv with reference labels (default: %(default)s)",
    )
    p.add_argument(
        "--output",
        default=str(_DEFAULT_OUTPUT),
        help="Path where sample predictions are/will be written (default: %(default)s)",
    )
    p.add_argument(
        "--no-generate",
        action="store_true",
        help="Skip running main.py; score an already-generated output file instead",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sample_path = pathlib.Path(args.sample)
    output_path = pathlib.Path(args.output)

    if not args.no_generate:
        ok = generate_predictions(sample_path, output_path)
        if not ok:
            sys.exit(1)

    code = evaluate(output_path, sample_path)
    sys.exit(code)

