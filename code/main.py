"""
main.py — Entry point for the Multi-Domain Support Triage Agent.

Usage:
    python main.py
    python main.py --input ../support_tickets/support_tickets.csv \\
                   --output ../support_tickets/output.csv
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
import traceback
from typing import List, Dict

import anthropic
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich import print as rprint

from utils import (
    load_env,
    read_tickets,
    write_output,
    INPUT_CSV,
    OUTPUT_CSV,
    OUTPUT_COLUMNS,
)
from retriever import build_retriever
from agent import TriageAgent

console = Console()


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-Domain Support Triage Agent (HackerRank Orchestrate)",
    )
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        default=INPUT_CSV,
        help=f"Path to input CSV (default: {INPUT_CSV})",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=OUTPUT_CSV,
        help=f"Path to output CSV (default: {OUTPUT_CSV})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process tickets but do not write output.csv",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(input_path: pathlib.Path, output_path: pathlib.Path, dry_run: bool = False) -> None:
    start_total = time.time()

    # --- Load API key ---
    console.rule("[bold]Step 1/4: Loading environment[/bold]")
    api_key = load_env()
    client = anthropic.Anthropic(api_key=api_key)
    console.print("[green]✓ ANTHROPIC_API_KEY loaded[/green]")

    # --- Load and validate input CSV ---
    console.rule("[bold]Step 2/4: Loading input tickets[/bold]")
    if not input_path.exists():
        console.print(f"[red]ERROR: Input file not found: {input_path}[/red]")
        sys.exit(1)

    df = read_tickets(input_path)
    console.print(f"[green]✓ Loaded {len(df)} tickets from {input_path}[/green]")

    # Validate required columns
    required = {"issue", "subject", "company"}
    missing = required - set(df.columns)
    if missing:
        console.print(f"[red]ERROR: Input CSV missing columns: {missing}[/red]")
        sys.exit(1)

    # --- Build retrieval index ---
    console.rule("[bold]Step 3/4: Building retrieval index[/bold]")
    retriever = build_retriever()

    # --- Triage all tickets ---
    console.rule("[bold]Step 4/4: Triaging tickets[/bold]")
    agent = TriageAgent(retriever=retriever, client=client)

    results: List[Dict[str, str]] = []
    errors: List[tuple[int, str]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Triaging…", total=len(df))

        for idx, row in df.iterrows():
            row_num = int(idx) + 1  # type: ignore[arg-type]
            subject = str(row.get("subject", ""))[:60]
            progress.update(task, description=f"[{row_num}/{len(df)}] {subject!r}")

            try:
                t0 = time.time()
                result = agent.triage(row.to_dict())
                elapsed = time.time() - t0

                row_dict = result.to_dict()
                row_dict["issue"] = str(row.get("issue", ""))
                row_dict["subject"] = str(row.get("subject", ""))
                row_dict["company"] = str(row.get("company", ""))
                results.append(row_dict)

                status_color = "green" if result.status == "replied" else "yellow"
                console.print(
                    f"  [{status_color}]{result.status.upper()}[/{status_color}] "
                    f"[dim]{result.request_type}[/dim] "
                    f"[cyan]{result.product_area or '(no area)'}[/cyan] "
                    f"[dim]({elapsed:.1f}s)[/dim]"
                )

            except Exception as e:
                errors.append((row_num, str(e)))
                console.print(f"  [red]ERROR row {row_num}: {e}[/red]")
                traceback.print_exc()
                # Still append a safe fallback so CSV row count matches
                results.append({
                    "issue": str(row.get("issue", "")),
                    "subject": str(row.get("subject", "")),
                    "company": str(row.get("company", "")),
                    "status": "escalated",
                    "product_area": "",
                    "response": "Your ticket has been escalated to our support team for manual review.",
                    "justification": f"Processing error: {e}",
                    "request_type": "product_issue",
                })

            progress.advance(task)

    # --- Write output ---
    if not dry_run:
        write_output(results, output_path)
        console.print(f"\n[bold green]✓ Output written to {output_path}[/bold green]")
    else:
        console.print("\n[yellow]Dry run: output NOT written.[/yellow]")

    # --- Summary table ---
    total_elapsed = time.time() - start_total
    _print_summary(results, errors, total_elapsed)


def _print_summary(
    results: List[Dict[str, str]],
    errors: List[tuple[int, str]],
    elapsed: float,
) -> None:
    replied = sum(1 for r in results if r["status"] == "replied")
    escalated = sum(1 for r in results if r["status"] == "escalated")

    table = Table(title="Triage Summary", show_header=True, header_style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Total tickets", str(len(results)))
    table.add_row("Replied", f"[green]{replied}[/green]")
    table.add_row("Escalated", f"[yellow]{escalated}[/yellow]")
    table.add_row("Processing errors", f"[red]{len(errors)}[/red]" if errors else "[green]0[/green]")
    table.add_row("Total time", f"{elapsed:.1f}s")
    table.add_row("Avg per ticket", f"{elapsed/max(len(results),1):.1f}s")

    console.print()
    console.print(table)

    if errors:
        console.print("\n[bold red]Errors:[/bold red]")
        for row_num, msg in errors:
            console.print(f"  Row {row_num}: {msg}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    run(
        input_path=args.input,
        output_path=args.output,
        dry_run=args.dry_run,
    )
