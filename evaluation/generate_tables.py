"""
Generate results tables in CSV, Markdown, and Excel formats.

Usage:
    python evaluation/generate_tables.py
    python evaluation/generate_tables.py --results-dir results/ --output-dir results/
"""
import os
import sys
import json
import csv
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.compare import load_all_results, build_comparison_table, format_comparison_markdown


def generate_csv(rows, output_path):
    """Generate results CSV file."""
    if not rows:
        print("No results to write.")
        return

    fieldnames = ['model', 'dataset', 'seq_len', 'pred_len',
                  'mse', 'mae', 'rmse', 'mape', 'smape',
                  'params', 'train_time_min']

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"  CSV:      {output_path}")


def generate_markdown(rows, output_path):
    """Generate results Markdown table."""
    md = format_comparison_markdown(rows)

    header = f"# TimeMixer Experiment Results\n\n"
    header += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Paper comparison table
    paper_values = {
        ('Traffic', 96): {'mse': 0.462, 'mae': 0.285},
        ('Traffic', 192): {'mse': 0.474, 'mae': 0.292},
        ('Traffic', 336): {'mse': 0.494, 'mae': 0.302},
        ('Traffic', 720): {'mse': 0.529, 'mae': 0.322},
        ('Weather', 96): {'mse': 0.164, 'mae': 0.214},
        ('Weather', 192): {'mse': 0.210, 'mae': 0.257},
        ('Weather', 336): {'mse': 0.265, 'mae': 0.298},
        ('Weather', 720): {'mse': 0.338, 'mae': 0.348},
    }

    paper_section = "## Paper Reference Values\n\n"
    paper_section += "| Dataset | Horizon | Paper MSE | Paper MAE |\n"
    paper_section += "|---------|---------|-----------|----------|\n"
    for (ds, h), vals in sorted(paper_values.items()):
        paper_section += f"| {ds} | {h} | {vals['mse']:.3f} | {vals['mae']:.3f} |\n"

    full_content = header + "## Our Results\n\n" + md + "\n\n" + paper_section

    with open(output_path, 'w') as f:
        f.write(full_content)

    print(f"  Markdown: {output_path}")


def generate_latex(rows, output_path):
    """Generate LaTeX table for report."""
    if not rows:
        return

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Forecasting Results}")
    lines.append(r"\begin{tabular}{llccccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & Dataset & Horizon & MSE & MAE & RMSE \\")
    lines.append(r"\midrule")

    for r in rows:
        mse_str = f"{r['mse']:.4f}" if isinstance(r['mse'], float) else str(r['mse'])
        mae_str = f"{r['mae']:.4f}" if isinstance(r['mae'], float) else str(r['mae'])
        rmse_str = f"{r['rmse']:.4f}" if isinstance(r['rmse'], float) else str(r['rmse'])
        lines.append(f"{r['model']} & {r['dataset']} & {r['pred_len']} & "
                     f"{mse_str} & {mae_str} & {rmse_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  LaTeX:    {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate result tables')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory containing metrics.json files')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for generated files')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading results from: {args.results_dir}")
    results = load_all_results(args.results_dir)
    rows = build_comparison_table(results)
    print(f"Found {len(rows)} experiment results.\n")

    print("Generating tables:")
    generate_csv(rows, os.path.join(args.output_dir, 'results.csv'))
    generate_markdown(rows, os.path.join(args.output_dir, 'results.md'))
    generate_latex(rows, os.path.join(args.output_dir, 'results.tex'))

    print(f"\nDone! All tables saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
