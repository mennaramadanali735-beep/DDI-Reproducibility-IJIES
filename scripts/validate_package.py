"""Validate source syntax, release hygiene and saved topology summaries without training."""
import ast
import csv
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    errors = []
    cells = 0
    notebooks = sorted(ROOT.rglob('*.ipynb'))
    for path in notebooks:
        notebook = json.loads(path.read_text())
        assert notebook['nbformat'] == 4
        for index, cell in enumerate(notebook['cells']):
            source = ''.join(cell['source'])
            if re.search(r'[\u0600-\u06ff]', source):
                errors.append(f'{path.name}:{index}: Arabic source text remains')
            if cell['cell_type'] != 'code':
                continue
            cells += 1
            if cell.get('outputs') or cell.get('execution_count') is not None:
                errors.append(f'{path.name}:{index}: saved execution state remains')
            # Installation magics are valid in Colab, but not ordinary Python syntax.
            source = re.sub(r'(?m)^(?:!pip |%pip ).*$', 'pass', source)
            try:
                ast.parse(source)
            except SyntaxError as exc:
                errors.append(f'{path.name}:{index}: {exc}')
    scripts = sorted(ROOT.rglob('*.py'))
    for path in scripts:
        try:
            ast.parse(path.read_text())
        except SyntaxError as exc:
            errors.append(f'{path.name}: {exc}')

    with (ROOT / 'evidence/topology__TEST_RESULTS_9_RUNS.csv').open() as handle:
        runs = list(csv.DictReader(handle))
    with (ROOT / 'evidence/topology__TEST_SUMMARY_MEAN_SAMPLE_SD.csv').open() as handle:
        summary = list(csv.DictReader(handle))
    assert len(runs) == 9
    for row in summary:
        group = [r for r in runs if r['condition'] == row['condition']]
        assert sorted(int(r['seed']) for r in group) == [42, 43, 44]
        for metric in ['accuracy', 'macro_f1', 'weighted_f1']:
            values = [float(r[metric]) for r in group]
            assert abs(statistics.mean(values) - float(row[metric + '_mean'])) < 1e-12
            assert abs(statistics.stdev(values) - float(row[metric + '_std'])) < 1e-12

    with (ROOT / 'evidence/topology__CALIBRATION_REPRODUCTION.csv').open() as handle:
        calibration = list(csv.DictReader(handle))
    assert len(calibration) == 9
    for row in calibration:
        for key, value in row.items():
            if key.startswith('difference_'):
                assert abs(float(value)) <= 1e-6

    report = {
        'notebooks_checked': len(notebooks),
        'code_cells_checked': cells,
        'python_files_checked': len(scripts),
        'syntax_errors': errors,
        'saved_topology_summary_recomputed': True,
        'recorded_calibration_differences_checked': True,
        'training_executed': False,
        'checkpoint_inference_executed': False,
        'end_to_end_reproduction_executed': False,
    }
    output = ROOT / 'docs/validation_report.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
