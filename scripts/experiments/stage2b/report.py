"""Automated Stage 2B readiness report (measured values kept apart from estimates)."""
from datetime import datetime
from pathlib import Path
import re

from experiments.stage2b.common import PRIMARY, file_record, load_json, utc_now, write_json

LINE = re.compile(r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+)\] \w+: (.*)$')
PROCESSED = re.compile(r'Processed (\d+) entries in total \(avg\. speed ([0-9.]+) entries/s\)')
# Production train800k workers (windowed/fixed-LR logs): accepted rows per second during training.
REFERENCE_THROUGHPUT = (8188.0, 13032.0)
FULL_EPOCH_ACCEPTED = 21_512_821


def log_timings(path, epoch):
    """Timings of the last Weaver process in an appended per-generation log."""
    lines = Path(path).read_text(errors='replace').splitlines()
    starts = [i for i, line in enumerate(lines) if line.endswith('INFO: args:')]
    events = []
    for line in lines[starts[-1] if starts else 0:]:
        match = LINE.match(line)
        if match:
            events.append((datetime.strptime(match[1], '%Y-%m-%d %H:%M:%S,%f'), re.sub(r'\x1b\[[0-9;]*m', '', match[2])))
    if not events:
        return None
    first = events[0][0]
    marks = {text: stamp for stamp, text in events if text in (f'Epoch #{epoch} training', f'Epoch #{epoch} validating')}
    processed = [(stamp, PROCESSED.search(text)) for stamp, text in events if PROCESSED.search(text)]
    timings = dict(process_seconds=(events[-1][0] - first).total_seconds())
    if f'Epoch #{epoch} training' in marks:
        timings['startup_seconds'] = (marks[f'Epoch #{epoch} training'] - first).total_seconds()
    if processed:
        timings.update(train_entries=int(processed[0][1][1]), train_rate=float(processed[0][1][2]))
        if f'Epoch #{epoch} training' in marks:
            timings['train_seconds'] = (processed[0][0] - marks[f'Epoch #{epoch} training']).total_seconds()
    if len(processed) > 1 and f'Epoch #{epoch} validating' in marks:
        timings.update(validation_entries=int(processed[1][1][1]), validation_rate=float(processed[1][1][2]),
                       validation_seconds=(processed[1][0] - marks[f'Epoch #{epoch} validating']).total_seconds())
    return timings


def smoke_measurements(run):
    manifest = load_json(Path(run) / 'manifest.json')
    rows = []
    for generation in manifest['generations']:
        for name, worker in generation['workers'].items():
            timing = log_timings(worker['log'], generation['epoch']) if Path(worker.get('log', '')).is_file() else None
            metrics = worker.get('metrics') or {}
            rows.append(dict(generation=generation['index'], epoch=generation['epoch'], member=name, lr=worker.get('lr'),
                             primary=metrics.get(PRIMARY), cuda_mb=metrics.get('train_max_cuda_memory_mb'),
                             amp_skipped=metrics.get('train_amp_skipped_optimizer_steps'), **(timing or {})))
    return rows


def estimates(rows):
    """Explicit extrapolations for one full train8m epoch per member (not measurements)."""
    startup = max((r.get('startup_seconds') or 0 for r in rows), default=0)
    validation = max((r.get('validation_seconds') or 0 for r in rows), default=0)
    low, high = (FULL_EPOCH_ACCEPTED / rate / 60 for rate in reversed(REFERENCE_THROUGHPUT))
    per_epoch = (low + (startup + validation) / 60, high + (startup + validation) / 60)
    return dict(full_epoch_accepted_rows=FULL_EPOCH_ACCEPTED, reference_throughput_rows_per_second=REFERENCE_THROUGHPUT,
                training_minutes=(low, high), measured_startup_seconds=startup, measured_validation_seconds=validation,
                member_epoch_minutes=per_epoch,
                stage3_draft_gpu_hours=[round(5 * 20 * m / 60, 1) for m in per_epoch],
                basis='training time = 21.5M accepted rows / production train800k throughput; '
                      'startup and validation measured in this smoke; 5 members x 20 epochs per arm')


def build(output, status, *, smoke=None, parity=None, dry_runs=None, provenance=None):
    output = Path(output)
    dataset = status.get('dataset', {})
    verification = (smoke or {}).get('verification') or {}
    observations = (smoke or {}).get('observations') or {}
    rows = smoke_measurements(smoke['run']) if smoke and Path(smoke['run'], 'manifest.json').is_file() else []
    failed = [c for c in verification.get('checks', []) if not c['passed']]
    ready = bool(verification.get('passed') and (parity or {}).get('passed')
                 and dry_runs and all(r['returncode'] == 0 and r['uses_train8m'] for r in dry_runs.values()))
    summary = dict(schema_version=1, generated_at=utc_now(), ready=ready, error=status.get('error'), git=status.get('git'),
                   spec_sha256=status.get('spec_sha256'), policy_sha256=status.get('policy_sha256'),
                   dataset=dataset, smoke=dict(attempt=(smoke or {}).get('attempt'), run=(smoke or {}).get('run'),
                                               passed=verification.get('passed'), failed_checks=failed,
                                               measured=verification.get('measured'), observations=observations,
                                               generations=rows),
                   parity=parity, stage3_dry_runs=dry_runs, estimates=estimates(rows) if rows else None,
                   provenance=provenance)
    write_json(output / 'readiness.json', summary)
    (output / 'report.md').write_text(markdown(summary, verification))
    artifacts = [file_record(output / name) for name in ('readiness.json', 'report.md')]
    write_json(output / 'report_integrity.json', dict(generated_at=utc_now(), artifacts=artifacts))
    return summary


def _fmt(value, digits=1):
    return '—' if value is None else f'{value:,.{digits}f}' if isinstance(value, float) else f'{value:,}'


def markdown(summary, verification):
    dataset, smoke = summary['dataset'], summary['smoke']
    lines = ['# Stage 2B: готовность train8m', '',
             f'**Итог: {"ГОТОВО" if summary["ready"] else "НЕ ГОТОВО"}.** Сформировано {summary["generated_at"]}; '
             f'git `{(summary.get("git") or {}).get("commit")}`; spec `{summary.get("spec_sha256")}`.', '',
             'Это smoke-проверка инфраструктуры, а не научный результат: 256 шагов на поколение не дают '
             'сравнимой метрики качества.', '']
    if summary.get('error'):
        lines += [f'**Ошибка:** `{summary["error"]}`', '']
    lines += ['## Данные train8m', '',
             f'- Состояние: `{dataset.get("state")}`; строк: {_fmt(dataset.get("rows_total"))}; '
             f'размер: {_fmt(dataset.get("size_bytes_total"))} байт.',
             f'- Weaver audit fingerprint: `{dataset.get("fingerprint")}`.',
             f'- Манифест: `{dataset.get("manifest")}`.', '', '## GPU smoke (измерено)', '',
             f'Попытка {smoke.get("attempt")}, run `{smoke.get("run")}`.', '',
             '| Поколение | Эпоха | Член | LR | Старт процесса, с | Обучение, с | Скорость, строк/с | '
             'Валидация, с | CUDA max, MB | AMP skipped | Primary, % |', '|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in smoke.get('generations') or []:
        lines.append(f'| {r["generation"]} | {r["epoch"]} | {r["member"]} | {r.get("lr")} | {_fmt(r.get("startup_seconds"))} | '
                     f'{_fmt(r.get("train_seconds"))} | {_fmt(r.get("train_rate"))} | {_fmt(r.get("validation_seconds"))} | '
                     f'{_fmt(r.get("cuda_mb"))} | {_fmt(r.get("amp_skipped"))} | {_fmt(r.get("primary"), 4)} |')
    observations = smoke.get('observations') or {}
    lines += ['', f'Пик GPU-памяти по nvidia-smi: {_fmt(observations.get("peak_gpu_memory_mib"))} MiB; '
              f'пик RSS дерева процессов: {_fmt(observations.get("peak_rss_bytes"))} байт.',
              f'Прерывание: {observations.get("interruption")}', '', '| Проверка | Результат |', '|---|---|']
    for check in verification.get('checks', []):
        lines.append(f'| {check["check"]} | {"OK" if check["passed"] else "FAIL: " + str(check.get("detail"))[:200]} |')
    parity = summary.get('parity') or {}
    lines += ['', '## Черновики Stage 3', '',
              f'Паритет конфигураций: {"OK" if parity.get("passed") else "FAIL"}; '
              f'каденс windowed: {parity.get("cadence")}.']
    for key, result in (summary.get('stage3_dry_runs') or {}).items():
        lines.append(f'- dry-run `{key}`: returncode {result["returncode"]}, команд Weaver {result["weaver_commands"]}, '
                     f'train8m: {result["uses_train8m"]}')
    estimate = summary.get('estimates')
    if estimate:
        lines += ['', '## Оценки (экстраполяция, не измерение)', '',
                  f'- Полная эпоха train8m на члена: {estimate["member_epoch_minutes"][0]:.0f}–'
                  f'{estimate["member_epoch_minutes"][1]:.0f} мин ({estimate["basis"]}).',
                  f'- GPU-часы на одну руку черновика Stage 3: {estimate["stage3_draft_gpu_hours"][0]}–'
                  f'{estimate["stage3_draft_gpu_hours"][1]}.']
    lines += ['', '## Ограничения', '',
              '- Смоук ограничен 256 шагами на поколение (Weaver infinity mode); полный проход эпохи train8m '
              'и его traversal audit выполняются только в Stage 3.',
              '- windowed_pbt_v2 и cadenced_pbt_v1 запрещают ограничение выборки, поэтому GPU-смоук использует '
              'fixed_lr_grid; windowed-черновик проверен реальным --dry-run и CPU-тестами.',
              '- Физические event ID отсутствуют; изоляция подтверждена окнами Entry$ и скалярной идентичностью.']
    return '\n'.join(lines) + '\n'
