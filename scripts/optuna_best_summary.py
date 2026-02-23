import json
import os
import re
import sys


def read_latest_metrics(trial_dir):
    p = os.path.join(trial_dir, 'latest_run.json')
    if not os.path.exists(p):
        # try inside dated folder
        for name in os.listdir(trial_dir):
            candidate = os.path.join(trial_dir, name, 'latest_run.json')
            if os.path.exists(candidate):
                p = candidate
                break
    try:
        with open(p, 'r', encoding='utf-8') as f:
            j = json.load(f)
        return {
            'auprc': j.get('best_val_auprc'),
            'auroc': j.get('best_val_auroc'),
            'f1_macro': j.get('best_val_f1_macro'),
            'primary_metric': j.get('primary_metric', 'auprc' if j.get('best_val_auprc') is not None else 'auroc'),
        }
    except Exception:
        return None


def read_config_yaml(trial_dir):
    # locate .hydra/config.yaml
    hydra = os.path.join(trial_dir, '.hydra', 'config.yaml')
    if not os.path.exists(hydra):
        # try dated folder
        for name in os.listdir(trial_dir):
            candidate = os.path.join(trial_dir, name, '.hydra', 'config.yaml')
            if os.path.exists(candidate):
                hydra = candidate
                break
    if not os.path.exists(hydra):
        return {}
    try:
        import yaml
        with open(hydra, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception:
        # fallback: simple regex extraction for common params
        params = {}
        with open(hydra, 'r', encoding='utf-8') as f:
            s = f.read()
        m = re.search(r"learning_rate:\s*([0-9.eE+-]+)", s)
        if m:
            params['training'] = params.get('training', {})
            params['training']['learning_rate'] = float(m.group(1))
        m = re.search(r"weight_decay:\s*([0-9.eE+-]+)", s)
        if m:
            params['training'] = params.get('training', {})
            params['training']['weight_decay'] = float(m.group(1))
        m = re.search(r"dropout:\s*([0-9.eE+-]+)", s)
        if m:
            params['model'] = params.get('model', {})
            params['model']['params'] = params['model'].get('params', {})
            params['model']['params']['dropout'] = float(m.group(1))
        return params


def summarize(study_dir):
    items = [d for d in os.listdir(study_dir) if os.path.isdir(os.path.join(study_dir, d))]
    trials = [d for d in items if re.search(r'_t\d+$', d)]
    results = []
    for t in trials:
        td = os.path.join(study_dir, t)
        metrics = read_latest_metrics(td)
        cfg = read_config_yaml(td)
        results.append((t, metrics, cfg))
    # also consider any top-level latest_run.json
    # pick best by primary metric (AUPRC preferred, fallback AUROC)
    results = [r for r in results if r[1] is not None]
    if not results:
        print(f'No usable trials found in {study_dir}')
        return
    def metric_value(item):
        metrics = item[1] or {}
        value = metrics.get('auprc')
        if value is None:
            value = metrics.get('auroc')
        return float(value) if value is not None else float('-inf')

    best = max(results, key=metric_value)
    print(f'Study: {os.path.basename(study_dir)}')
    print(f' Best trial: {best[0]}')
    best_metrics = best[1] or {}
    if best_metrics.get('auprc') is not None:
        print(f" Best val AUPRC: {float(best_metrics['auprc']):.6f}")
    if best_metrics.get('auroc') is not None:
        print(f" Best val AUROC: {float(best_metrics['auroc']):.6f}")
    if best_metrics.get('f1_macro') is not None:
        print(f" Best val F1-macro: {float(best_metrics['f1_macro']):.6f}")
    cfg = best[2]
    # print a few likely hyperparameters
    def get_path(d, *keys):
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                return None
            d = d[k]
        return d

    lr = get_path(cfg, 'training', 'learning_rate')
    wd = get_path(cfg, 'training', 'weight_decay')
    dropout = get_path(cfg, 'model', 'params', 'dropout')
    hidden = get_path(cfg, 'model', 'params', 'hidden_dims')
    bs = get_path(cfg, 'training', 'batch_size')
    print(' Hyperparameters:')
    if lr is not None:
        print(f'  - learning_rate: {lr}')
    if wd is not None:
        print(f'  - weight_decay: {wd}')
    if dropout is not None:
        print(f'  - dropout: {dropout}')
    if hidden is not None:
        print(f'  - hidden_dims: {hidden}')
    if bs is not None:
        print(f'  - batch_size: {bs}')
    print('')


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else 'outputs/optuna_gpt_labels_trials'
    if not os.path.isabs(base):
        base = os.path.abspath(base)
    if not os.path.exists(base):
        print('Path not found:', base)
        return
    summarize(base)


if __name__ == '__main__':
    main()
