"""GA-style report plotting utilities for MountainCar experiments.
Produces comparison bar charts and per-config learning curves from outputs/logs.
"""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib
# use non-interactive backend for headless environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_theme(style='whitegrid', context='talk')
except Exception:
    sns = None


def _load_summaries(logs_dir: Path):
    summaries = []
    episode_logs = []
    for p in sorted(Path(logs_dir).glob('*_summary.json')):
        try:
            s = json.loads(p.read_text())
            fallback = p.name.replace('_summary.json', '')
            summaries.append({
                'config_name': s.get('experiment_name', fallback),
                'seed': s.get('seed'),
                'agent': s.get('agent'),
                'reward_mean': s.get('reward_mean'),
                'reward_std': s.get('reward_std'),
                'success_rate': s.get('success_rate'),
                'eval_mean_reward_last': s.get('eval_mean_reward_last'),
            })
        except Exception:
            continue
    for p in sorted(Path(logs_dir).glob('*_episodes.csv')):
        try:
            df = pd.read_csv(p)
            name = p.name.replace('_episodes.csv', '')
            episode_logs.append((name, df))
        except Exception:
            continue
    return pd.DataFrame(summaries), episode_logs


def plot_comparisons(logs_dir: str | Path = 'outputs/logs', out_dir: str | Path = 'outputs/figures'):
    logs_dir = Path(logs_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries, episode_logs = _load_summaries(logs_dir)
    if summaries.empty:
        print('No summaries found in', logs_dir)
        return

    agg = summaries.groupby('config_name', as_index=False).agg({'eval_mean_reward_last': 'mean', 'success_rate': 'mean', 'reward_mean': 'mean'})

    if 'eval_mean_reward_last' in agg.columns:
        plt.figure(figsize=(10, 4))
        if sns is not None:
            sns.barplot(data=agg, x='config_name', y='eval_mean_reward_last')
        else:
            plt.bar(agg['config_name'], agg['eval_mean_reward_last'])
        plt.xticks(rotation=45, ha='right')
        plt.title('Final evaluation mean reward by config')
        plt.tight_layout()
        path = out_dir / 'ga_comparison_eval_mean_reward.png'
        plt.savefig(path, dpi=180)
        plt.show()
        print('Saved', path)

    if 'success_rate' in agg.columns:
        plt.figure(figsize=(10, 4))
        if sns is not None:
            sns.barplot(data=agg, x='config_name', y='success_rate')
        else:
            plt.bar(agg['config_name'], agg['success_rate'])
        plt.xticks(rotation=45, ha='right')
        plt.title('Success rate by config')
        plt.tight_layout()
        path = out_dir / 'ga_comparison_success_rate.png'
        plt.savefig(path, dpi=180)
        plt.show()
        print('Saved', path)

    # per-model learning curves
    if episode_logs:
        grouped = {}
        for name, df in episode_logs:
            cfg = name
            grouped.setdefault(cfg, []).append(df)
        for cfg, dfs in grouped.items():
            plt.figure(figsize=(10, 4))
            for df in dfs:
                if 'episode' in df.columns and 'reward' in df.columns:
                    y = df['reward'].rolling(window=50, min_periods=1).mean()
                    plt.plot(df['episode'], y, alpha=0.7)
            plt.xlabel('Episode')
            plt.ylabel('Reward (rolling mean)')
            plt.title(f'Learning curve: {cfg}')
            plt.tight_layout()
            path = out_dir / f'ga_{cfg}_learning_curve.png'
            plt.savefig(path, dpi=180)
            plt.show()
            print('Saved', path)
    else:
        print('No per-episode logs found.')


if __name__ == '__main__':
    plot_comparisons()
