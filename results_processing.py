import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def pool(stats):
    """Pool several runs into one.

    stats: list of (mean, standard_error, n_episodes) tuples.
    Returns (mean, standard_error, n_episodes) for the combined episodes.
    """
    means = np.array([s[0] for s in stats], dtype=float)
    ses = np.array([s[1] for s in stats], dtype=float)
    ns = np.array([s[2] for s in stats], dtype=float)

    N = ns.sum()
    M = (ns * means).sum() / N

    variances = ses**2 * ns
    total_var = (ns * variances).sum() / N + (ns * (means - M) ** 2).sum() / N

    return M, np.sqrt(total_var / N), int(N)


def get_task_steps_stats(env_folder: Path, task: str):
    task_dir = env_folder.joinpath(*task.split("/"))
    steps = [
        json.loads(f.read_text())["num_steps"] 
        for f in task_dir.rglob("*run*.json")
    ]
    if not steps:
        return None

    n = len(steps)
    se = np.std(steps, ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return float(np.mean(steps)), float(se), n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("keyword", help="keyword of the model, such as: gemini or Qwen")
    parser.add_argument("envs", nargs="*", default=["babyai", "crafter"], help="environments")
    args = parser.parse_args()

    # results path
    results_dir = Path("results")
    dirs = sorted([p for p in results_dir.glob(f"*{args.keyword}*") if p.is_dir()])
    
    print(f"Found {len(dirs)} directories for '{args.keyword}':")
    for d in dirs:
        print(f"  - {d.name}")

    for env in args.envs:
        overall = []
        per_task = defaultdict(list)
        
        # for steps
        steps_overall = []
        steps_per_task = defaultdict(list)

        for d in dirs:
            file = d / env / f"{env}_summary.json"
            if not file.exists():
                file = d / f"{env}_summary.json"
            if not file.exists():
                continue

            with open(file) as f:
                data = json.load(f)

            overall.append((data["progression_percentage"], data["standard_error"], data["episodes_played"]))
            for task, t in data.get("tasks", {}).items():
                per_task[task].append((t["progression_percentage"], t["standard_error"], t["episodes_played"]))

            # to read steps
            env_folder = d / env if (d / env).is_dir() else d
            for task in data.get("tasks", {}).keys():
                stat = get_task_steps_stats(env_folder, task)
                if stat:
                    steps_per_task[task].append(stat)
                    steps_overall.append(stat)

        if not overall:
            continue

        print(f"\n=== {env.upper()} ===")
        m, se, n = pool(overall)
        print(f"Overall: {m:.2f} ± {se:.2f} (SE), {n} episodes")
        
        if steps_overall:
            sm, sse, _ = pool(steps_overall)
            print(f"Overall Steps: {sm:.2f} ± {sse:.2f} (SE)")

        for task, stats in sorted(per_task.items()):
            m, se, n = pool(stats)
            print(f"  {task}: {m:.2f} ± {se:.2f} (SE), {n} episodes")
            
            if task in steps_per_task and steps_per_task[task]:
                sm, sse, _ = pool(steps_per_task[task])
                print(f"    Steps: {sm:.2f} ± {sse:.2f} (SE)")


if __name__ == "__main__":
    main()