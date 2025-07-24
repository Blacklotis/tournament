#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from tqdm import trange
import shutil

PLAYER_NAMES = [f"Player {i}" for i in range(1, 9)]
DUEL_SKILLS = np.linspace(25, 50, 8)
FFA_SKILLS = np.linspace(25, 50, 8)

def load_players(config_path=None):
    if config_path and Path(config_path).exists():
        df = pd.read_csv(config_path)
        return df["Player"].tolist(), df["Duel_Skill"].to_numpy(), df["FFA_Skill"].to_numpy()
    else:
        names = [f"Player {i}" for i in range(1, 9)]
        duel = np.linspace(25, 50, 8)
        ffa = np.linspace(25, 50, 8)
        return names, duel, ffa

def logistic_win_probability(skill_a, skill_b, scale=10):
    return 1 / (1 + np.exp((skill_b - skill_a) / scale))

def simulate_ffa_ranks(ffa_skills, multipliers):
    noisy = ffa_skills + np.random.normal(0, 5, len(ffa_skills))
    ranks = np.argsort(-noisy)
    scores = np.zeros(len(ffa_skills))
    kills = []
    for i, killer_idx in enumerate(ranks[:-1]):
        victim_idx = ranks[i + 1]
        scores[killer_idx] += multipliers[killer_idx]
        kills.append((PLAYER_NAMES[killer_idx], PLAYER_NAMES[victim_idx]))
    return ranks, scores, kills

def placement_multipliers(ranks, min_mult=1.0, max_mult=2.0):
    n = len(ranks)
    return [max_mult - (i / (n - 1)) * (max_mult - min_mult) for i in range(n)]

def simulate_reduced_round_robin(skills, n_matches):
    n = len(skills)
    wins = np.zeros(n)
    logs = []
    for i in range(n):
        opponents = np.random.choice([j for j in range(n) if j != i], min(n_matches, n - 1), replace=False)
        for j in opponents:
            prob_i = logistic_win_probability(skills[i], skills[j])
            result = "win" if np.random.rand() < prob_i else "loss"
            wins[i if result == "win" else j] += 1
            logs.append((i, j, result, prob_i))
    return wins, logs

def run_simulation(args):
    global PLAYER_NAMES, DUEL_SKILLS, FFA_SKILLS
    PLAYER_NAMES, DUEL_SKILLS, FFA_SKILLS = load_players(args.config)

    if not args.config:
        args.rrr_matches = len(PLAYER_NAMES)

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_rounds = len(PLAYER_NAMES) // 2
    matches_per_round = args.rrr_matches // total_rounds

    formats = {f"FFA_Round{r}": [] for r in range(total_rounds + 1)}
    formats["RRR"] = []
    formats["FFA"] = []
    match_logs, ffa_logs = [], []

    player_summary = pd.DataFrame({
        "Player": PLAYER_NAMES,
        "Duel_Skill": DUEL_SKILLS,
        "FFA_Skill": FFA_SKILLS
    })

    for sim in trange(args.simulations, desc="Simulating Formats"):
        initial_mults = np.random.uniform(args.min_mult, args.max_mult, size=len(DUEL_SKILLS))
        base_skills = DUEL_SKILLS * initial_mults

        for ffa_round in range(total_rounds + 1):
            skills = base_skills.copy()
            total_wins = np.zeros(len(skills))
            logs = []

            for r in range(total_rounds):
                wins, round_logs = simulate_reduced_round_robin(skills, matches_per_round)
                total_wins += wins
                logs.extend(round_logs)

                if r + 1 == ffa_round:
                    ranks, ffa_scores, kills = simulate_ffa_ranks(FFA_SKILLS, initial_mults)
                    mults = placement_multipliers(ranks, args.min_mult, args.max_mult)
                    skills = DUEL_SKILLS * mults
                    ffa_logs += [{
                        "Format": f"FFA_Round{ffa_round}",
                        "Sim": sim,
                        "Player": PLAYER_NAMES[idx],
                        "Rank": rank + 1,
                        "FFA_Score": ffa_scores[idx],
                        "Killed_By": next((killer for killer, victim in kills if victim == PLAYER_NAMES[idx]), None),
                        "Killed": next((victim for killer, victim in kills if killer == PLAYER_NAMES[idx]), None)
                    } for rank, idx in enumerate(ranks)]

            formats[f"FFA_Round{ffa_round}"].append(total_wins)
            match_logs += [{"Format": f"FFA_Round{ffa_round}", "Sim": sim,
                            "P1": PLAYER_NAMES[i], "P2": PLAYER_NAMES[j],
                            "Result": res, "Prob": round(prob, 3)}
                           for i, j, res, prob in logs]

        # RRR control
        wins_rrr, logs_rrr = simulate_reduced_round_robin(DUEL_SKILLS, args.rrr_matches)
        formats["RRR"].append(wins_rrr)
        match_logs += [{"Format": "RRR", "Sim": sim,
                        "P1": PLAYER_NAMES[i], "P2": PLAYER_NAMES[j],
                        "Result": res, "Prob": round(prob, 3)}
                       for i, j, res, prob in logs_rrr]

        # FFA control
        ffa_ranks, ffa_scores, kills = simulate_ffa_ranks(FFA_SKILLS, np.ones(len(FFA_SKILLS)))
        placements = np.zeros(len(PLAYER_NAMES))
        for pos, player_idx in enumerate(ffa_ranks):
            placements[player_idx] = len(PLAYER_NAMES) - pos
            ffa_logs.append({
                "Format": "FFA", "Sim": sim,
                "Player": PLAYER_NAMES[player_idx],
                "Rank": pos + 1,
                "FFA_Score": ffa_scores[player_idx],
                "Killed_By": next((killer for killer, victim in kills if victim == PLAYER_NAMES[player_idx]), None),
                "Killed": next((victim for killer, victim in kills if killer == PLAYER_NAMES[player_idx]), None)
            })
        formats["FFA"].append(placements)

    summary_rows = []
    for player, duel_skill, ffa_skill in zip(PLAYER_NAMES, DUEL_SKILLS, FFA_SKILLS):
        row = {"Player": player, "Duel_Skill": duel_skill, "FFA_Skill": ffa_skill}
        for fmt, data in formats.items():
            arr = np.array(data)
            stddev = np.std(np.argsort(-arr, axis=1), axis=0).mean()
            row[fmt] = round(stddev, 2)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    pd.DataFrame(match_logs).to_csv(output_dir / "match_logs.csv", index=False)
    pd.DataFrame(ffa_logs).to_csv(output_dir / "ffa_logs.csv", index=False)

    matchups_df = pd.DataFrame([{"Simulation": sim, "Format": fmt, "Player": PLAYER_NAMES[i], "Wins": score}
                                for fmt, data in formats.items()
                                for sim, scores in enumerate(data)
                                for i, score in enumerate(scores)])
    matchups_df.to_csv(output_dir / "format_matchups.csv", index=False)

    plt.figure(figsize=(10, 6))
    sns.violinplot(data=matchups_df, x="Format", y="Wins", inner="box", cut=0)
    plt.title("Score Distribution per Format")
    plt.tight_layout()
    plt.savefig(output_dir / "graph_score_distributions.png")

    winners = matchups_df.loc[matchups_df.groupby(["Format", "Simulation"])["Wins"].idxmax()]
    winner_counts = winners.groupby(["Format", "Player"]).size().reset_index(name="Wins")
    winner_pcts = winner_counts.copy()
    winner_pcts["Pct"] = winner_pcts.groupby("Format")["Wins"].transform(lambda x: 100 * x / x.sum())

    duel_map = dict(zip(PLAYER_NAMES, np.round(DUEL_SKILLS, 2).astype(str)))
    ffa_map = dict(zip(PLAYER_NAMES, np.round(FFA_SKILLS, 2).astype(str)))

    winner_pcts["PlayerLabel"] = winner_pcts["Player"] + " (D:" + winner_pcts["Player"].map(duel_map) + " / F:" + winner_pcts["Player"].map(ffa_map) + ")"

    plt.figure(figsize=(12, 6))
    sns.barplot(data=winner_pcts, x="Format", y="Pct", hue="PlayerLabel")
    plt.title("Winner Frequency by Player and Format")
    plt.tight_layout()
    plt.savefig(output_dir / "graph_winner_frequency.png")

    print("✅ Simulation complete.")
    print("📁 Output:", output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tournament Format Variability Simulator")
    parser.add_argument("--config", type=str, help="Optional path to player config CSV")
    parser.add_argument("--simulations", type=int, default=1000)
    parser.add_argument("--rrr_matches", type=int, default=8)
    parser.add_argument("--min_mult", type=float, default=1.0)
    parser.add_argument("--max_mult", type=float, default=2.0)
    parser.add_argument("--output_dir", type=str, default="tournament_results")
    args = parser.parse_args()
    run_simulation(args)
