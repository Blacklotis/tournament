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

def load_players(config_path=None, num_players=8):
    if config_path and Path(config_path).exists():
        df = pd.read_csv(config_path)
        return df["Player"].tolist(), df["Duel_Skill"].to_numpy(), df["FFA_Skill"].to_numpy()
    else:
        names = [f"Player {i}" for i in range(1, num_players + 1)]
        duel = np.linspace(25, 50, num_players)
        ffa = np.linspace(25, 50, num_players)
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

def simulate_reduced_round_robin(skills):
    n = len(skills)
    wins = np.zeros(n)
    logs = []
    for i in range(n):
        opponents = np.random.choice([j for j in range(n) if j != i], min(1, n - 1), replace=False)
        for j in opponents:
            prob_i = logistic_win_probability(skills[i], skills[j])
            result = "win" if np.random.rand() < prob_i else "loss"
            wins[i if result == "win" else j] += 1
            logs.append((i, j, result, prob_i))
    return wins, logs

def run_simulation(args):
    def run_one_sim(name_order, duel_skills, ffa_skills, output_path):
        global PLAYER_NAMES, DUEL_SKILLS, FFA_SKILLS
        PLAYER_NAMES = name_order
        DUEL_SKILLS = duel_skills
        FFA_SKILLS = ffa_skills

        if Path(output_path).exists():
            shutil.rmtree(output_path)
        Path(output_path).mkdir(parents=True, exist_ok=True)

        total_rounds = len(PLAYER_NAMES) // 2
        matches_per_round = 1

        formats = {f"FFA_Round{r}": [] for r in range(total_rounds + 1)}
        formats["RRR"] = []
        formats["FFA"] = []
        match_logs, ffa_logs = [], []

        for sim in trange(args.simulations, desc=f"Simulating ({output_path.name})"):
            initial_mults = np.random.uniform(args.min_mult, args.max_mult, size=len(DUEL_SKILLS))
            base_skills = DUEL_SKILLS * initial_mults

            for ffa_round in range(total_rounds + 1):
                skills = base_skills.copy()
                total_wins = np.zeros(len(skills))
                logs = []

                for r in range(total_rounds):
                    wins, round_logs = simulate_reduced_round_robin(skills)
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

            wins_rrr, logs_rrr = simulate_reduced_round_robin(DUEL_SKILLS)
            formats["RRR"].append(wins_rrr)
            match_logs += [{"Format": "RRR", "Sim": sim,
                            "P1": PLAYER_NAMES[i], "P2": PLAYER_NAMES[j],
                            "Result": res, "Prob": round(prob, 3)}
                           for i, j, res, prob in logs_rrr]

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

        pd.DataFrame(match_logs).to_csv(output_path / "match_logs.csv", index=False)
        pd.DataFrame(ffa_logs).to_csv(output_path / "ffa_logs.csv", index=False)
        pd.DataFrame(summary_rows).to_csv(output_path / "summary.csv", index=False)

        matchups_df = pd.DataFrame([{"Simulation": sim, "Format": fmt, "Player": PLAYER_NAMES[i], "Wins": score}
                                    for fmt, data in formats.items()
                                    for sim, scores in enumerate(data)
                                    for i, score in enumerate(scores)])
        matchups_df.to_csv(output_path / "format_matchups.csv", index=False)

        plt.figure(figsize=(10, 6))
        sns.violinplot(data=matchups_df, x="Format", y="Wins", inner="box", cut=0)
        plt.title("Score Distribution per Format")
        plt.tight_layout()
        plt.savefig(output_path / "graph_score_distributions.png")

        winners = matchups_df.loc[matchups_df.groupby(["Format", "Simulation"])["Wins"].idxmax()]
        winner_counts = winners.groupby(["Format", "Player"]).size().reset_index(name="Wins")
        winner_pcts = winner_counts.copy()
        winner_pcts["Pct"] = winner_pcts.groupby("Format")["Wins"].transform(lambda x: 100 * x / x.sum())

        duel_map = dict(zip(PLAYER_NAMES, np.round(DUEL_SKILLS, 2).astype(str)))
        ffa_map = dict(zip(PLAYER_NAMES, np.round(FFA_SKILLS, 2).astype(str)))
        winner_pcts["PlayerLabel"] = winner_pcts["Player"] + " (D:" + winner_pcts["Player"].map(duel_map) + " / F:" + winner_pcts["Player"].map(ffa_map) + ")"

        # Sort hue order by Duel Skill
        hue_order = sorted(PLAYER_NAMES, key=lambda p: duel_map[p])
        hue_labels = [p + " (D:" + duel_map[p] + " / F:" + ffa_map[p] + ")" for p in hue_order]

        plt.figure(figsize=(12, 6))
        sns.barplot(data=winner_pcts, x="Format", y="Pct", hue="PlayerLabel", hue_order=hue_labels)
        plt.title("Winner Frequency by Player and Format (sorted by Duel Skill)")
        plt.tight_layout()
        plt.savefig(output_path / "graph_winner_frequency.png")

        print(f"✅ Finished: {output_path}")

    # load from config or generate default
    if args.config:
        names, duel, ffa = load_players(args.config, args.num_players)
        run_one_sim(names, duel, ffa, Path(args.output_dir))
    else:
        n = args.num_players
        base_names = [f"Player {i}" for i in range(1, n + 1)]
        duel = np.linspace(25, 50, n)
        ffa = np.linspace(25, 50, n)

        # run once sorted by duel skill
        duel_order = [x for _, x in sorted(zip(duel, base_names))]
        duel_sorted_duel = np.sort(duel)
        duel_sorted_ffa = np.array([ffa[base_names.index(name)] for name in duel_order])
        run_one_sim(duel_order, duel_sorted_duel, duel_sorted_ffa, Path(args.output_dir) / "duel_sorted")

        # run once sorted by ffa skill
        ffa_order = [x for _, x in sorted(zip(ffa, base_names))]
        ffa_sorted_ffa = np.sort(ffa)
        ffa_sorted_duel = np.array([duel[base_names.index(name)] for name in ffa_order])
        run_one_sim(ffa_order, ffa_sorted_duel, ffa_sorted_ffa, Path(args.output_dir) / "ffa_sorted")


    print("✅ Simulation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tournament Format Variability Simulator")
    parser.add_argument("--config", type=str, help="Optional path to player config CSV")
    parser.add_argument("--num_players", type=int, default=8,help="Only used if no config is provided. Specifies number of default players.")
    parser.add_argument("--simulations", type=int, default=8)
    parser.add_argument("--min_mult", type=float, default=1.0)
    parser.add_argument("--max_mult", type=float, default=2.0)
    parser.add_argument("--output_dir", type=str, default="tournament_results")
    args = parser.parse_args()
    run_simulation(args)
