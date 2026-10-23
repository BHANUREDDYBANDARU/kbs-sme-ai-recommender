"""Run a reproducible simulation benchmark and generate the paper figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from recommender import ACTIONS, FEATURES, LogisticPairModel, knowledge_scores, rank_metrics, recommendation, simulate_interactions

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"


def font(size: int, bold: bool = False):
    path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"
    return ImageFont.truetype(path, size)


def benchmark() -> tuple[dict, list[dict], tuple[np.ndarray, LogisticPairModel]]:
    runs, example = [], None
    for seed in range(11, 16):
        profiles, labels = simulate_interactions(480, seed)
        idx = np.random.default_rng(seed + 99).permutation(len(profiles))
        train, test = idx[:360], idx[360:]
        model = LogisticPairModel().fit(profiles[train], labels[train])
        knowledge = knowledge_scores(profiles[test])
        data_only = model.predict_proba(profiles[test])
        hybrid = 0.55 * knowledge + 0.45 * data_only
        popularity = np.tile(labels[train].mean(axis=0), (len(test), 1))
        systems = {"Popularity": popularity, "Data-driven": data_only,
                   "Knowledge-based": knowledge, "Hybrid KBS-AI": hybrid}
        for name, scores in systems.items():
            runs.append({"seed": seed, "system": name, **rank_metrics(scores, labels[test])})
        if seed == 11:
            example = (profiles[test][0], model)
    summary = {}
    for system in ("Popularity", "Data-driven", "Knowledge-based", "Hybrid KBS-AI"):
        entries = [r for r in runs if r["system"] == system]
        summary[system] = {metric: round(float(np.mean([r[metric] for r in entries])), 3)
                           for metric in ("precision_at_3", "ndcg_at_3", "catalog_coverage")}
        summary[system]["precision_at_3_sd"] = round(float(np.std([r["precision_at_3"] for r in entries], ddof=1)), 3)
        summary[system]["ndcg_at_3_sd"] = round(float(np.std([r["ndcg_at_3"] for r in entries], ddof=1)), 3)
    return summary, runs, example


def fig_architecture():
    image = Image.new("RGB", (1800, 860), "white")
    draw = ImageDraw.Draw(image)
    navy, blue, teal, gray = "#0B1F3A", "#2E75B6", "#1B998B", "#EAF0F6"
    boxes = [(70, 260, 490, 610, "SME data", ["digital maturity", "capabilities and goals", "budget and constraints"]),
             (670, 115, 1130, 385, "Knowledge layer", ["business rules", "action-capability links", "explanation traces"]),
             (670, 485, 1130, 755, "Data-driven layer", ["pairwise logistic model", "implementation outcomes", "success probabilities"]),
             (1310, 260, 1730, 610, "Ranked actions", ["top-3 recommendations", "fit score", "human-readable reasons"])]
    for x1, y1, x2, y2, title, lines in boxes:
        draw.rounded_rectangle((x1, y1, x2, y2), radius=26, fill=gray, outline=blue, width=5)
        draw.text((x1 + 28, y1 + 28), title, fill=navy, font=font(36, True))
        for j, line in enumerate(lines):
            draw.text((x1 + 32, y1 + 104 + j * 58), "• " + line, fill=navy, font=font(27))
    for xy in ((490, 435, 670, 250), (490, 435, 670, 620), (1130, 250, 1310, 435), (1130, 620, 1310, 435)):
        draw.line(xy, fill=teal, width=10)
        x1, y1, x2, y2 = xy
        draw.polygon([(x2, y2), (x2 - 25, y2 - 16), (x2 - 25, y2 + 16)], fill=teal)
    draw.rounded_rectangle((1180, 382, 1260, 490), radius=12, fill=teal)
    draw.text((1197, 407), "α", fill="white", font=font(42, True))
    draw.text((70, 55), "Hybrid knowledge-based AI recommender architecture", fill=navy, font=font(44, True))
    image.save(FIGURES / "figure_1_architecture.png")


def fig_benchmark(summary: dict):
    image = Image.new("RGB", (1800, 900), "white")
    draw = ImageDraw.Draw(image)
    navy, blue, teal, orange, gray = "#0B1F3A", "#2E75B6", "#1B998B", "#F28E2B", "#EAF0F6"
    draw.text((70, 45), "Simulation benchmark: ranking quality at K = 3", fill=navy, font=font(42, True))
    systems = list(summary)
    metrics = [("precision_at_3", "Precision@3", blue), ("ndcg_at_3", "NDCG@3", teal)]
    left, top, width, height = 150, 170, 1530, 540
    for y in np.linspace(0, 1, 6):
        py = top + height - int(y * height)
        draw.line((left, py, left + width, py), fill=gray, width=2)
        draw.text((75, py - 13), f"{y:.1f}", fill=navy, font=font(22))
    group = width / len(systems)
    bar_w = 95
    for i, name in enumerate(systems):
        center = left + group * (i + 0.5)
        for j, (metric, _, color) in enumerate(metrics):
            value = summary[name][metric]
            x1 = int(center + (j - 0.5) * 125 - bar_w / 2)
            y1 = int(top + height - value * height)
            draw.rectangle((x1, y1, x1 + bar_w, top + height), fill=color)
            draw.text((x1 + 12, y1 - 36), f"{value:.2f}", fill=navy, font=font(21, True))
        draw.multiline_text((int(center - 120), top + height + 30), name, fill=navy, font=font(25), align="center", spacing=4)
    draw.rectangle((1040, 780, 1065, 805), fill=blue); draw.text((1078, 777), "Precision@3", fill=navy, font=font(24))
    draw.rectangle((1325, 780, 1350, 805), fill=teal); draw.text((1363, 777), "NDCG@3", fill=navy, font=font(24))
    image.save(FIGURES / "figure_2_benchmark.png")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary, runs, (profile, model) = benchmark()
    sample = [{"rank": i + 1, "action": ACTIONS[fit.action], "score": round(fit.score, 3),
               "explanation": "; ".join(fit.reasons)} for i, fit in enumerate(recommendation(profile, model)[:3])]
    (RESULTS / "summary.json").write_text(json.dumps({"summary": summary, "sample_recommendations": sample}, indent=2))
    with (RESULTS / "benchmark_runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=runs[0].keys()); writer.writeheader(); writer.writerows(runs)
    fig_architecture(); fig_benchmark(summary)
    print(json.dumps({"summary": summary, "sample_recommendations": sample}, indent=2))


if __name__ == "__main__":
    main()
