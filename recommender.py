"""Hybrid knowledge-based recommender for SME digital-transformation actions.

The demonstration uses simulated, transparent SME profiles.  Replace
``simulate_interactions`` with real, governed SME implementation outcomes before
making empirical claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


ACTIONS = (
    "Cloud CRM",
    "Business intelligence dashboard",
    "E-commerce personalization",
    "Workflow automation",
    "Cybersecurity foundation",
    "Demand forecasting",
)

FEATURES = (
    "customer_data",
    "process_digitization",
    "analytics_capability",
    "digital_sales",
    "cybersecurity_maturity",
    "budget_capacity",
    "growth_ambition",
)


@dataclass(frozen=True)
class Fit:
    action: int
    score: float
    reasons: Tuple[str, ...]


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -35, 35)
    return 1.0 / (1.0 + np.exp(-values))


def knowledge_fit(profile: np.ndarray, action: int) -> Fit:
    """Score a firm-action pair using auditable business rules in [0, 1]."""
    c, p, a, s, cy, b, g = profile
    if action == 0:  # CRM
        raw, reasons = 0.45 * c + 0.30 * s + 0.15 * b + 0.10 * g, (
            "customer-data readiness", "digital-sales opportunity")
    elif action == 1:  # BI
        raw, reasons = 0.45 * a + 0.25 * c + 0.20 * b + 0.10 * p, (
            "analytics capability", "decision-support need")
    elif action == 2:  # personalization
        raw, reasons = 0.40 * s + 0.30 * c + 0.20 * a + 0.10 * g, (
            "digital-sales maturity", "customer-data readiness")
    elif action == 3:  # automation
        raw, reasons = 0.45 * p + 0.25 * b + 0.20 * a + 0.10 * g, (
            "process digitization", "automation capacity")
    elif action == 4:  # cyber
        raw, reasons = 0.40 * (1 - cy) + 0.25 * c + 0.20 * s + 0.15 * b, (
            "cybersecurity gap", "data-exposure risk")
    else:  # forecasting
        raw, reasons = 0.40 * a + 0.25 * p + 0.20 * b + 0.15 * g, (
            "analytics capability", "growth-planning need")
    return Fit(action, float(np.clip(raw, 0, 1)), reasons)


def pair_features(profile: np.ndarray, action: int) -> np.ndarray:
    """Encode a firm-action pair for the data-driven model."""
    one_hot = np.eye(len(ACTIONS))[action]
    fit = knowledge_fit(profile, action).score
    return np.r_[1.0, profile, one_hot, fit, profile * fit]


def simulate_interactions(n_firms: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create a controlled test bed with latent implementation success labels."""
    rng = np.random.default_rng(seed)
    profiles = rng.beta(2.3, 2.0, size=(n_firms, len(FEATURES)))
    action_bias = np.array([0.10, 0.06, -0.03, 0.08, 0.12, 0.01])
    labels = np.zeros((n_firms, len(ACTIONS)), dtype=int)
    for firm, profile in enumerate(profiles):
        for action in range(len(ACTIONS)):
            fit = knowledge_fit(profile, action).score
            # Latent outcome: business-rule fit plus noisy implementation factors.
            latent = -1.45 + 3.25 * fit + action_bias[action] + rng.normal(0, 0.55)
            labels[firm, action] = int(rng.random() < sigmoid(np.array([latent]))[0])
    return profiles, labels


class LogisticPairModel:
    """Small dependency-free logistic model for reproducible pair scoring."""

    def __init__(self, learning_rate: float = 0.12, epochs: int = 900, l2: float = 0.002):
        self.learning_rate, self.epochs, self.l2 = learning_rate, epochs, l2
        self.weights: np.ndarray | None = None

    def fit(self, profiles: np.ndarray, labels: np.ndarray) -> "LogisticPairModel":
        X, y = self._expand(profiles, labels)
        self.weights = np.zeros(X.shape[1])
        for _ in range(self.epochs):
            probabilities = sigmoid(X @ self.weights)
            gradient = (X.T @ (probabilities - y)) / len(y)
            gradient[1:] += self.l2 * self.weights[1:]
            self.weights -= self.learning_rate * gradient
        return self

    def predict_proba(self, profiles: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Fit the model before predicting.")
        scores = np.empty((len(profiles), len(ACTIONS)))
        for i, profile in enumerate(profiles):
            for action in range(len(ACTIONS)):
                scores[i, action] = sigmoid(np.array([pair_features(profile, action) @ self.weights]))[0]
        return scores

    @staticmethod
    def _expand(profiles: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.vstack([pair_features(profile, action)
                       for profile in profiles for action in range(len(ACTIONS))])
        y = labels.reshape(-1).astype(float)
        return X, y


def knowledge_scores(profiles: np.ndarray) -> np.ndarray:
    return np.array([[knowledge_fit(profile, action).score for action in range(len(ACTIONS))]
                     for profile in profiles])


def rank_metrics(scores: np.ndarray, labels: np.ndarray, k: int = 3) -> Dict[str, float]:
    precisions, ndcgs, coverages = [], [], []
    for row, truth in zip(scores, labels):
        ranked = np.argsort(-row)[:k]
        gains = truth[ranked]
        precisions.append(float(gains.mean()))
        discounts = 1 / np.log2(np.arange(2, k + 2))
        dcg = float((gains * discounts).sum())
        ideal = np.sort(truth)[::-1][:k]
        idcg = float((ideal * discounts).sum())
        ndcgs.append(dcg / idcg if idcg else 0.0)
        coverages.extend(ranked.tolist())
    return {"precision_at_3": float(np.mean(precisions)),
            "ndcg_at_3": float(np.mean(ndcgs)),
            "catalog_coverage": float(len(set(coverages)) / len(ACTIONS))}


def recommendation(profile: np.ndarray, model: LogisticPairModel, alpha: float = 0.55) -> List[Fit]:
    knowledge = np.array([knowledge_fit(profile, action).score for action in range(len(ACTIONS))])
    learned = model.predict_proba(profile.reshape(1, -1))[0]
    combined = alpha * knowledge + (1 - alpha) * learned
    ranked = np.argsort(-combined)
    return [Fit(int(action), float(combined[action]), knowledge_fit(profile, int(action)).reasons)
            for action in ranked]
