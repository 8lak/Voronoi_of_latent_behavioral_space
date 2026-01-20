import argparse
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from matplotlib.widgets import Slider, Button, RadioButtons

import pandas as pd


# --- LOAD DATA FROM WEEK 1 ---
BEHAVIORS = [
    'Work', 'Focused Learning', 'Skill practicing', 'Physical Endeavors',
    'Scrolling', 'jorking', 'Passive media', 'Active Media',
    'System Architecture'
]

REGRESSION_PACK_PATH = "regression_analysis/linear_equation_pack.json"
REGRESSION_STATS_PATH = "data_in_csv/21_set_raw.csv"

# The 2D Coordinates for cluster centroids in latent space
voronoi_sites = pd.read_csv("data_in_csv/action_space/action_space_voronoi_sites.csv")
CENTROIDS = np.array(voronoi_sites[['x', 'y']].values)

# Semantic labels for the three latent clusters
CLUSTER_IDS = np.array([
    "High-intensity work",
    "Rest",
    "Low-intensity generation",
])

# Per-centroid behavior feature means (one row per centroid)
means_df = pd.read_csv("data_in_csv/action_space/action_space_means.csv")
FEATURE_MEANS = means_df[BEHAVIORS].to_numpy()

# Full manifold points (for convex hull)
MANIFOLD_POINTS_FILE = "data_in_csv/action_space/action_manifold_points.csv"
manifold_points_df = pd.read_csv(MANIFOLD_POINTS_FILE)
ALL_POINTS = manifold_points_df[['x', 'y']].values

# Matrix A (Affordance) - 9x4 (behavior x internal state)
interaction_matrix = pd.read_csv(
    "interaction_matrices/behavioral+internal",
    header=None,
    sep='\s+'
)
A = interaction_matrix.to_numpy(dtype=float)


@dataclass
class LinearEq:
    intercept: float
    coef: Dict[str, float]
    sigma: float
    norm: Optional[str] = None
    mean: Optional[Dict[str, float]] = None
    scale: Optional[Dict[str, float]] = None
    min: Optional[Dict[str, float]] = None
    range: Optional[Dict[str, float]] = None

    def _normalize(self, x: Dict[str, float]) -> Dict[str, float]:
        if self.norm == "minmax" and self.min is not None and self.range is not None:
            out = {}
            for k, v in x.items():
                lo = self.min.get(k, 0.0)
                r = self.range.get(k, 1.0)
                out[k] = (float(v) - lo) / r if r != 0 else 0.0
            return out
        if self.mean is None or self.scale is None:
            return x
        out = {}
        for k, v in x.items():
            mu = self.mean.get(k, 0.0)
            s = self.scale.get(k, 1.0)
            out[k] = (float(v) - mu) / s if s != 0 else 0.0
        return out

    def predict(self, x: Dict[str, float], add_noise: bool = False) -> float:
        x_use = self._normalize(x)
        y = self.intercept
        for k, b in self.coef.items():
            y += b * float(x_use.get(k, 0.0))
        if add_noise and self.sigma > 0:
            y += float(np.random.normal(0.0, self.sigma))
        return float(y)


@dataclass
class TransitionModel:
    eq_hours: LinearEq
    eq_bedstd: LinearEq
    eq_energy: LinearEq
    eq_sq: LinearEq
    eq_wake_yesterday: Optional[LinearEq] = None

    @staticmethod
    def from_dict(data: Dict[str, object]) -> "TransitionModel":
        def _eq(key: str) -> LinearEq:
            d = data[key]
            return LinearEq(
                intercept=float(d["intercept"]),
                coef={k: float(v) for k, v in d["coef"].items()},
                sigma=float(d.get("sigma", 0.0)),
                norm=d.get("norm"),
                mean=d.get("mean"),
                scale=d.get("scale"),
                min=d.get("min"),
                range=d.get("range"),
            )
        return TransitionModel(
            eq_hours=_eq("eq_hours"),
            eq_bedstd=_eq("eq_bedstd"),
            eq_energy=_eq("eq_energy"),
            eq_sq=_eq("eq_sq"),
            eq_wake_yesterday=_eq("eq_wake_yesterday") if "eq_wake_yesterday" in data else None,
        )


def _load_regression_pack(path: str) -> Optional[TransitionModel]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    return TransitionModel.from_dict(data)


def _load_regression_stats(path: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return {}, {}
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    means = df.mean(numeric_only=True).to_dict()
    stds = df.std(numeric_only=True).replace(0, np.nan).to_dict()
    return means, stds


# =============================================================================
# OBSERVER: Value Function and TD-Error Tracking
# =============================================================================

@dataclass
class Observer:
    """
    The Observer maintains a value function and computes TD-errors.
    
    This is the foundation for:
    1. Surprise detection (large |δ| indicates unexpected outcome)
    2. Cognitive Reflex triggers (persistent high |δ| → model suspect)
    3. Master learning signal (δ as training signal)
    """
    gamma: float = 0.95          # Discount factor
    alpha_v: float = 0.1         # Value learning rate
    state_resolution: float = 0.5  # Discretization granularity for tabular V(s)
    
    # Learned value function (tabular)
    value_table: Dict[Tuple, float] = field(default_factory=dict)
    
    # History tracking
    td_errors: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    
    def _state_key(self, state: np.ndarray) -> Tuple:
        """Discretize continuous state for tabular value function."""
        discretized = np.round(state / self.state_resolution) * self.state_resolution
        return tuple(discretized)
    
    def get_value(self, state: np.ndarray) -> float:
        """Get estimated value of a state."""
        key = self._state_key(state)
        return self.value_table.get(key, 0.0)
    
    def compute_reward(
        self,
        state: np.ndarray,
        ideal: np.ndarray,
        action_idx: int,
        fatigue: float,
        lambda_variance: float = 0.1,
        lambda_fatigue: float = 0.2,
    ) -> float:
        """
        Compute reward for current transition.
        
        Reward = progress toward ideal - variance penalty - fatigue penalty
        
        This encodes: "Get close to ideal state sustainably"
        """
        # Progress toward ideal (negative distance)
        distance_to_ideal = np.linalg.norm(state - ideal)
        progress_reward = -distance_to_ideal
        
        # Variance penalty (prefer stable states)
        # Use state magnitude as proxy for deviation from neutral
        state_variance = np.var(state)
        variance_penalty = lambda_variance * state_variance
        
        # Fatigue penalty (sustainable operation)
        fatigue_penalty = lambda_fatigue * fatigue
        
        reward = progress_reward - variance_penalty - fatigue_penalty
        return float(reward)
    
    def update(
        self,
        state_prev: np.ndarray,
        state_curr: np.ndarray,
        reward: float,
    ) -> float:
        """
        Compute TD-error and update value function.
        
        δ = r + γV(s') - V(s)
        V(s) ← V(s) + α·δ
        
        Returns the TD-error for use by Cognitive Reflex.
        """
        key_prev = self._state_key(state_prev)
        key_curr = self._state_key(state_curr)
        
        V_prev = self.value_table.get(key_prev, 0.0)
        V_curr = self.value_table.get(key_curr, 0.0)
        
        # TD-error
        td_error = reward + self.gamma * V_curr - V_prev
        
        # Update value estimate
        self.value_table[key_prev] = V_prev + self.alpha_v * td_error
        
        # Store history
        self.td_errors.append(td_error)
        self.rewards.append(reward)
        self.values.append(V_prev)
        
        return td_error
    
    def get_surprise_level(self, window: int = 10) -> float:
        """Get mean absolute TD-error over recent window."""
        if not self.td_errors:
            return 0.0
        recent = self.td_errors[-window:]
        return float(np.mean(np.abs(recent)))
    
    def check_cognitive_reflex(
        self,
        threshold_spike: float = 0.5,
        threshold_persistent: float = 0.3,
        window: int = 10,
    ) -> Optional[str]:
        """
        Check if TD-error pattern warrants plasticity adjustment.
        
        Returns:
            None: No action needed
            "increase_exploration": Policy might be wrong, explore more
            "recompute_manifold": Model might be wrong, update map
        """
        if len(self.td_errors) < 2:
            return None
        
        current_error = abs(self.td_errors[-1])
        mean_recent_error = self.get_surprise_level(window)
        
        if current_error > threshold_spike:
            if mean_recent_error > threshold_persistent:
                # Persistent high surprise → model suspect
                return "recompute_manifold"
            else:
                # Spike surprise → policy wrong
                return "increase_exploration"
        
        return None
    
    def reset(self):
        """Clear history but keep learned values."""
        self.td_errors.clear()
        self.rewards.clear()
        self.values.clear()


# =============================================================================
# AEGIS MAP: Main Agent Class
# =============================================================================

class AegisMap:
    def __init__(
        self,
        centroids,
        labels,
        A_matrix,
        feature_means,
        will_power,
        ideal_vector,
        homeostatic_gamma=0.05,
        rest_state=None,
        internal_dim_names=None,
        regression_model=None,
        regression_stats=None,
        use_regression=False,
        regression_noise=False,
    ):
        self.will_power = will_power
        self.ideal_vector = np.asarray(ideal_vector, dtype=float)
        self.homeostatic_gamma = float(homeostatic_gamma)
        self.rest_state = np.zeros_like(self.ideal_vector) if rest_state is None else np.asarray(rest_state, dtype=float)
        self.centroids = centroids
        self.labels = labels
        self.A = A_matrix
        self.feature_means = feature_means
        self.internal_dim_names = internal_dim_names or [f"dim {i}" for i in range(4)]
        self.feature_colors = None
        self.speed = 1.0
        
        # Regression model
        self.regression_model = regression_model
        self.use_regression = use_regression and regression_model is not None
        self.regression_noise = regression_noise
        self.regression_means = regression_stats[0] if regression_stats else {}
        self.regression_stds = regression_stats[1] if regression_stats else {}
        self.last_regression_pred = {}
        self.context = {
            "hours_slept": float(self.regression_means.get("hours_slept", 0.0)),
            "bedtime_std_4_days": float(self.regression_means.get("bedtime_std_4_days", 0.0)),
            "Waking_Energy_Yesterday": float(self.regression_means.get("Waking_Energy_Yesterday", 0.0)),
            "NIC": float(self.regression_means.get("NIC", 0.0)),
            "CAF": float(self.regression_means.get("CAF", 0.0)),
        }

        # Internal state
        self.wp_fatigue = 0.0
        self.internal_vector = None
        self.internal_magnitude = None
        self.last_action = None
        self.last_optimal_idx = None
        self.last_deterministic_idx = None
        
        # History
        self.state_history = []
        self.magnitude_history = []
        self.action_history = []
        
        # === NEW: Observer for TD-error tracking ===
        self.observer = Observer(gamma=0.95, alpha_v=0.1, state_resolution=0.5)
        self.last_td_error = 0.0
        self.last_reward = 0.0
        self.cognitive_reflex_signal = None
        
        # Convex hull for boundary
        self.hull = ConvexHull(ALL_POINTS)
        hull_points = ALL_POINTS[self.hull.vertices]
        self.hull_path = Path(hull_points)
        
        # Grid for Voronoi rendering
        self.grid_res = 200
        x_min, x_max = ALL_POINTS[:, 0].min() - 0.05, ALL_POINTS[:, 0].max() + 0.05
        y_min, y_max = ALL_POINTS[:, 1].min() - 0.05, ALL_POINTS[:, 1].max() + 0.05
        self.xx, self.yy = np.meshgrid(
            np.linspace(x_min, x_max, self.grid_res),
            np.linspace(y_min, y_max, self.grid_res)
        )
        self.grid_points = np.c_[self.xx.ravel(), self.yy.ravel()]

    # -------------------------------------------------------------------------
    # Coordinate transformations
    # -------------------------------------------------------------------------
    
    def _unit01_to_internal(self, y01: float) -> float:
        """[0,1] -> [-2,2]"""
        return float(np.clip(4.0 * (float(y01) - 0.5), -2.0, 2.0))

    def _internal_to_unit01(self, z: float) -> float:
        """[-2,2] -> [0,1]"""
        return float(np.clip(0.5 + float(z) / 4.0, 0.0, 1.0))

    def _bound_state(self, v):
        """Bound each internal dimension to [-2, 2]."""
        x = np.asarray(v, dtype=float).copy()
        x[0] = np.clip(x[0], -2.0, 2.0)  # sleep quality
        x[1] = np.clip(x[1], -2.0, 2.0)  # waking energy
        x[2] = np.clip(x[2], -2.0, 2.0)  # dominant emotion (valence)
        x[3] = np.clip(x[3], -2.0, 2.0)  # emotional intensity
        return x

    # -------------------------------------------------------------------------
    # Regression-based state prediction
    # -------------------------------------------------------------------------

    def _predict_regression_state(self, action_vector):
        """Predict next state using regression models."""
        if not self.regression_model:
            return None

        behavior_inputs = {name: float(action_vector[i]) for i, name in enumerate(BEHAVIORS)}
        substances = {
            "NIC": float(self.context.get("NIC", 0.0)),
            "CAF": float(self.context.get("CAF", 0.0)),
        }
        x_behaviors = {**behavior_inputs, **substances}

        # Predict next context
        hours = self.regression_model.eq_hours.predict(x_behaviors, add_noise=self.regression_noise)
        bedstd = self.regression_model.eq_bedstd.predict(x_behaviors, add_noise=self.regression_noise)
        wake_y = float(self.context.get("Waking_Energy_Yesterday", 0.5))

        hours = np.clip(hours, 0.0, 1.0)
        bedstd = np.clip(bedstd, 0.0, 1.0)
        wake_y = np.clip(wake_y, 0.0, 1.0)

        self.context.update({
            "hours_slept": hours,
            "bedtime_std_4_days": bedstd,
            "Waking_Energy_Yesterday": wake_y,
        })

        # Predict energy and sleep quality
        x_context = {
            "hours_slept": hours,
            "bedtime_std_4_days": bedstd,
            "Waking_Energy_Yesterday": wake_y,
            **substances,
        }
        energy = self.regression_model.eq_energy.predict(x_context, add_noise=self.regression_noise)
        sleep_q = self.regression_model.eq_sq.predict(x_context, add_noise=self.regression_noise)

        sleep_internal = self._unit01_to_internal(np.clip(sleep_q, 0.0, 1.0))
        energy_internal = self._unit01_to_internal(np.clip(energy, 0.0, 1.0))

        self.last_regression_pred = {
            "hours_slept": hours,
            "bedtime_std_4_days": bedstd,
            "Waking_Energy_Yesterday": wake_y,
            "energy_raw": energy,
            "sleep_q_raw": sleep_q,
            "sleep_internal": sleep_internal,
            "energy_internal": energy_internal,
        }
        return sleep_internal, energy_internal

    # -------------------------------------------------------------------------
    # Utility computation
    # -------------------------------------------------------------------------

    def _utilities_to_probs(self, utilities):
        """Convert raw utilities into softmax probabilities."""
        u = utilities - np.max(utilities)
        exp_u = np.exp(u)
        return exp_u / np.sum(exp_u)

    def calculate_utility_components(self, internal_state):
        """Calculate needs, ideal, and combined utilities."""
        internal_bounded = self._bound_state(internal_state)
        ideal_bounded = self._bound_state(self.ideal_vector)

        wp_eff = self.will_power * (1.0 - self.wp_fatigue)

        behavior_weights = self.A @ internal_bounded
        behavior_weights_ideal = self.A @ ideal_bounded

        needs = self.feature_means @ behavior_weights
        ideal = self.feature_means @ behavior_weights_ideal
        utilities = wp_eff * ideal + (1 - wp_eff) * needs

        return needs, ideal, utilities

    # -------------------------------------------------------------------------
    # Action selection with optional constraint checking
    # -------------------------------------------------------------------------

    def select_action_with_constraints(
        self,
        utilities: np.ndarray,
        min_energy: float = -1.5,
        predict_horizon: int = 1,
    ) -> int:
        """
        Select highest-utility action that doesn't violate energy constraint.
        
        This is the MPC-style Emissary: optimize subject to safety.
        """
        sorted_indices = np.argsort(utilities)[::-1]  # highest utility first
        
        for idx in sorted_indices:
            # Simulate this action's effect on energy
            mu = self.feature_means[idx]
            
            # Quick energy prediction (simplified)
            EFFORT = np.array([0, 1, 2, 3, 7])
            effort = float(np.maximum(0.0, mu[EFFORT]).sum())
            
            if self.use_regression:
                # Use regression to predict next energy
                predicted = self._predict_regression_state(mu)
                if predicted is not None:
                    _, predicted_energy = predicted
                else:
                    predicted_energy = self.internal_vector[1] - 0.1 * effort
            else:
                predicted_energy = self.internal_vector[1] - 0.1 * effort
            
            if predicted_energy >= min_energy:
                return idx  # First safe action by utility
        
        # No safe action exists—return least-bad (last in sorted = lowest utility but checked last)
        # Actually return the one with highest predicted energy
        return sorted_indices[-1]

    # -------------------------------------------------------------------------
    # State update (one day step)
    # -------------------------------------------------------------------------

    def _inject_noise(self, action, needs):
        """Add behavioral noise to action."""
        optimal = np.asarray(action, dtype=float)
        deterministic = np.asarray(needs, dtype=float)
        beta = 0.15
        mixed = (1 - beta) * optimal + beta * deterministic
        noise = np.random.normal(0.0, 0.02, size=mixed.shape)
        return mixed + noise

    def _internal_update(self, utilities, needs, MPC, eta=0.10) -> None:
        """
        Execute one day step: select action, update state, compute TD-error.
        """
        # Store previous state for TD computation
        state_prev = self.internal_vector.copy() if self.internal_vector is not None else None
        
        if MPC:
            optimal_action = self.select_action_with_constraints(utilities)
        else:
            optimal_action = int(np.argmax(utilities))
        deterministic_action = int(np.argmax(needs))

        optimal_vector = np.asarray(self.feature_means[optimal_action], dtype=float)
        deterministic_vector = np.asarray(self.feature_means[deterministic_action], dtype=float)

        action = self._inject_noise(optimal_vector, deterministic_vector)
        mu_pos = action

        # Willpower fatigue update
        override = np.linalg.norm(optimal_vector - deterministic_vector)
        override_norm = override / (override + 1.0)
        alpha = 0.2
        self.wp_fatigue = (1 - alpha) * self.wp_fatigue + alpha * override_norm
        self.wp_fatigue = float(np.clip(self.wp_fatigue, 0.0, 1.0))

        wp_eff = self.will_power * (1.0 - self.wp_fatigue)
        cost_mult = 1.0 + wp_eff * override

        # Dimension indices
        SLEEP_DIM = 0
        ENERGY_DIM = 1
        VAL_DIM = 2
        INT_DIM = 3

        # Behavior indices
        EFFORT = np.array([0, 1, 2, 3, 7])
        POS = np.array([2, 3, 7])
        NEG = np.array([4, 5, 6])
        STIM = np.array([0, 1, 3, 7, 4])
        CALM = np.array([5, 6])

        new_state = np.asarray(self.internal_vector, dtype=float).copy()
        effort = float(np.maximum(0.0, mu_pos[EFFORT]).sum())
        s = float(self.speed)
        eta_energy = s * eta
        eta_sleep = s * (0.5 * eta)

        # Apply regression or thermo update
        predicted = None
        if self.use_regression:
            predicted = self._predict_regression_state(mu_pos)

        if predicted is not None:
            sleep_internal, energy_internal = predicted
            new_state[SLEEP_DIM] = float(sleep_internal)
            new_state[ENERGY_DIM] = float(energy_internal)
            # Apply within-day drain ON TOP of regression baseline
            new_state[ENERGY_DIM] -= eta_energy * cost_mult * effort
            new_state[SLEEP_DIM] -= eta_sleep * cost_mult * effort
        else:
            # Thermo-only: single drain
            new_state[ENERGY_DIM] -= eta_energy * cost_mult * effort
            new_state[SLEEP_DIM] -= eta_sleep * cost_mult * effort

        # Valence dynamics
        eta_v = s * 0.15
        new_state[VAL_DIM] += eta_v * (mu_pos[POS].sum() - mu_pos[NEG].sum())

        # Arousal dynamics
        eta_a = s * 0.12
        new_state[INT_DIM] += eta_a * (mu_pos[STIM].sum() - mu_pos[CALM].sum())

        # Homeostasis (only for valence and arousal)
        mask = np.array([0.0, 0.0, 1.0, 1.0], dtype=float)
        homeo = (s * self.homeostatic_gamma) * mask * (self.rest_state - new_state)
        new_state += homeo

        # Bound state
        new_state = self._bound_state(new_state)
        self.internal_vector = new_state
        self.internal_magnitude = float(np.linalg.norm(new_state))

        # Store action info
        self.last_action = action
        self.last_optimal_idx = optimal_action
        self.last_deterministic_idx = deterministic_action

        # Update context for next day
        self.context["Waking_Energy_Yesterday"] = self._internal_to_unit01(new_state[ENERGY_DIM])

        # === NEW: Observer update (TD-error computation) ===
        if state_prev is not None:
            # Compute reward
            self.last_reward = self.observer.compute_reward(
                state=new_state,
                ideal=self.ideal_vector,
                action_idx=optimal_action,
                fatigue=self.wp_fatigue,
            )
            
            # Compute TD-error and update value function
            self.last_td_error = self.observer.update(
                state_prev=state_prev,
                state_curr=new_state,
                reward=self.last_reward,
            )
            
            # Check cognitive reflex
            self.cognitive_reflex_signal = self.observer.check_cognitive_reflex()
            
            if self.cognitive_reflex_signal:
                print(f"  [Cognitive Reflex] Signal: {self.cognitive_reflex_signal}")

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------

    def compute_weighted_voronoi(self, utilities):
        """Compute utility-weighted Voronoi diagram."""
        dx = self.grid_points[:, np.newaxis, 0] - self.centroids[np.newaxis, :, 0]
        dy = self.grid_points[:, np.newaxis, 1] - self.centroids[np.newaxis, :, 1]
        dist_sq = dx**2 + dy**2

        probs = self._utilities_to_probs(utilities)
        shrink = 1.0 - probs
        weighted_dist = dist_sq * shrink.reshape(1, -1)
        winners = np.argmin(weighted_dist, axis=1).reshape(self.xx.shape)

        inside = self.hull_path.contains_points(self.grid_points).reshape(self.xx.shape)
        winners[~inside] = -1

        return winners

    def plot_state_history(self, ax_state=None):
        """Plot internal state trajectory."""
        if not self.state_history:
            return

        hist = np.asarray(self.state_history, dtype=float)
        mu = hist.mean(axis=0)
        sigma = hist.std(axis=0)
        sigma[sigma < 1e-8] = 1.0
        norm_hist = (hist - mu) / sigma

        if ax_state is None:
            _, ax_state = plt.subplots()

        ax_state.clear()
        n_dims = min(4, norm_hist.shape[1])
        for d in range(n_dims):
            label = self.internal_dim_names[d] if d < len(self.internal_dim_names) else f"dim {d}"
            color = self.feature_colors[d] if self.feature_colors and d < len(self.feature_colors) else None
            ax_state.plot(norm_hist[:, d], label=label, color=color)
        ax_state.set_xlabel("Step (Day)")
        ax_state.set_ylabel("Z-score")
        ax_state.set_title("Internal state over time")
        ax_state.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)

    def plot_magnitude_history(self, ax_mag=None):
        """Plot state magnitude over time."""
        if not self.magnitude_history:
            return

        if ax_mag is None:
            _, ax_mag = plt.subplots()

        ax_mag.clear()
        ax_mag.plot(self.magnitude_history, color="black", linestyle="--")
        ax_mag.set_xlabel("Step (Day)")
        ax_mag.set_ylabel("Magnitude")
        ax_mag.set_title("State magnitude")

    def plot_td_error_history(self, ax_td=None):
        """Plot TD-error history with surprise threshold."""
        if not self.observer.td_errors:
            return

        if ax_td is None:
            _, ax_td = plt.subplots()

        ax_td.clear()
        errors = np.array(self.observer.td_errors)
        t = np.arange(len(errors))
        
        # Plot TD-error
        ax_td.plot(t, errors, color="purple", label="TD-error", linewidth=1.5)
        ax_td.axhline(0, color="gray", linestyle="--", alpha=0.5)
        
        # Plot surprise threshold
        threshold = 0.5
        ax_td.axhline(threshold, color="red", linestyle=":", alpha=0.7, label=f"Spike threshold ({threshold})")
        ax_td.axhline(-threshold, color="red", linestyle=":", alpha=0.7)
        
        # Shade high-surprise regions
        high_surprise = np.abs(errors) > threshold
        if np.any(high_surprise):
            ax_td.fill_between(t, errors.min(), errors.max(), where=high_surprise, 
                             color="red", alpha=0.1, label="High surprise")
        
        ax_td.set_xlabel("Step (Day)")
        ax_td.set_ylabel("TD-error (δ)")
        ax_td.set_title("Prediction error over time")
        ax_td.legend(loc="upper right", fontsize=7)

    def plot_reward_history(self, ax_reward=None):
        """Plot reward history."""
        if not self.observer.rewards:
            return

        if ax_reward is None:
            _, ax_reward = plt.subplots()

        ax_reward.clear()
        rewards = np.array(self.observer.rewards)
        t = np.arange(len(rewards))
        
        ax_reward.plot(t, rewards, color="green", label="Reward")
        ax_reward.axhline(0, color="gray", linestyle="--", alpha=0.5)
        
        # Plot cumulative reward
        ax_reward_twin = ax_reward.twinx()
        cumulative = np.cumsum(rewards)
        ax_reward_twin.plot(t, cumulative, color="darkgreen", linestyle="--", alpha=0.7, label="Cumulative")
        ax_reward_twin.set_ylabel("Cumulative", color="darkgreen")
        
        ax_reward.set_xlabel("Step (Day)")
        ax_reward.set_ylabel("Reward")
        ax_reward.set_title("Reward over time")

    def plot_action_history(self, ax_actions=None, normalize=True):
        """Plot action frequency distribution."""
        if ax_actions is None:
            _, ax_actions = plt.subplots()

        ax_actions.clear()
        total = len(self.action_history)
        counts = np.bincount(self.action_history, minlength=len(self.labels)) if total > 0 else np.zeros(len(self.labels))
        
        if normalize and total > 0:
            values = counts / total
            ylabel = "Proportion"
        else:
            values = counts
            ylabel = "Count"

        x = np.arange(len(self.labels))
        ax_actions.bar(x, values, color="tab:blue", alpha=0.7)
        ax_actions.set_xticks(x)
        ax_actions.set_xticklabels(self.labels, rotation=20, ha="right")
        ax_actions.set_ylabel(ylabel)
        ax_actions.set_title("Action frequency")
        
        if total > 0:
            for i, (v, c) in enumerate(zip(values, counts)):
                ax_actions.text(i, v, f"{int(c)}", ha="center", va="bottom", fontsize=8)

    def plot(self, state_vector, ax=None):
        """Plot warped action space."""
        needs, ideal, utilities = self.calculate_utility_components(state_vector)
        region_map = self.compute_weighted_voronoi(utilities)
        masked_regions = np.ma.masked_where(region_map < 0, region_map)

        valid = region_map >= 0
        if np.any(valid):
            counts = np.bincount(region_map[valid].ravel(), minlength=len(self.labels))
            area_fracs = counts / counts.sum()
        else:
            area_fracs = np.zeros(len(self.labels))

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        ax.clear()
        ax.set_title("Warped action space", fontsize=12, fontweight="bold")
        cmap = plt.get_cmap('tab20', len(self.labels))
        ax.contourf(
            self.xx, self.yy, masked_regions,
            levels=np.arange(len(self.labels) + 1) - 0.5,
            cmap=cmap, alpha=0.6,
        )
        ax.scatter(self.centroids[:, 0], self.centroids[:, 1], c='black', s=30)

        # Legend table
        table_data = []
        row_colors = []
        for i, name in enumerate(self.labels):
            perc = float(area_fracs[i]) * 100.0
            table_data.append([name, f"{perc:.1f}%"])
            row_colors.append(cmap(i))

        action_table = ax.table(
            cellText=[["Action", "Percentage"]] + table_data,
            cellLoc="center", loc="left",
            bbox=[-0.55, 0.15, 0.45, 0.7],
        )
        for (row, col), cell in action_table.get_celld().items():
            cell.set_edgecolor("black")
            if row == 0:
                cell.set_text_props(weight="bold", color="black")
                cell.set_facecolor("lightgray")
            else:
                idx = row - 1
                if idx < len(row_colors):
                    cell.set_facecolor(row_colors[idx])
                cell.set_text_props(color="black")

        # Show last action info
        if self.last_optimal_idx is not None and self.last_deterministic_idx is not None:
            info = f"Chosen: {self.last_optimal_idx} (needs-only: {self.last_deterministic_idx})"
            ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=8, va="bottom", ha="left")

        # Show TD-error and reward
        if self.observer.td_errors:
            td_info = f"δ={self.last_td_error:.3f}  r={self.last_reward:.3f}  surprise={self.observer.get_surprise_level():.3f}"
            ax.text(0.98, 0.02, td_info, transform=ax.transAxes, fontsize=8, va="bottom", ha="right", color="purple")

        for simplex in self.hull.simplices:
            ax.plot(ALL_POINTS[simplex, 0], ALL_POINTS[simplex, 1], 'k--', lw=1)
        ax.set_axis_off()

    def print_interaction_matrix(self):
        """Print the A matrix."""
        df = pd.DataFrame(self.A, index=BEHAVIORS, columns=self.internal_dim_names)
        print("\nInteraction matrix A (behaviors x internal state):")
        print(df.to_string())

    def print_step_debug(self, internal_state, step_idx=None):
        """Print debug info for a step."""
        header = f"\n--- Step {step_idx} ---" if step_idx is not None else "\n--- Step ---"
        print(header)
        
        needs, ideal, utilities = self.calculate_utility_components(internal_state)
        print(f"  needs = {needs}")
        print(f"  ideal = {ideal}")
        print(f"  utilities = {utilities}")
        print(f"  wp_eff = {self.will_power * (1.0 - self.wp_fatigue):.3f}")

        optimal_action = int(np.argmax(utilities))
        deterministic_action = int(np.argmax(needs))
        optimal_vector = self.feature_means[optimal_action]
        deterministic_vector = self.feature_means[deterministic_action]

        beta = 0.15
        mixed = (1 - beta) * optimal_vector + beta * deterministic_vector
        noise = np.random.normal(0.0, 0.02, size=optimal_vector.shape)
        observed_vector = mixed + noise

        return needs, ideal, utilities, observed_vector


# =============================================================================
# CLI MODE
# =============================================================================

def run_cli(aegis, initial_state, steps, seed=None):
    """Run simulation in CLI mode."""
    if seed is not None:
        np.random.seed(seed)

    internal_state = np.asarray(initial_state, dtype=float)
    print("Starting internal state:", internal_state)
    aegis.print_interaction_matrix()

    aegis.internal_vector = aegis._bound_state(internal_state.copy())
    aegis.internal_magnitude = np.linalg.norm(aegis.internal_vector)
    aegis.state_history = [aegis.internal_vector.copy()]
    aegis.magnitude_history = [aegis.internal_magnitude]

    for step_idx in range(steps):
        needs, ideal, utilities, observed_vector = aegis.print_step_debug(
            aegis.internal_vector, step_idx=step_idx + 1,
        )

        original_inject = aegis._inject_noise
        try:
            aegis._inject_noise = lambda *_: observed_vector
            aegis._internal_update(utilities, needs)
        finally:
            aegis._inject_noise = original_inject

        print(f"  chosen optimal = {aegis.last_optimal_idx}, deterministic = {aegis.last_deterministic_idx}")
        print(f"  new state = {aegis.internal_vector}")
        print(f"  reward = {aegis.last_reward:.4f}, TD-error = {aegis.last_td_error:.4f}")
        print(f"  surprise level = {aegis.observer.get_surprise_level():.4f}")
        
        if aegis.cognitive_reflex_signal:
            print(f"  >>> COGNITIVE REFLEX: {aegis.cognitive_reflex_signal}")


# =============================================================================
# MAIN (GUI MODE)
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aegis internal warping demo")
    parser.add_argument("--cli", action="store_true", help="Run CLI debug mode")
    parser.add_argument("--steps", type=int, default=3, help="Number of steps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--state", type=str, default="0,0,0,0", help="Initial state (comma-separated)")
    parser.add_argument("--will", type=float, default=0.5, help="Will power")
    parser.add_argument("--use-regression", action="store_true", help="Use regression models")
    parser.add_argument("--regression-noise", action="store_true", help="Add regression noise")
    args = parser.parse_args()

    internal_dim_names = [
        "Sleep quality",
        "Waking energy",
        "Dominant emotion",
        "Emotional intensity / stress",
    ]
    
    regression_model = _load_regression_pack(REGRESSION_PACK_PATH)
    regression_stats = _load_regression_stats(REGRESSION_STATS_PATH)
    
    if args.use_regression and regression_model is None:
        print(f"Warning: regression pack not found at {REGRESSION_PACK_PATH}")
    
    aegis = AegisMap(
        CENTROIDS, CLUSTER_IDS, A, FEATURE_MEANS,
        0.5, [0, 1, 1, 1],
        internal_dim_names=internal_dim_names,
        regression_model=regression_model,
        regression_stats=regression_stats,
        use_regression=args.use_regression,
        regression_noise=args.regression_noise,
    )

    if args.cli:
        initial_state = np.array([float(x.strip()) for x in args.state.split(",") if x.strip()])
        aegis.will_power = args.will
        run_cli(aegis, initial_state=initial_state, steps=args.steps, seed=args.seed)
        raise SystemExit(0)

    # GUI setup
    feature_cmap = plt.get_cmap("tab10")
    feature_colors = [feature_cmap(i % feature_cmap.N) for i in range(len(internal_dim_names))]
    aegis.feature_colors = feature_colors

    # Layout: 5 rows (added TD-error plot)
    fig = plt.figure(figsize=(18, 12))
    grid = fig.add_gridspec(5, 2, height_ratios=[3, 1, 1, 1, 1])
    ax = fig.add_subplot(grid[0, :])
    ax_state = fig.add_subplot(grid[1, 0])
    ax_mag = fig.add_subplot(grid[1, 1])
    ax_td = fig.add_subplot(grid[2, 0])        # NEW: TD-error plot
    ax_reward = fig.add_subplot(grid[2, 1])    # NEW: Reward plot
    ax_actions = fig.add_subplot(grid[3, :])
    ax_subset = fig.add_subplot(grid[4, :])
    
    plt.subplots_adjust(left=0.3, right=0.68, bottom=0.15, hspace=0.6)

    # Initial state
    state = np.array([1.0, 1.0, 0.0, 0.0])
    aegis.plot(state, ax=ax)
    aegis.print_interaction_matrix()

    # Slider axes
    ax_sleep = plt.axes([0.80, 0.30, 0.18, 0.025])
    ax_energy = plt.axes([0.80, 0.26, 0.18, 0.025])
    ax_emotion = plt.axes([0.80, 0.22, 0.18, 0.025])
    ax_stress = plt.axes([0.80, 0.18, 0.18, 0.025])
    ax_caf = plt.axes([0.80, 0.14, 0.18, 0.025])
    ax_nic = plt.axes([0.80, 0.10, 0.18, 0.025])
    ax_will = plt.axes([0.80, 0.06, 0.18, 0.025])
    ax_speed = plt.axes([0.80, 0.02, 0.18, 0.025])
    ax_wpinfo = plt.axes([0.80, 0.34, 0.18, 0.06])
    ax_wpinfo.axis("off")
    ax_step = plt.axes([0.05, 0.90, 0.15, 0.04])
    ax_reset = plt.axes([0.05, 0.85, 0.15, 0.04])  # NEW: Reset button
    ax_subset_selector = plt.axes([0.05, 0.02, 0.18, 0.15])

    # Sliders
    s_sleep = Slider(ax_sleep, "Sleep quality", -2.0, 2.0, valinit=1.0)
    s_energy = Slider(ax_energy, "Waking energy", -2.0, 2.0, valinit=1.0)
    s_emotion = Slider(ax_emotion, "Dominant emotion", -2.0, 2.0, valinit=0.0)
    s_stress = Slider(ax_stress, "Emotional intensity", -2.0, 2.0, valinit=0.0)
    s_caf = Slider(ax_caf, "CAF", 0.0, 1.0, valinit=aegis.context.get("CAF", 0.0))
    s_nic = Slider(ax_nic, "NIC", 0.0, 1.0, valinit=aegis.context.get("NIC", 0.0))
    s_will = Slider(ax_will, "Will Power", 0.0, 1.0, valinit=0.5)
    s_speed = Slider(ax_speed, "Speed", 0.0, 2.0, valinit=1.0)
    b_step = Button(ax_step, "Step (1 Day)")
    b_reset = Button(ax_reset, "Reset")
    subset_selector = RadioButtons(ax_subset_selector, internal_dim_names, active=0)
    ax_subset_selector.set_title("Subset feature", fontsize=8)

    def _style_slider(slider, color):
        slider.poly.set_facecolor(color)
        slider.poly.set_alpha(0.6)
        slider.track.set_facecolor("lightgray")
        slider.label.set_color(color)
        slider.valtext.set_color(color)

    _style_slider(s_sleep, feature_colors[0])
    _style_slider(s_energy, feature_colors[1])
    _style_slider(s_emotion, feature_colors[2])
    _style_slider(s_stress, feature_colors[3])
    _style_slider(s_caf, "black")
    _style_slider(s_nic, "black")
    _style_slider(s_speed, "black")

    for label, color in zip(subset_selector.labels, feature_colors):
        label.set_color(color)

    wp_eff_line = ax_will.axvline(0.5, color="red", lw=1.5, alpha=0.8)
    wp_text = ax_wpinfo.text(0.0, 0.5, "", fontsize=8, va="center", ha="left")
    subset_feature_idx = [0]

    def _update_wp_display():
        wp_eff = float(aegis.will_power) * (1.0 - float(aegis.wp_fatigue))
        wp_eff = float(np.clip(wp_eff, 0.0, 1.0))
        wp_eff_line.set_xdata([wp_eff, wp_eff])
        if s_will.val > 0:
            ratio = wp_eff / max(1e-6, float(s_will.val))
            s_will.poly.set_alpha(0.25 + 0.75 * float(np.clip(ratio, 0.0, 1.0)))
        else:
            s_will.poly.set_alpha(0.25)
        wp_text.set_text(
            f"wp_eff = will × (1 - fatigue)\n"
            f"= {aegis.will_power:.2f} × (1 - {aegis.wp_fatigue:.2f})\n"
            f"= {wp_eff:.2f}"
        )

    def update(_):
        state[:] = [s_sleep.val, s_energy.val, s_emotion.val, s_stress.val]
        aegis.will_power = s_will.val
        aegis.speed = s_speed.val
        aegis.context["CAF"] = float(s_caf.val)
        aegis.context["NIC"] = float(s_nic.val)
        _update_wp_display()
        aegis.plot(state, ax=ax)
        fig.canvas.draw_idle()

    for s in (s_sleep, s_energy, s_emotion, s_stress, s_caf, s_nic, s_will, s_speed):
        s.on_changed(update)

    def on_subset_change(label):
        subset_feature_idx[0] = internal_dim_names.index(label)
        if aegis.state_history:
            hist = np.asarray(aegis.state_history, dtype=float)
            mu = hist.mean(axis=0)
            sigma = hist.std(axis=0)
            sigma[sigma < 1e-8] = 1.0
            norm_hist = (hist - mu) / sigma
            
            ax_subset.clear()
            idx = subset_feature_idx[0]
            ax_subset.plot(norm_hist[:, idx], label=internal_dim_names[idx], color=feature_colors[idx])
            ax_subset.set_xlabel("Step (Day)")
            ax_subset.set_ylabel("Z-score")
            ax_subset.set_title(f"{internal_dim_names[idx]} over time")
            ax_subset.legend(loc="upper right", fontsize=8)
        fig.canvas.draw_idle()

    subset_selector.on_clicked(on_subset_change)

    def on_step(event):
        # Initialize if needed
        if aegis.internal_vector is None:
            aegis.internal_vector = aegis._bound_state(state.copy())
            aegis.internal_magnitude = np.linalg.norm(aegis.internal_vector)
            aegis.state_history.append(aegis.internal_vector.copy())
            aegis.magnitude_history.append(aegis.internal_magnitude)

        aegis.will_power = s_will.val
        needs, ideal, utilities, observed_vector = aegis.print_step_debug(
            aegis.internal_vector, step_idx=len(aegis.state_history),
        )
        
        original_inject = aegis._inject_noise
        try:
            aegis._inject_noise = lambda *_: observed_vector
            aegis._internal_update(utilities, needs)
        finally:
            aegis._inject_noise = original_inject

        _update_wp_display()
        
        # Print Observer info
        print(f"  reward = {aegis.last_reward:.4f}")
        print(f"  TD-error = {aegis.last_td_error:.4f}")
        print(f"  surprise = {aegis.observer.get_surprise_level():.4f}")
        if aegis.cognitive_reflex_signal:
            print(f"  >>> COGNITIVE REFLEX: {aegis.cognitive_reflex_signal}")

        # Update history
        new_values = aegis.internal_vector.copy()
        aegis.state_history.append(new_values.copy())
        if aegis.internal_magnitude is not None:
            aegis.magnitude_history.append(aegis.internal_magnitude)
        if aegis.last_optimal_idx is not None:
            aegis.action_history.append(int(aegis.last_optimal_idx))

        # Update sliders silently
        sliders = [s_sleep, s_energy, s_emotion, s_stress]
        for s in sliders:
            s.eventson = False
        s_sleep.set_val(new_values[0])
        s_energy.set_val(new_values[1])
        s_emotion.set_val(new_values[2])
        s_stress.set_val(new_values[3])
        for s in sliders:
            s.eventson = True

        # Refresh plots
        state[:] = new_values
        aegis.plot(state, ax=ax)
        aegis.plot_state_history(ax_state=ax_state)
        aegis.plot_magnitude_history(ax_mag=ax_mag)
        aegis.plot_td_error_history(ax_td=ax_td)
        aegis.plot_reward_history(ax_reward=ax_reward)
        aegis.plot_action_history(ax_actions=ax_actions)
        on_subset_change(internal_dim_names[subset_feature_idx[0]])
        fig.canvas.draw_idle()

    def on_reset(event):
        # Reset all state
        aegis.internal_vector = None
        aegis.internal_magnitude = None
        aegis.wp_fatigue = 0.0
        aegis.state_history.clear()
        aegis.magnitude_history.clear()
        aegis.action_history.clear()
        aegis.observer.reset()
        aegis.last_td_error = 0.0
        aegis.last_reward = 0.0
        aegis.cognitive_reflex_signal = None
        aegis.last_optimal_idx = None
        aegis.last_deterministic_idx = None
        
        # Reset context
        aegis.context = {
            "hours_slept": float(aegis.regression_means.get("hours_slept", 0.0)),
            "bedtime_std_4_days": float(aegis.regression_means.get("bedtime_std_4_days", 0.0)),
            "Waking_Energy_Yesterday": float(aegis.regression_means.get("Waking_Energy_Yesterday", 0.0)),
            "NIC": float(s_nic.val),
            "CAF": float(s_caf.val),
        }
        
        # Reset sliders to initial
        sliders = [s_sleep, s_energy, s_emotion, s_stress]
        for s in sliders:
            s.eventson = False
        s_sleep.set_val(1.0)
        s_energy.set_val(1.0)
        s_emotion.set_val(0.0)
        s_stress.set_val(0.0)
        for s in sliders:
            s.eventson = True
        
        state[:] = [1.0, 1.0, 0.0, 0.0]
        _update_wp_display()
        
        # Clear plots
        aegis.plot(state, ax=ax)
        ax_state.clear()
        ax_mag.clear()
        ax_td.clear()
        ax_reward.clear()
        ax_actions.clear()
        ax_subset.clear()
        
        print("\n=== RESET ===\n")
        fig.canvas.draw_idle()

    b_step.on_clicked(on_step)
    b_reset.on_clicked(on_reset)

    plt.show()