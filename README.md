# Project Aegis: Spectrally-Guided Hierarchical Control for Behavioral Self-Regulation

> **Status:** Independent Research (Fall 2025 - Present)
> **Stack:** Python, PyTorch, Scikit-learn, Model Predictive Control (MPC), Spectral Graph Theory

## TL;DR
A comprehensive research project modeling human behavioral dynamics. I developed a pipeline that transforms abstract daily behavior into a navigable geometric manifold, then built a Hierarchical RL agent that navigates this space to balance long-term goals with immediate physiological constraints.

---

## 📄 Research Papers
This repository contains the code and documentation for two distinct phases of research:

1.  **[Phase I: Latent Action Space (Geometry)](papers/Final_paper_rd_CG.pdf)**  
    *Focus: Spectral Graph Theory & Manifold Learning (Math 437)*
2.  **[Phase II: Project Aegis (Control System)](papers/project_aegis_technical_report.pdf)**  
    *Focus: Hierarchical Reinforcement Learning & Control Theory (Independent Research)*

---

## Phase I: The Geometric Foundation
**Problem:** Traditional RL assumes discrete, independent actions. However, human behavior is continuous and relational (e.g., "Deep Work" is closer to "Light Study" than it is to "Sleep").
**Solution:** I used **Spectral Embedding** to construct a "Latent Action Space"—a map where the agent decides "where to go" rather than just "what to do."

### Key Technical Contributions
- **Data Ingestion:** Processed 81 days of 21-dimensional behavioral tracking data (sleep, work intensity, emotion, etc.).
- **Spectral Embedding:** Constructed a nearest-neighbor graph and applied the **Graph Laplacian**.
- **Manifold Learning:** Projected data to a 2D manifold via **Laplacian Eigenmaps**, preserving local neighborhood structure.
- **Space Partitioning:** Used **DBSCAN** to identify 6 behavioral archetypes and partitioned the continuous space via **Voronoi tessellation**.

---

## Phase II: The Hierarchical Control System
**Problem:** Humans often possess "akrasia"—knowing what is good for them but failing to do it. Standard RL agents don't suffer from burnout or willpower depletion.
**Solution:** I designed a hierarchical architecture separating **Strategic Planning** (Master) from **Tactical Execution** (Emissary).

### System Architecture
1.  **The Plant (Body):** Enforces thermodynamic constraints on physiological resources (energy, sleep debt).
2.  **The Emissary (Tactical):** Performs action selection via **Model Predictive Control (MPC)**. It tries to satisfy the Master's goals but will override them if resources are critically low.
3.  **The Master (Strategic):** Sets high-level objectives based on long-term rewards.
4.  **The Observer:** Tracks prediction errors via **TD-learning** to detect when the world model is stale.

### Key Findings
- **Willpower Dynamics:** Modeled "willpower" as a depletable resource. Sustained override of immediate needs accumulates "fatigue," eventually forcing the agent into recovery cycles.
- **Emergent Behavior:** The agent demonstrates realistic burnout and recovery cycles without hard-coded rules—simply by navigating the learned manifold under constraints.

---

## Technical Stack
- **Languages:** Python, NumPy
- **Machine Learning:** PyTorch (Neural approximations), Scikit-learn (Clustering/PCA)
- **Math/Theory:** Spectral Graph Theory (Laplacians), Convex Optimization, Voronoi Geometry
- **Control:** Model Predictive Control (MPC), PID Dynamics

## Usage
Install dependencies, then run the simulation demo.

```bash
pip install -r requirements.txt
```

### Simulation (GUI)
```bash
python aegis_actor_demos/internal_warping_of_behaviors.py
```
This launches an interactive Matplotlib GUI with sliders for internal state and context.

### Simulation (CLI debug)
```bash
python aegis_actor_demos/internal_warping_of_behaviors.py --cli --steps 5 --state "1,1,0,0" --seed 0
```
Use `--will` to set willpower, and `--use-regression` to enable the regression pack
(`regression_analysis/linear_equation_pack.json`) with stats from `data_in_csv/21_set_raw.csv`.
