# Latent Action Space: Spectral Methods for Behavioral Modeling

> **Focus:** Spectral Graph Theory, Computational Geometry
> **Stack:** Python, Scikit-learn, NumPy, Matplotlib

## TL;DR
A pipeline that transforms abstract, high-dimensional behavioral data into a navigable 2D geometric surface. By applying Spectral Embedding, this project allows an RL agent to understand "where to go" (geometry) rather than just "what to do" (discrete actions).

## The Problem
Traditional RL action spaces are often discrete and disjointed. High-dimensional behavioral data (sleep, work intensity, emotion) is difficult to structure in a way that allows an agent to understand the "closeness" of different states.

## Technical Approach
1.  **Data Ingestion:** 81 days of 21-dimensional behavioral tracking data.
2.  **Spectral Embedding:** Constructed a nearest-neighbor graph and applied the **Graph Laplacian**.
3.  **Dimensionality Reduction:** Projected data to a 2D manifold via **Laplacian Eigenmaps**, preserving local neighborhood structure.
4.  **Space Partitioning:** Used **DBSCAN** to identify 6 behavioral archetypes and partitioned the continuous space via **Voronoi tessellation**.

## Key Results
- Created a geometric action space where geometric proximity encodes behavioral similarity.
- Demonstrated that double normalization of the Normalized Laplacian (local node scaling + global degree weighting) provides unbiased structural representation.

## Research Paper
[Link to PDF](papers/Final_paper_rd_CG.pdf)