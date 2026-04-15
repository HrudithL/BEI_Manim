# Concept Visualization in Manim

Short educational animations built with Manim Community Edition to explain core statistics concepts visually.

## Project Scope

This project currently includes three standalone scenes:

- `OLSVisualization` (`code/ols_regression/ols_scene.py`)
- `ConfidenceIntervals` (`code/confidence_intervals/ci_scene.py`)
- `CLTVisualization` (`code/central_limit_theorem/clt_scene.py`)

All visuals are generated in code (no external image/video assets required).

## Repository Structure

```text
Manim_BEI/
├─ code/
│  ├─ ols_regression/
│  │  └─ ols_scene.py
│  ├─ confidence_intervals/
│  │  └─ ci_scene.py
│  └─ central_limit_theorem/
│     └─ clt_scene.py
├─ videos/                           # curated final mp4 exports
└─ README.md
```

## Setup

Install Manim CE in your environment:

```bash
pip install manim
```

If rendering fails due to system dependencies, install the Manim prerequisites for your OS (FFmpeg and LaTeX packages as needed by your scene content).

## Rendering Commands

Use medium quality while iterating (`-pqm`) and high quality for final output (`-pqh`).

### Ordinary Least Squares

- **Scene class:** `OLSVisualization`
- **Source:** `code/ols_regression/ols_scene.py`
- **Reference:** [Seeing Theory - Regression Analysis](https://seeing-theory.brown.edu/regression-analysis/index.html)

```bash
manim -pqm code/ols_regression/ols_scene.py OLSVisualization
manim -pqh code/ols_regression/ols_scene.py OLSVisualization
```

### Confidence Intervals

- **Scene class:** `ConfidenceIntervals`
- **Source:** `code/confidence_intervals/ci_scene.py`
- **Reference:** [Seeing Theory - Confidence Intervals](https://seeing-theory.brown.edu/frequentist-inference/index.html#section2)

```bash
manim -pqm code/confidence_intervals/ci_scene.py ConfidenceIntervals
manim -pqh code/confidence_intervals/ci_scene.py ConfidenceIntervals
```

### Central Limit Theorem

- **Scene class:** `CLTVisualization`
- **Source:** `code/central_limit_theorem/clt_scene.py`
- **Reference:** [Seeing Theory - Probability Distributions (CLT)](https://seeing-theory.brown.edu/probability-distributions/index.html#section3)

```bash
manim -pqm code/central_limit_theorem/clt_scene.py CLTVisualization
manim -pqh code/central_limit_theorem/clt_scene.py CLTVisualization
```

## Output Locations

This repo keeps curated exports in `videos/`:

- `videos/ols_regression.mp4`
- `videos/confidence_intervals.mp4`
- `videos/CLTVisualization.mp4`

## Notes

- Scene scripts use fixed random seeds for reproducible animations.
