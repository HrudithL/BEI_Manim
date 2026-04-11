# Concept Visualization in Manim

## Animation 1 — Ordinary Least Squares

**Concept:** Ordinary Least Squares (OLS) regression
**Seeing Theory reference:** [Regression Analysis — Seeing Theory (Brown University)](https://seeing-theory.brown.edu/regression-analysis/index.html)

### Storyline

```
Data Cloud → Candidate Line → Residuals → SSE (geometric squares)
           → Slope Sweep (live SSE) → Optimal OLS Line + Formulas
```

The animation walks through:
1. Scatter plot of observed data points
2. A candidate (non-optimal) regression line
3. Residuals drawn as vertical segments — eᵢ = yᵢ − ŷᵢ
4. SSE visualized geometrically: each squared residual = area of a square
5. Slope sweep rotating the line around the centroid (x̄, ȳ) while SSE updates live
6. Convergence to the OLS optimal line with closed-form estimators

### Requirements met

| Requirement | How |
|---|---|
| > 45 seconds | ~75 seconds total |
| One connected storyline | Data → Line → Residuals → SSE → Optimal |
| Mathematical equations | SSE formula, β₁ and β₀ estimators, argmin expression |
| Visual representation | Scatter plot, regression line, residual segments, geometric squares |
| Animated change over time | Slope sweep with live SSE update |
| Understandable without narration | On-screen captions at each step |

### How to run

Install Manim Community Edition if needed:

```bash
pip install manim
```

Render (medium quality, auto-preview):

```bash
manim -pqm code/ols_regression/ols_scene.py OLSVisualization
```

For the final high-quality render:

```bash
manim -pqh code/ols_regression/ols_scene.py OLSVisualization
```

The rendered file will appear in `media/videos/ols_scene/`. Copy it to `videos/ols_regression.mp4`.

### External assets

None — all visuals are generated programmatically with Manim CE.
