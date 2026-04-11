from manim import *
import numpy as np

# ── Data ─────────────────────────────────────────────────────────────────────
DATA_X = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
DATA_Y = np.array([2.1, 3.8, 3.5, 5.2, 5.8, 7.1, 6.9, 8.5])


def ols_params(x, y):
    xm, ym = x.mean(), y.mean()
    b1 = np.dot(x - xm, y - ym) / np.dot(x - xm, x - xm)
    return ym - b1 * xm, b1


def compute_sse(x, y, b0, b1):
    return float(np.sum((y - (b0 + b1 * x)) ** 2))


B0_OPT, B1_OPT = ols_params(DATA_X, DATA_Y)
SSE_OPT = compute_sse(DATA_X, DATA_Y, B0_OPT, B1_OPT)
X_MEAN = float(DATA_X.mean())
Y_MEAN = float(DATA_Y.mean())

# Initial candidate line — visibly non-optimal
B0_CAND, B1_CAND = 0.5, 0.75

# ── Layout constants ──────────────────────────────────────────────────────────
#
#  Scene: 14.22 × 8  →  x ∈ [−7.11, 7.11],  y ∈ [−4, 4]
#
#  Axes:  x_length=5.8, y_length=4.4, center at (−2.6, 0.4)
#         right edge  ≈ −2.6 + 2.9 = 0.3
#         bottom edge ≈  0.4 − 2.2 = −1.8   (caption zone starts well below)
#
#  Right panel:  x starts at 0.7  →  6.4 units wide  (formulas safe)
#  Caption strip: y ≈ −3.4        →  1.6 units below axes bottom
#
AXES_SHIFT   = LEFT * 2.6 + UP * 0.4
AXES_X_LEN   = 5.8
AXES_Y_LEN   = 4.4

# Top-left anchor of the right-panel formula column
PANEL_X      =  0.7          # screen x of left edge of formulas
PANEL_TOP_Y  =  3.3          # screen y of first formula

# Caption y (to_edge DOWN already gives ≈ −3.7, but we pin it to be safe)
CAPTION_Y    = -3.35


# ── Scene ─────────────────────────────────────────────────────────────────────
class OLSVisualization(Scene):
    """
    Storyline: Data Cloud → Candidate Line → Residuals → SSE Squares
               → Slope Sweep (SSE live) → Optimal OLS Line + Formulas
    """

    def construct(self):
        self._act1_title()
        self._act2_scatter()
        self._act3_candidate_line()
        self._act4_residuals()
        self._act5_sse_squares()
        self._act6_slope_sweep()
        self._act7_optimal_and_formulas()

    # ── Layout helpers ────────────────────────────────────────────────────────

    def _build_axes(self):
        ax = Axes(
            x_range=[0, 9.5, 1],
            y_range=[0, 10.5, 2],
            x_length=AXES_X_LEN,
            y_length=AXES_Y_LEN,
            axis_config={"include_tip": True, "color": GRAY_B},
            x_axis_config={"numbers_to_include": list(range(1, 9))},
            y_axis_config={"numbers_to_include": list(range(0, 11, 2))},
        ).shift(AXES_SHIFT)
        xl = ax.get_x_axis_label("x", edge=RIGHT, direction=RIGHT, buff=0.15)
        yl = ax.get_y_axis_label("y", edge=UP,    direction=UP,    buff=0.15)
        return ax, VGroup(xl, yl)

    def _panel_obj(self, mobject, row=0, prev=None, gap=0.35):
        """Position a formula in the right panel.

        row=0  → top anchor; otherwise placed below `prev` with `gap`.
        """
        if prev is None:
            mobject.move_to([PANEL_X, PANEL_TOP_Y, 0], aligned_edge=UL)
        else:
            mobject.next_to(prev, DOWN, buff=gap).align_to(prev, LEFT)
        return mobject

    def _caption(self, text, font_size=23, color=GRAY_B):
        """Single-line caption pinned to the caption strip."""
        cap = Text(text, font_size=font_size, color=color)
        cap.move_to([0, CAPTION_Y, 0])
        return cap

    def _line_on_axes(self, ax, b0, b1, color=RED_B, width=3):
        return ax.plot(
            lambda x: b0 + b1 * x,
            x_range=[0.8, 8.5],
            color=color,
            stroke_width=width,
        )

    def _residual_segments(self, ax, b0, b1, color=GREEN_B):
        return VGroup(*[
            Line(ax.c2p(x, y), ax.c2p(x, b0 + b1 * x),
                 color=color, stroke_width=3)
            for x, y in zip(DATA_X, DATA_Y)
        ])

    # ── Act 1 : Title ─────────────────────────────────────────────────────────

    def _act1_title(self):
        title = Text("Ordinary Least Squares", font_size=52, weight=BOLD)
        subtitle = Text(
            "Fitting a line to data by minimizing squared error",
            font_size=26, color=GRAY_B,
        ).next_to(title, DOWN, buff=0.5)

        self.play(Write(title, run_time=1.5))
        self.play(FadeIn(subtitle, shift=UP * 0.3))
        self.wait(2.0)
        self.play(FadeOut(VGroup(title, subtitle)))

    # ── Act 2 : Scatter plot ──────────────────────────────────────────────────

    def _act2_scatter(self):
        ax, axis_labels = self._build_axes()
        self.play(Create(ax), Write(axis_labels), run_time=1.5)

        dots = VGroup(*[
            Dot(ax.c2p(x, y), color=YELLOW, radius=0.09)
            for x, y in zip(DATA_X, DATA_Y)
        ])
        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in dots],
                        lag_ratio=0.18, run_time=2.0)
        )

        lbl = Text("Observed data points", font_size=26, color=YELLOW)
        lbl.move_to([PANEL_X + 2.5, PANEL_TOP_Y, 0])   # right panel, top row
        self.play(FadeIn(lbl))
        self.wait(1.5)
        self.play(FadeOut(lbl))

        self.ax = ax
        self.axis_labels = axis_labels
        self.dots = dots

    # ── Act 3 : Candidate line ────────────────────────────────────────────────

    def _act3_candidate_line(self):
        ax = self.ax

        line = self._line_on_axes(ax, B0_CAND, B1_CAND)

        eq = MathTex(
            r"\hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x",
            font_size=34, color=RED_B,
        )
        self._panel_obj(eq)                              # row 0, top of panel

        caption = self._caption(
            "Any line is a candidate — but which fits best?"
        )

        self.play(Create(line), Write(eq))
        self.play(Write(caption))
        self.wait(2.0)
        self.play(FadeOut(caption))

        self.cand_line = line
        self.eq_label  = eq

    # ── Act 4 : Residuals ─────────────────────────────────────────────────────

    def _act4_residuals(self):
        ax = self.ax

        res_segs = self._residual_segments(ax, B0_CAND, B1_CAND)

        # annotate one residual — place in open space below the line, above x-axis
        # candidate line at x=5: y = 0.5 + 0.75*5 = 4.25, so y≈1.8 is well below
        res_lbl = MathTex(
            r"e_i = y_i - \hat{y}_i", font_size=28, color=GREEN_B
        ).move_to(ax.c2p(5.5, 1.6))

        cap1 = self._caption(
            "Residuals: vertical distance from each point to the line"
        )
        cap2 = self._caption(
            "Each residual measures how wrong the prediction is"
        )

        self.play(
            LaggedStart(*[Create(s) for s in res_segs],
                        lag_ratio=0.12, run_time=1.8)
        )
        self.play(Write(res_lbl), Write(cap1))
        self.wait(1.5)
        self.play(Transform(cap1, cap2))
        self.wait(1.5)
        self.play(FadeOut(cap1), FadeOut(res_lbl))

        self.res_segs = res_segs

    # ── Act 5 : SSE — geometric squares ───────────────────────────────────────

    def _act5_sse_squares(self):
        ax = self.ax

        sse_formula = MathTex(
            r"\text{SSE} = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2",
            font_size=30,
        )
        self._panel_obj(sse_formula, prev=self.eq_label, gap=0.45)

        self.play(Write(sse_formula))
        self.wait(0.4)

        # geometric squares: side length = |residual| in data units
        squares = VGroup()
        for x, y in zip(DATA_X, DATA_Y):
            yhat  = B0_CAND + B1_CAND * x
            side  = abs(y - yhat)
            y_lo, y_hi = min(y, yhat), max(y, yhat)
            sq = Polygon(
                ax.c2p(x,        y_lo),
                ax.c2p(x + side, y_lo),
                ax.c2p(x + side, y_hi),
                ax.c2p(x,        y_hi),
                fill_color=GREEN, fill_opacity=0.22,
                stroke_color=GREEN_B, stroke_width=1.5,
            )
            squares.add(sq)

        sse_val = compute_sse(DATA_X, DATA_Y, B0_CAND, B1_CAND)
        sse_num = MathTex(
            rf"\text{{SSE}} = {sse_val:.2f}",
            font_size=28, color=GREEN_B,
        )
        self._panel_obj(sse_num, prev=sse_formula, gap=0.3)

        caption = self._caption("Each squared residual = area of a square")

        self.play(
            LaggedStart(*[DrawBorderThenFill(s) for s in squares],
                        lag_ratio=0.1, run_time=2.0)
        )
        self.play(Write(sse_num), Write(caption))
        self.wait(2.5)
        self.play(FadeOut(caption), FadeOut(squares), FadeOut(self.res_segs))

        self.sse_formula  = sse_formula
        self.sse_num_obj  = sse_num
        # Store the fixed screen position so always_redraw can reuse it
        self._sse_num_pos = sse_num.get_center()

    # ── Act 6 : Slope sweep — SSE live ────────────────────────────────────────

    def _act6_slope_sweep(self):
        ax      = self.ax
        slope_t = ValueTracker(B1_CAND)

        def intercept(b1):
            return Y_MEAN - b1 * X_MEAN          # pivot through centroid

        sweep_line = always_redraw(lambda: self._line_on_axes(
            ax, intercept(slope_t.get_value()), slope_t.get_value()
        ))

        sse_num_pos = self._sse_num_pos           # captured fixed position
        sse_live = always_redraw(lambda: MathTex(
            rf"\text{{SSE}} = "
            rf"{compute_sse(DATA_X, DATA_Y, intercept(slope_t.get_value()), slope_t.get_value()):.2f}",
            font_size=28, color=GREEN_B,
        ).move_to(sse_num_pos))

        self.remove(self.cand_line, self.sse_num_obj)
        self.add(sweep_line, sse_live)

        centroid_dot = Dot(ax.c2p(X_MEAN, Y_MEAN), color=WHITE, radius=0.09)
        centroid_lbl = MathTex(r"(\bar{x},\bar{y})", font_size=20, color=WHITE)
        centroid_lbl.next_to(centroid_dot, DR, buff=0.12)

        caption = Tex(
            r"Sweep the slope --- the line rotates around $(\bar{x},\,\bar{y})$",
            font_size=24, color=GRAY_B,
        ).move_to([0, CAPTION_Y, 0])

        self.play(FadeIn(centroid_dot), Write(centroid_lbl), Write(caption))

        self.play(slope_t.animate.set_value(0.35), run_time=2.0, rate_func=smooth)
        self.wait(0.3)
        self.play(slope_t.animate.set_value(1.55), run_time=3.5, rate_func=smooth)
        self.wait(0.3)
        self.play(slope_t.animate.set_value(B1_OPT), run_time=2.0, rate_func=smooth)
        self.wait(0.8)

        self.play(FadeOut(caption), FadeOut(centroid_dot), FadeOut(centroid_lbl))

        self.sweep_line = sweep_line
        self.sse_live   = sse_live

    # ── Act 7 : Optimal line + closed-form formulas ───────────────────────────

    def _act7_optimal_and_formulas(self):
        ax = self.ax

        opt_line = self._line_on_axes(ax, B0_OPT, B1_OPT, color=BLUE, width=4)

        opt_sse_lbl = MathTex(
            rf"\text{{SSE}}_{{\min}} = {SSE_OPT:.2f}",
            font_size=28, color=BLUE,
        ).move_to(self._sse_num_pos)

        self.remove(self.sweep_line, self.sse_live)
        self.play(Create(opt_line), Write(opt_sse_lbl))
        self.wait(0.5)

        # OLS closed-form estimators
        beta1_eq = MathTex(
            r"\hat{\beta}_1 = "
            r"\frac{\sum_i(x_i-\bar{x})(y_i-\bar{y})}{\sum_i(x_i-\bar{x})^2}",
            font_size=26,
        )
        self._panel_obj(beta1_eq, prev=opt_sse_lbl, gap=0.45)

        beta0_eq = MathTex(
            r"\hat{\beta}_0 = \bar{y} - \hat{\beta}_1\bar{x}",
            font_size=26,
        )
        self._panel_obj(beta0_eq, prev=beta1_eq, gap=0.35)

        self.play(Write(beta1_eq))
        self.play(Write(beta0_eq))
        self.wait(0.5)

        # residuals on the optimal line in blue
        final_res = self._residual_segments(ax, B0_OPT, B1_OPT, color=BLUE_B)
        self.play(
            LaggedStart(*[Create(r) for r in final_res],
                        lag_ratio=0.1, run_time=1.5)
        )

        caption = self._caption(
            "OLS: the unique line that minimizes SSE",
            color=BLUE_B,
        )
        self.play(Write(caption))
        self.wait(3.5)

        # ── end card ──────────────────────────────────────────────────────────
        self.play(FadeOut(VGroup(
            ax, self.axis_labels, self.dots, opt_line,
            self.eq_label, self.sse_formula, opt_sse_lbl,
            beta1_eq, beta0_eq, final_res, caption,
        )))

        end_title = Text("Ordinary Least Squares", font_size=48, weight=BOLD)
        end_sub   = Text(
            "minimizes the sum of squared residuals",
            font_size=26, color=GRAY_B,
        ).next_to(end_title, DOWN, buff=0.45)
        end_eq = MathTex(
            r"\hat{\boldsymbol{\beta}} = "
            r"\arg\min_{\beta}"
            r"\sum_{i=1}^{n}(y_i - \beta_0 - \beta_1 x_i)^2",
            font_size=30,
        ).next_to(end_sub, DOWN, buff=0.5)

        self.play(Write(end_title))
        self.play(FadeIn(end_sub, shift=UP * 0.2))
        self.play(Write(end_eq))
        self.wait(3.0)
