from manim import *
import numpy as np

# ── Parameters ──────────────────────────────────────────────────────────────────
#
#  Population: Normal(μ=0, σ=1).
#  Each trial: draw n=30 observations, compute x-bar, build the 95% CI:
#      x-bar ± z* · σ/√n   where z* = 1.96
#  After N_CI trials the panel of horizontal CI segments illustrates
#  that ~95% of them capture the true μ.

RNG    = np.random.default_rng(7)
MU     = 0.0
SIGMA  = 1.0
N      = 30
Z95    = 1.96
MARGIN = Z95 * SIGMA / np.sqrt(N)   # ≈ 0.358

N_CI   = 24   # total CIs shown in Act 5
N_SHOW = 5    # drawn slowly one-by-one; remainder are fast


def normal_pdf(x, mu=0.0, sigma=1.0):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


# Pre-generate all samples so the animation is deterministic
ALL_SAMPLES = RNG.normal(MU, SIGMA, size=(N_CI, N))
XBARS       = ALL_SAMPLES.mean(axis=1)
CI_LOWER    = XBARS - MARGIN
CI_UPPER    = XBARS + MARGIN
CONTAINS    = (CI_LOWER <= MU) & (MU <= CI_UPPER)


# ── Layout constants ────────────────────────────────────────────────────────────
DIST_SHIFT = LEFT * 3.8 + DOWN * 0.3    # population distribution panel (left)
CI_SHIFT   = RIGHT * 2.5 + DOWN * 0.2   # stacked-CI panel (right)
CAPTION_Y  = -3.55


# ── Scene ───────────────────────────────────────────────────────────────────────
class ConfidenceIntervals(Scene):
    """
    Storyline:
      Title
      → Population N(0,1) with true mean μ
      → CI formula: x-bar ± z*(σ/√n), z*=1.96 for 95%
      → One sample drawn → x-bar computed → CI bracket on x-axis
      → 30 CIs stacked: green captures μ, red misses — ~95% are green
      → End card with correct interpretation
    """

    def construct(self):
        self._act1_title()
        self._act2_population()
        self._act3_formula()
        self._act4_single_ci()
        self._act5_many_cis()
        self._act6_end_card()

    # ── Helpers ─────────────────────────────────────────────────────────────────

    def _caption(self, text, font_size=22, color=GRAY_B):
        return Text(text, font_size=font_size, color=color).move_to([0, CAPTION_Y, 0])

    # ── Act 1: Title ─────────────────────────────────────────────────────────────
    def _act1_title(self):
        title = Text("Confidence Intervals", font_size=54, weight=BOLD)
        sub   = Text(
            "What they are — and what they're not",
            font_size=26, color=GRAY_B,
        ).next_to(title, DOWN, buff=0.5)

        self.play(Write(title, run_time=1.5))
        self.play(FadeIn(sub, shift=UP * 0.3))
        self.wait(2.5)
        self.play(FadeOut(VGroup(title, sub)))

    # ── Act 2: Population distribution ──────────────────────────────────────────
    def _act2_population(self):
        ax = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 0.55, 0.1],
            x_length=5.5,
            y_length=3.6,
            axis_config={"include_tip": True, "color": GRAY_B},
            x_axis_config={"numbers_to_include": [-3, -2, -1, 0, 1, 2, 3]},
            y_axis_config={"numbers_to_include": [0.1, 0.2, 0.3, 0.4]},
        ).shift(DIST_SHIFT)

        curve = ax.plot(lambda x: normal_pdf(x), x_range=[-3.8, 3.8],
                        color=BLUE_C, stroke_width=3)
        area  = ax.get_area(curve, x_range=[-3.8, 3.8],
                             color=[BLUE_D, BLUE_C], opacity=0.2)

        pop_lbl = MathTex(r"X \sim \mathcal{N}(\mu,\,\sigma^2)",
                          font_size=24, color=BLUE_C).next_to(ax, UP, buff=0.22)
        mu_lbl  = MathTex(r"\mu = 0",    font_size=24, color=YELLOW
                          ).move_to(ax.c2p(2.5, 0.46))
        sig_lbl = MathTex(r"\sigma = 1", font_size=24, color=GRAY_B
                          ).next_to(mu_lbl, DOWN, buff=0.22)
        mu_line = DashedLine(
            ax.c2p(0, 0), ax.c2p(0, 0.44),
            color=YELLOW, stroke_width=2.5, dash_length=0.12,
        )

        cap1 = self._caption("Population: Normal with mean μ = 0, std σ = 1")
        self.play(Create(ax), run_time=1.0)
        self.play(Create(curve), FadeIn(area), Write(pop_lbl), run_time=1.2)
        self.play(Create(mu_line), Write(mu_lbl), Write(sig_lbl))
        self.play(Write(cap1))
        self.wait(2.0)

        cap2 = self._caption("In practice μ is unknown — we estimate it from a sample of size n")
        self.play(Transform(cap1, cap2))
        self.wait(2.5)
        self.play(FadeOut(cap1))

        # Store for later acts
        self._dist_ax    = ax
        self._dist_curve = curve
        self._dist_area  = area
        self._pop_lbl    = pop_lbl
        self._mu_line    = mu_line
        self._mu_lbl_d   = mu_lbl
        self._sig_lbl    = sig_lbl

    # ── Act 3: CI formula ────────────────────────────────────────────────────────
    def _act3_formula(self):
        ci_title = Text("95% Confidence Interval", font_size=26, color=GREEN_B
                        ).shift(RIGHT * 2.8 + UP * 2.6)
        formula  = MathTex(
            r"\bar{x} \;\pm\; z^* \cdot \frac{\sigma}{\sqrt{n}}",
            font_size=42,
        ).next_to(ci_title, DOWN, buff=0.45)
        where_z  = MathTex(r"z^* = 1.96 \text{ (95\%)}",
                           font_size=26, color=YELLOW).next_to(formula, DOWN, buff=0.45)
        where_n  = MathTex(r"n = 30",
                           font_size=26, color=BLUE_B).next_to(where_z, DOWN, buff=0.25)
        where_m  = MathTex(
            r"\text{margin} = 1.96 \cdot \frac{1}{\sqrt{30}} \approx 0.36",
            font_size=22, color=GREEN_B,
        ).next_to(where_n, DOWN, buff=0.28)

        cap = MathTex(
            r"\text{The CI is centered at } \bar{x} \text{ with } \pm \text{ margin of error}",
            font_size=34,
            color=GRAY_B,
        ).move_to([0, CAPTION_Y, 0])
        self.play(Write(ci_title))
        self.play(Write(formula))
        self.play(FadeIn(where_z, shift=LEFT * 0.12))
        self.play(FadeIn(where_n, shift=LEFT * 0.12))
        self.play(FadeIn(where_m, shift=LEFT * 0.12))
        self.play(Write(cap))
        self.wait(2.5)
        self.play(FadeOut(cap))

        self._formula_group = VGroup(ci_title, formula, where_z, where_n, where_m)

    # ── Act 4: Single CI construction ────────────────────────────────────────────
    def _act4_single_ci(self):
        ax   = self._dist_ax
        xbar = float(XBARS[0])
        lo   = float(CI_LOWER[0])
        hi   = float(CI_UPPER[0])

        # Show 15 sample dots on the x-axis
        sample_pts = np.clip(ALL_SAMPLES[0], -3.8, 3.8)
        dots = VGroup(*[
            Dot(ax.c2p(v, 0), color=YELLOW, radius=0.065)
            for v in sample_pts[:15]
        ])
        xbar_dot = Dot(ax.c2p(xbar, 0), color=GREEN_B, radius=0.10)
        xbar_lbl = MathTex(
            rf"\bar{{x}} = {xbar:.2f}", font_size=25, color=GREEN_B,
        ).next_to(self._sig_lbl, DOWN, buff=0.28).align_to(self._mu_lbl_d, LEFT)

        # Remove y-axis numbers before sampling begins to reduce clutter.
        if hasattr(ax.y_axis, "numbers") and len(ax.y_axis.numbers) > 0:
            self.play(FadeOut(ax.y_axis.numbers), run_time=0.4)

        cap = MathTex(
            rf"\text{{Draw }} n = 30 \;\to\; \text{{sample mean }} \bar{{x}} = {xbar:.2f}",
            font_size=34,
            color=GRAY_B,
        ).move_to([0, CAPTION_Y, 0])
        self.play(Write(cap))
        self.play(LaggedStart(*[GrowFromCenter(d) for d in dots],
                              lag_ratio=0.05, run_time=1.0))
        self.play(GrowFromCenter(xbar_dot), Write(xbar_lbl))
        self.wait(0.8)
        self.play(FadeOut(cap))

        # CI bracket at a small height above x-axis
        ci_y  = 0.09
        ci_ln = Line(ax.c2p(lo, ci_y), ax.c2p(hi, ci_y), color=GREEN_B, stroke_width=5)
        tk_l  = Line(ax.c2p(lo, ci_y - 0.03), ax.c2p(lo, ci_y + 0.03),
                     color=GREEN_B, stroke_width=3)
        tk_r  = Line(ax.c2p(hi, ci_y - 0.03), ax.c2p(hi, ci_y + 0.03),
                     color=GREEN_B, stroke_width=3)
        ci_grp = VGroup(ci_ln, tk_l, tk_r)

        cap2 = self._caption(f"95% CI:  [{lo:.2f},  {hi:.2f}]  — does it capture μ = 0?")
        self.play(Create(ci_grp, run_time=0.8))
        self.play(Write(cap2))
        self.wait(1.0)

        # Flash the true mean line to draw attention
        self.play(self._mu_line.animate.set_color(RED).set_stroke(width=5.0), run_time=0.3)
        self.play(self._mu_line.animate.set_color(YELLOW).set_stroke(width=2.5), run_time=0.3)

        col = GREEN_B if CONTAINS[0] else RED
        msg = "Yes — μ is captured!" if CONTAINS[0] else "No — μ is missed!"
        result_lbl = Text(msg, font_size=25, color=col).next_to(ax, DOWN, buff=0.35)
        self.play(FadeIn(result_lbl, shift=UP * 0.1))
        self.wait(2.0)

        self.play(FadeOut(VGroup(dots, xbar_dot, xbar_lbl, ci_grp, result_lbl, cap2)))
        self.play(FadeOut(self._formula_group))

    # ── Act 5: Many CIs ──────────────────────────────────────────────────────────
    def _act5_many_cis(self):
        # Right-hand panel: x = parameter values, y = CI index
        ci_ax = Axes(
            x_range=[-1.1, 1.1, 0.5],
            y_range=[0, N_CI + 1, 5],
            x_length=4.5,
            y_length=5.2,
            axis_config={"include_tip": False, "color": GRAY_B},
            x_axis_config={"numbers_to_include": [-1.0, -0.5, 0.0, 0.5, 1.0]},
            y_axis_config={"numbers_to_include": []},
        ).shift(CI_SHIFT)
        # Align second panel x-axis baseline with the population panel x-axis.
        x_axis_y_diff = self._dist_ax.c2p(0, 0)[1] - ci_ax.c2p(0, 0)[1]
        ci_ax.shift(UP * x_axis_y_diff)

        self.play(Create(ci_ax, run_time=0.8))

        # Dashed vertical line at true mean
        mu_vert = DashedLine(
            ci_ax.c2p(MU, 0), ci_ax.c2p(MU, N_CI + 0.6),
            color=YELLOW, stroke_width=2.5, dash_length=0.12,
        )
        mu_vert_lbl = MathTex(r"\mu = 0", font_size=22, color=YELLOW
                               ).next_to(ci_ax.c2p(MU, N_CI + 0.6), UP, buff=0.1)
        self.play(Create(mu_vert), Write(mu_vert_lbl))

        cap = self._caption("Each row = one 95% CI built from a fresh n = 30 sample")
        self.play(Write(cap))
        self.wait(0.5)

        # Accumulate CI line segments
        green_count = 0
        red_count   = 0
        all_ci_mobs = VGroup()

        for i in range(N_CI):
            lo  = float(CI_LOWER[i])
            hi  = float(CI_UPPER[i])
            y   = float(i + 1)
            col = GREEN if CONTAINS[i] else RED

            seg = Line(ci_ax.c2p(lo, y), ci_ax.c2p(hi, y), color=col, stroke_width=3)
            dot = Dot(ci_ax.c2p(float(XBARS[i]), y), color=col, radius=0.048)
            grp = VGroup(seg, dot)
            all_ci_mobs.add(grp)

            rt = 0.35 if i < N_SHOW else 0.07
            self.play(Create(grp), run_time=rt)

            if CONTAINS[i]:
                green_count += 1
            else:
                red_count += 1

        self.wait(1.5)
        self.play(FadeOut(cap))

        # Show capture rate
        pct = 100 * green_count / N_CI
        summary = Text(
            f"{green_count}/{N_CI} intervals capture μ  ({pct:.0f}%)",
            font_size=22, color=GRAY_B,
        ).move_to([0, CAPTION_Y, 0])
        self.play(Write(summary))
        self.wait(1.5)

        cap2 = self._caption("~95% capture μ — this is what '95% confidence' means")
        self.play(Transform(summary, cap2))
        self.wait(3.0)
        self.play(FadeOut(summary))

        self._ci_ax       = ci_ax
        self._mu_vert     = mu_vert
        self._mu_vert_lbl = mu_vert_lbl
        self._all_ci_mobs = all_ci_mobs

    # ── Act 6: End card ──────────────────────────────────────────────────────────
    def _act6_end_card(self):
        self.play(FadeOut(VGroup(
            self._dist_ax,  self._dist_curve, self._dist_area,
            self._pop_lbl,  self._mu_line,    self._mu_lbl_d, self._sig_lbl,
            self._ci_ax,    self._mu_vert,    self._mu_vert_lbl,
            self._all_ci_mobs,
        )))

        title = Text("Confidence Intervals", font_size=46, weight=BOLD)
        formula = MathTex(
            r"\bar{x} \;\pm\; z^* \cdot \frac{\sigma}{\sqrt{n}}",
            font_size=40, color=GREEN_B,
        ).next_to(title, DOWN, buff=0.6)
        interp = VGroup(
            Text("95% CI: in repeated sampling, about 95% of intervals contain μ",
                 font_size=24, color=GRAY_B),
            Text("It does not mean: this one interval has a 95% chance for μ",
                 font_size=24, color=GRAY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).next_to(formula, DOWN, buff=0.55)
        interp.shift(UP * 0.2)

        self.play(Write(title))
        self.play(Write(formula))
        self.play(FadeIn(interp, shift=UP * 0.2))
        self.wait(5.0)
