from manim import *
import numpy as np

# ── RNG & Population ──────────────────────────────────────────────────────────
#
#  Population: Exponential(mean=2)  →  right-skewed, clearly non-Normal.
#  We draw SAMPLE_SIZE observations per trial and record the sample mean.
#  After N_SAMPLES trials the histogram of those means should look Normal
#  by the Central Limit Theorem.

RNG = np.random.default_rng(42)

POP_MEAN    = 2.0
POP_STD     = 2.0          # Exp: std = mean
SAMPLE_SIZE = 30
N_SAMPLES   = 500

ALL_SAMPLES  = RNG.exponential(scale=POP_MEAN, size=(N_SAMPLES, SAMPLE_SIZE))
SAMPLE_MEANS = ALL_SAMPLES.mean(axis=1)

UNIFORM_SAMPLES = RNG.uniform(0.0, 4.0, size=(N_SAMPLES, SAMPLE_SIZE))
UNIFORM_MEANS = UNIFORM_SAMPLES.mean(axis=1)

mix_choice = RNG.random((N_SAMPLES, SAMPLE_SIZE)) < 0.5
mix_left = RNG.normal(loc=1.0, scale=0.35, size=(N_SAMPLES, SAMPLE_SIZE))
mix_right = RNG.normal(loc=3.0, scale=0.35, size=(N_SAMPLES, SAMPLE_SIZE))
BIMODAL_SAMPLES = np.where(mix_choice, mix_left, mix_right)
BIMODAL_SAMPLES = np.clip(BIMODAL_SAMPLES, 0.02, 8.8)
BIMODAL_MEANS = BIMODAL_SAMPLES.mean(axis=1)

CLT_MEAN = POP_MEAN
CLT_STD  = POP_STD / np.sqrt(SAMPLE_SIZE)   # ≈ 0.365


def normal_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


# Histogram bins for sample means (density mode)
HIST_BINS = np.linspace(0.5, 3.5, 31)
HIST_BW = HIST_BINS[1] - HIST_BINS[0]

# ── Layout constants ──────────────────────────────────────────────────────────
#
#  Left panel  (population):   center ≈ (−3.3, −0.3)
#  Right panel (sample means): center ≈ ( 3.0, −0.3)
#  Caption strip:              y ≈ −3.55

POP_SHIFT  = LEFT  * 3.3 + DOWN * 0.3
SAMP_SHIFT = RIGHT * 3.0 + DOWN * 0.3
CAPTION_Y  = -3.55

N_EXPLICIT = 5   # samples shown one-by-one before the fast fill


# ── Scene ─────────────────────────────────────────────────────────────────────
class CLTVisualization(Scene):
    """
    Storyline:
      Title
      → Skewed population (Exponential) with μ, σ labelled
      → Draw n=30 samples one at a time, watch mean fly to right histogram
      → Fast-fill 500 sample means into the histogram
      → Bell curve (Normal) overlaid on the histogram
      → CLT formula and standardised form end card
    """

    def construct(self):
        self._act1_title()
        self._act2_population()
        self._act3_first_samples()
        self._act4_fill_histogram()
        self._act5_normal_overlay()
        self._act6_formula()

    # ── Axis / bar helpers ────────────────────────────────────────────────────

    def _build_pop_axes(self):
        ax = Axes(
            x_range=[0, 9, 2],
            y_range=[0, 0.65, 0.2],
            x_length=5.2,
            y_length=3.6,
            axis_config={"include_tip": False, "color": GRAY_B},
            x_axis_config={"numbers_to_include": [0, 2, 4, 6, 8]},
            y_axis_config={"numbers_to_include": [0.2, 0.4, 0.6]},
        ).shift(POP_SHIFT)
        return ax

    def _build_samp_axes(self, y_max=1.6):
        if y_max <= 0.3:
            y_step = 0.05
        elif y_max <= 0.6:
            y_step = 0.1
        elif y_max <= 1.0:
            y_step = 0.2
        else:
            y_step = 0.3
        y_numbers = [round(y_step * i, 2) for i in range(1, 4)]
        ax = Axes(
            x_range=[0.5, 3.5, 0.5],
            y_range=[0, y_max, y_step],
            x_length=5.2,
            y_length=3.6,
            axis_config={"include_tip": False, "color": GRAY_B},
            x_axis_config={"numbers_to_include": [1.0, 1.5, 2.0, 2.5, 3.0]},
            y_axis_config={"numbers_to_include": y_numbers},
        ).shift(SAMP_SHIFT)
        return ax

    def _caption(self, text, font_size=23, color=GRAY_B):
        cap = Text(text, font_size=font_size, color=color)
        cap.move_to([0, CAPTION_Y, 0])
        return cap

    def _axis_tips(self, ax):
        x_end = ax.x_axis.get_end()
        y_end = ax.y_axis.get_end()
        x_tip = Arrow(
            x_end,
            x_end + RIGHT * 1e-3,
            buff=0,
            stroke_width=0,
            color=GRAY_B,
            tip_length=0.16,
            max_tip_length_to_length_ratio=10_000,
        )
        y_tip = Arrow(
            y_end,
            y_end + UP * 1e-3,
            buff=0,
            stroke_width=0,
            color=GRAY_B,
            tip_length=0.16,
            max_tip_length_to_length_ratio=10_000,
        )
        return VGroup(x_tip, y_tip)

    def _dynamic_samp_y_max(self, n, means_source):
        n = max(1, int(n))
        shown = means_source[:n]
        counts, _ = np.histogram(shown, bins=HIST_BINS, density=False)
        heights = counts / max(1, len(shown))
        max_hist_height = float(np.max(heights)) if len(heights) else 1.0
        progress = min(1.0, n / N_SAMPLES)
        zoom_target = np.interp(progress, [0.0, 0.12, 1.0], [1.2, 0.8, 0.22])
        target = max(1.1 * max_hist_height, zoom_target)
        return float(np.clip(target, 0.18, 1.25))

    def _hist_bars(self, ax, means):
        """Build histogram bars as per-bin sample proportions on `ax`."""
        if len(means) == 0:
            return VGroup()
        counts, edges = np.histogram(means, bins=HIST_BINS, density=False)
        heights = counts / max(1, len(means))
        bw = edges[1] - edges[0]
        bars = VGroup()
        for height, left in zip(heights, edges[:-1]):
            if height < 1e-9:
                continue
            bar = Polygon(
                ax.c2p(left,      0),
                ax.c2p(left + bw, 0),
                ax.c2p(left + bw, height),
                ax.c2p(left,      height),
                fill_color=BLUE_D,
                fill_opacity=0.75,
                stroke_color=BLUE_B,
                stroke_width=1.2,
            )
            bars.add(bar)
        return bars

    # ── Act 1 : Title ─────────────────────────────────────────────────────────

    def _act1_title(self):
        title = Text("The Central Limit Theorem", font_size=52, weight=BOLD)
        subtitle = Text(
            "Why sample means always look Normal",
            font_size=27, color=GRAY_B,
        ).next_to(title, DOWN, buff=0.5)

        self.play(Write(title, run_time=1.5))
        self.play(FadeIn(subtitle, shift=UP * 0.3))
        self.wait(2.0)
        self.play(FadeOut(VGroup(title, subtitle)))

    # ── Act 2 : Skewed population ─────────────────────────────────────────────

    def _act2_population(self):
        ax = self._build_pop_axes()
        self.play(Create(ax), run_time=1.2)

        # Exponential PDF curve
        curve = ax.plot(
            lambda x: (1 / POP_MEAN) * np.exp(-x / POP_MEAN),
            x_range=[0.02, 8.8],
            color=ORANGE,
            stroke_width=3,
        )
        area = ax.get_area(
            curve, x_range=[0.02, 8.8],
            color=[ORANGE, RED_D], opacity=0.2,
        )

        pop_lbl = MathTex(
            r"\text{Population }X\sim\mathrm{Exp}(\text{mean}=2)",
            font_size=28,
            color=ORANGE,
        )
        pop_lbl.next_to(ax, UP, buff=0.18).shift(LEFT * 0.3)

        # Dashed mean line
        mean_line = DashedLine(
            ax.c2p(POP_MEAN, 0),
            ax.c2p(POP_MEAN, 0.55),
            color=YELLOW, stroke_width=2.5, dash_length=0.12,
        )
        mu_lbl = MathTex(r"\mu = 2", font_size=26, color=YELLOW).next_to(
            ax.c2p(POP_MEAN + 0.2, 0.42), RIGHT, buff=0.05
        )
        sig_lbl = MathTex(r"\sigma = 2", font_size=26, color=GRAY_B).next_to(
            mu_lbl, DOWN, buff=0.22
        )

        cap = self._caption("Start with a skewed population — far from Normal")

        self.play(Create(curve), FadeIn(area), Write(pop_lbl), run_time=1.5)
        self.play(Create(mean_line), Write(mu_lbl), Write(sig_lbl))
        self.play(Write(cap))
        self.wait(2.5)
        self.play(FadeOut(cap))

        self._pop_ax    = ax
        self._pop_curve = curve
        self._pop_area  = area
        self._pop_lbl   = pop_lbl
        self._mean_line = mean_line
        self._mu_lbl    = mu_lbl
        self._sig_lbl   = sig_lbl
        self._pop_axis_tips = VGroup()

    # ── Act 3 : First N_EXPLICIT samples shown one-by-one ────────────────────

    def _act3_first_samples(self):
        pop_ax  = self._pop_ax
        self._means_source = SAMPLE_MEANS
        n_shown = ValueTracker(0)

        samp_ax = self._build_samp_axes(
            self._dynamic_samp_y_max(n_shown.get_value(), self._means_source)
        )
        samp_ax.add_updater(
            lambda mob: mob.become(
                self._build_samp_axes(
                    self._dynamic_samp_y_max(n_shown.get_value(), self._means_source)
                )
            )
        )
        samp_tips = always_redraw(lambda: self._axis_tips(samp_ax))

        samp_lbl = always_redraw(lambda: MathTex(
            r"\text{Sampling Distribution of }\bar{X}",
            font_size=28,
            color=BLUE_B,
        ).next_to(samp_ax, UP, buff=0.18).shift(RIGHT * 0.3))

        n_eq = always_redraw(lambda: MathTex(
            r"n = 30\ \text{observations per sample}",
            font_size=24,
            color=BLUE_B,
        ).next_to(samp_lbl, DOWN, buff=0.1))

        self.play(Create(samp_ax), FadeIn(samp_tips), Write(samp_lbl), Write(n_eq), run_time=1.2)

        cap = self._caption("Sample n=30, compute the sample mean, record it")
        self.play(Write(cap))

        # Histogram tied to ValueTracker so it updates automatically
        hist = always_redraw(lambda: self._hist_bars(
            samp_ax, self._means_source[: int(n_shown.get_value())]
        ))
        self.add(hist)

        for i in range(N_EXPLICIT):
            sample   = ALL_SAMPLES[i]
            s_mean   = float(SAMPLE_MEANS[i])

            # Show 12 sample dots on the population x-axis
            display_vals = np.clip(sample[:12], 0.05, 8.8)
            dots = VGroup(*[
                Dot(pop_ax.c2p(v, 0), color=YELLOW, radius=0.065)
                for v in display_vals
            ])
            self.play(
                LaggedStart(*[GrowFromCenter(d) for d in dots],
                            lag_ratio=0.07, run_time=0.8)
            )

            # Mean value label on the population panel
            xbar_lbl = MathTex(
                rf"\bar{{X}} = {s_mean:.2f}", font_size=25, color=GREEN_B
            ).move_to(pop_ax.c2p(6.0, 0.50))
            self.play(Write(xbar_lbl, run_time=0.35))

            # Green dot flies from population baseline to sampling baseline
            fly_start = pop_ax.c2p(s_mean, 0.02)
            fly_end   = samp_ax.c2p(s_mean, 0.02)
            fly_dot   = Dot(fly_start, color=GREEN_B, radius=0.09)
            self.play(FadeIn(fly_dot, run_time=0.15))
            self.play(fly_dot.animate.move_to(fly_end), run_time=0.55)
            self.play(
                FadeOut(fly_dot,  run_time=0.2),
                FadeOut(dots,     run_time=0.2),
                FadeOut(xbar_lbl, run_time=0.2),
            )

            # Grow the histogram bar(s)
            self.play(n_shown.animate.set_value(i + 1), run_time=0.35)
            self.wait(0.15)

        self.play(FadeOut(cap))

        self._samp_ax  = samp_ax
        self._samp_tips = samp_tips
        self._samp_lbl = samp_lbl
        self._n_eq     = n_eq
        self._n_shown  = n_shown
        self._hist     = hist

    # ── Act 4 : Fast fill — all 500 means ────────────────────────────────────

    def _act4_fill_histogram(self):
        n_shown = self._n_shown

        count_lbl = always_redraw(lambda: Text(
            f"samples = {int(n_shown.get_value())}",
            font_size=20, color=GRAY_B,
        ).next_to(self._samp_ax, DOWN, buff=0.14))
        self.add(count_lbl)

        cap = self._caption("Repeat — collect 500 sample means...")
        self.play(Write(cap))

        self.play(
            n_shown.animate.set_value(N_SAMPLES),
            run_time=6.0,
            rate_func=smooth,
        )
        self.wait(1.0)
        self.play(FadeOut(cap))

        self._count_lbl = count_lbl

    # ── Act 5 : Normal bell curve overlaid ───────────────────────────────────

    def _act5_normal_overlay(self):
        samp_ax = self._samp_ax

        bell = samp_ax.plot(
            lambda x: normal_pdf(x, CLT_MEAN, CLT_STD) * HIST_BW,
            x_range=[0.7, 3.3],
            color=YELLOW,
            stroke_width=3.5,
        )

        bell_lbl = MathTex(
            r"\mathcal{N}\!\left(\mu,\,\frac{\sigma^2}{n}\right)",
            font_size=26, color=YELLOW,
        ).move_to(samp_ax.c2p(2.85, 0.22))

        cap1 = self._caption(
            "A Normal bell curve fits perfectly — whatever the population shape!"
        )
        cap2 = self._caption(
            "This is the Central Limit Theorem"
        )

        self.play(Create(bell, run_time=1.8), Write(bell_lbl))
        self.play(Write(cap1))
        self.wait(2.5)
        self.play(Transform(cap1, cap2))
        self.wait(2.0)
        self.play(FadeOut(cap1))

        pop_modes = [
            (
                r"\text{Population }X\sim\mathrm{Unif}(0,4)",
                PURPLE_B,
                lambda x: np.where((x >= 0) & (x <= 4), 0.25, 0.0),
                UNIFORM_MEANS,
            ),
            (
                r"\text{Population }X\sim\text{Bimodal}",
                TEAL_B,
                lambda x: 0.5 * normal_pdf(x, 1.0, 0.35) + 0.5 * normal_pdf(x, 3.0, 0.35),
                BIMODAL_MEANS,
            ),
        ]

        for label_tex, color, pdf_func, means in pop_modes:
            new_curve = self._pop_ax.plot(
                pdf_func,
                x_range=[0.02, 8.8],
                color=color,
                stroke_width=3,
            )
            new_area = self._pop_ax.get_area(
                new_curve, x_range=[0.02, 8.8],
                color=[color, color], opacity=0.2,
            )
            new_lbl = MathTex(label_tex, font_size=28, color=color)
            new_lbl.next_to(self._pop_ax, UP, buff=0.18).shift(LEFT * 0.3)

            cap_pop = self._caption("Change the population shape, then resample 500 means")
            self.play(
                Transform(self._pop_curve, new_curve),
                Transform(self._pop_area, new_area),
                Transform(self._pop_lbl, new_lbl),
                FadeIn(cap_pop),
                run_time=1.2,
            )
            self.wait(0.5)

            self._means_source = means
            self.play(self._n_shown.animate.set_value(0), run_time=0.5)
            self.play(
                self._n_shown.animate.set_value(N_SAMPLES),
                run_time=2.8,
                rate_func=smooth,
            )
            quick_bell = self._samp_ax.plot(
                lambda x: normal_pdf(x, CLT_MEAN, CLT_STD) * HIST_BW,
                x_range=[0.7, 3.3],
                color=YELLOW,
                stroke_width=3.5,
            )
            self.play(Transform(bell, quick_bell), run_time=0.25)
            self.wait(0.15)
            self.play(FadeOut(cap_pop), run_time=0.35)

        # Final right panel: only the Normal curve (no labels/counters/histogram text).
        clean_ax = Axes(
            x_range=[0.5, 3.5, 0.5],
            y_range=[0, 0.26, 0.05],
            x_length=5.2,
            y_length=3.6,
            axis_config={"include_tip": False, "color": GRAY_B},
            x_axis_config={"include_numbers": False},
            y_axis_config={"include_numbers": False},
        ).shift(SAMP_SHIFT)
        clean_bell = clean_ax.plot(
            lambda x: normal_pdf(x, CLT_MEAN, CLT_STD) * HIST_BW,
            x_range=[0.7, 3.3],
            color=YELLOW,
            stroke_width=3.8,
        )
        self.play(
            FadeOut(VGroup(self._hist, self._count_lbl, self._samp_tips, self._samp_lbl, self._n_eq, bell_lbl)),
            Transform(self._samp_ax, clean_ax),
            Transform(bell, clean_bell),
            run_time=1.0,
        )

        self._bell     = bell
        self._bell_lbl = bell_lbl

    # ── Act 6 : CLT formula + standardised end card ───────────────────────────

    def _act6_formula(self):
        # Fade out the population panel to free up left-side space
        self.play(FadeOut(VGroup(
            self._pop_ax, self._pop_curve, self._pop_area,
            self._pop_lbl, self._mean_line, self._mu_lbl, self._sig_lbl,
        )))

        # ── Main CLT statement ────────────────────────────────────────────────
        clt_main = MathTex(
            r"\bar{X}_n \;\xrightarrow{d}\; \mathcal{N}\!\left(\mu,\;\frac{\sigma^2}{n}\right)",
            font_size=44,
        ).move_to(LEFT * 3.2 + UP * 1.6)

        clt_mean_eq = MathTex(
            r"\mathbb{E}[\bar{X}] = \mu = 2",
            font_size=30, color=YELLOW,
        ).next_to(clt_main, DOWN, buff=0.5)

        clt_std_eq = MathTex(
            r"\operatorname{SD}(\bar{X}) = \frac{\sigma}{\sqrt{n}}"
            r"= \frac{2}{\sqrt{30}} \approx 0.37",
            font_size=27, color=GREEN_B,
        ).next_to(clt_mean_eq, DOWN, buff=0.35)

        clt_note = Text(
            "Works for ANY distribution\nwith finite variance",
            font_size=22, color=GRAY_B, line_spacing=1.2,
        ).next_to(clt_std_eq, DOWN, buff=0.45)

        clt_header = Text("The Central Limit Theorem", font_size=32, color=YELLOW, weight=BOLD)
        clt_header.to_edge(UP).shift(DOWN * 0.15)
        self.play(Write(clt_header))

        explain_1 = self._caption("First: sample means settle around the true mean")
        explain_2 = self._caption("Then: spread shrinks by 1/sqrt(n)")
        explain_3 = self._caption("Now combine both facts into the CLT statement")
        self.play(Write(explain_1))
        self.play(Write(clt_main))
        self.play(Transform(explain_1, explain_2))
        self.play(FadeIn(clt_mean_eq, shift=UP * 0.15))
        self.play(FadeIn(clt_std_eq,  shift=UP * 0.15))
        self.play(Transform(explain_1, explain_3))
        self.play(FadeIn(clt_note))
        self.wait(3.0)
        self.play(FadeOut(explain_1))

        # ── End card ──────────────────────────────────────────────────────────
        self._hist.clear_updaters()
        self._count_lbl.clear_updaters()

        self.play(FadeOut(VGroup(
            self._samp_ax, self._samp_tips, self._samp_lbl, self._n_eq,
            self._hist, self._count_lbl, self._bell, self._bell_lbl,
            self._pop_axis_tips, clt_main, clt_mean_eq, clt_std_eq, clt_note, clt_header,
        )))

        end_title = Text("Central Limit Theorem", font_size=50, weight=BOLD)
        end_sub = Text(
            "Sample means converge to Normal as n grows",
            font_size=26, color=GRAY_B,
        ).next_to(end_title, DOWN, buff=0.45)
        end_eq = MathTex(
            r"\sqrt{n}\;\frac{\bar{X}_n - \mu}{\sigma}"
            r"\;\xrightarrow{d}\;\mathcal{N}(0,\,1)",
            font_size=38,
        ).next_to(end_sub, DOWN, buff=0.5)

        self.play(Write(end_title))
        self.play(FadeIn(end_sub, shift=UP * 0.2))
        self.play(Write(end_eq))
        self.wait(3.5)
