from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[3]
PAPER_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "experiments" / "results"

PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#6E6E6E",
    "light_gray": "#D9D9D9",
    "soft_blue": "#DCEAF7",
    "soft_green": "#DDF3EA",
    "soft_orange": "#FBE9CC",
    "soft_red": "#F8DDD7",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(PAPER_DIR / name, format="pdf", bbox_inches="tight")
    plt.close(fig)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_results_path(*candidates: str) -> Path:
    for candidate in candidates:
        path = RESULTS_DIR / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not resolve any of: {candidates}")


def short_instance_label(name: str) -> str:
    mapping = {
        "neos-2657525-crna": "crna",
        "neos-4338804-snowy": "snowy",
        "neos-3046615-murg": "murg",
        "neos-4954672-berkel": "berkel",
        "ic97_potential": "ic97",
        "gmu-35-40": "gmu35",
        "gmu-35-50": "gmu50",
        "beasleyC3": "beasley",
        "glass4": "glass4",
        "mcsched": "mcsched",
        "50v-10": "50v10",
        "p200x1188c": "p200",
        "cod105": "cod105",
    }
    return mapping.get(name, name)


def make_framework() -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    band = FancyBboxPatch(
        (0.08, 0.83),
        0.84,
        0.10,
        boxstyle="round,pad=0.01,rounding_size=0.025",
        facecolor=PALETTE["soft_orange"],
        edgecolor=PALETTE["orange"],
        linewidth=1.1,
    )
    ax.add_patch(band)
    ax.text(
        0.50,
        0.88,
        "Optional two-phase controller: feasibility-first emphasis \u2192 nominal optimization balance",
        ha="center",
        va="center",
        fontsize=9,
        color="#4D3A00",
    )

    boxes = [
        (0.05, 0.40, 0.16, 0.22, PALETTE["soft_blue"], PALETTE["blue"], "Population\ninitialization"),
        (0.24, 0.40, 0.16, 0.22, PALETTE["soft_blue"], PALETTE["blue"], "PDHG\nrelaxation update"),
        (0.43, 0.40, 0.16, 0.22, PALETTE["soft_orange"], PALETTE["orange"], "WKB-inspired\ntunneling"),
        (0.62, 0.40, 0.16, 0.22, PALETTE["soft_green"], PALETTE["green"], "Progressive\nmeasurement"),
        (0.81, 0.40, 0.14, 0.22, PALETTE["soft_red"], PALETTE["red"], "Repair and\nincumbent check"),
    ]

    for x, y, w, h, fc, ec, label in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.03",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.2,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)

    def arrow(x0, y0, x1, y1, rad=0.0):
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.1,
                color=PALETTE["gray"],
                connectionstyle=f"arc3,rad={rad}",
            )
        )

    arrow(0.21, 0.51, 0.24, 0.51)
    arrow(0.40, 0.51, 0.43, 0.51)
    arrow(0.59, 0.51, 0.62, 0.51)
    arrow(0.78, 0.51, 0.81, 0.51)
    arrow(0.88, 0.39, 0.13, 0.39, rad=-0.35)

    ax.text(0.50, 0.15, "population refresh loop", ha="center", va="center", color=PALETTE["gray"], fontsize=8)

    save(fig, "framework_overview_nature.pdf")


def make_graphical_abstract() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 3.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    nodes = [
        (0.04, 0.22, 0.16, 0.54, PALETTE["soft_blue"], PALETTE["blue"], "MIP instance\n(MPS, bounds,\nconstraint senses)"),
        (0.24, 0.22, 0.18, 0.54, PALETTE["soft_blue"], PALETTE["blue"], "Population PDHG\non LP relaxation"),
        (0.46, 0.22, 0.16, 0.54, PALETTE["soft_orange"], PALETTE["orange"], "Barrier-aware\ntunneling"),
        (0.66, 0.22, 0.14, 0.54, PALETTE["soft_green"], PALETTE["green"], "Progressive\nmeasurement"),
        (0.83, 0.22, 0.13, 0.54, PALETTE["soft_red"], PALETTE["red"], "Strictly feasible\nincumbent"),
    ]

    for x, y, w, h, fc, ec, label in nodes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.03",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.2,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)

    for x0, x1 in [(0.20, 0.24), (0.42, 0.46), (0.62, 0.66), (0.80, 0.83)]:
        ax.add_patch(
            FancyArrowPatch(
                (x0, 0.49),
                (x1, 0.49),
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.2,
                color=PALETTE["gray"],
            )
        )

    ax.text(0.33, 0.10, "continuous relaxation quality", ha="center", va="center", color=PALETTE["gray"], fontsize=8)
    ax.text(0.54, 0.10, "\u2192", ha="center", va="center", color=PALETTE["gray"], fontsize=12)
    ax.text(0.74, 0.10, "integrality contraction + repair", ha="center", va="center", color=PALETTE["gray"], fontsize=8)

    save(fig, "graphical_abstract_nature.pdf")


def make_measurement_schedule() -> None:
    progress = np.linspace(0, 1, 300)
    strength = 0.1 + 0.9 * 0.5 * (1 - np.cos(np.pi * progress))

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.axvspan(0.0, 0.45, color=PALETTE["soft_blue"], alpha=0.8, lw=0)
    ax.axvspan(0.45, 1.0, color=PALETTE["soft_green"], alpha=0.8, lw=0)
    ax.plot(progress, strength, color=PALETTE["green"], linewidth=2.4)
    ax.scatter([0.0, 0.5, 1.0], [strength[0], strength[len(strength)//2], strength[-1]], color=PALETTE["green"], s=20, zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Normalized iteration")
    ax.set_ylabel("Measurement strength")
    ax.text(0.22, 0.92, "exploration dominated", ha="center", va="center", fontsize=8, color=PALETTE["blue"])
    ax.text(0.73, 0.92, "integer commitment", ha="center", va="center", fontsize=8, color=PALETTE["green"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=PALETTE["light_gray"], linewidth=0.7, alpha=0.7)
    save(fig, "measurement_schedule_nature.pdf")


def make_tunneling_figures() -> None:
    x = np.linspace(-3.2, 3.2, 600)
    potential = 0.26 + 0.85 * np.exp(-(x / 0.78) ** 2) + 0.025 * (x**2)

    fig, ax = plt.subplots(figsize=(4.1, 3.0))
    ax.fill_between(x, 0, potential, color=PALETTE["soft_orange"], alpha=0.95, zorder=1)
    ax.plot(x, potential, color=PALETTE["orange"], linewidth=2.0, zorder=2)

    energy = 0.53
    ax.hlines(
        energy,
        -3.0,
        3.0,
        colors=PALETTE["gray"],
        linestyles=(0, (4, 3)),
        linewidth=1.0,
        zorder=1,
    )
    ax.text(-2.85, energy + 0.035, "particle energy", color=PALETTE["gray"], fontsize=8)

    ax.annotate(
        "",
        xy=(2.0, 1.06),
        xytext=(-2.0, 1.06),
        arrowprops=dict(
            arrowstyle="-|>",
            color=PALETTE["red"],
            lw=1.8,
            connectionstyle="arc3,rad=0.34",
        ),
        zorder=3,
    )
    ax.text(-0.15, 1.17, "thermal escape", color=PALETTE["red"], fontsize=8, ha="center")

    ax.annotate(
        "",
        xy=(1.95, 0.18),
        xytext=(-1.95, 0.18),
        arrowprops=dict(
            arrowstyle="-|>",
            color=PALETTE["blue"],
            lw=1.9,
            connectionstyle="arc3,rad=-0.06",
        ),
        zorder=3,
    )
    ax.text(0.0, 0.06, "WKB-inspired tunneling", color=PALETTE["blue"], fontsize=8, ha="center")

    ax.text(-2.3, 0.26, "start basin", fontsize=8, color=PALETTE["gray"])
    ax.text(1.55, 0.26, "target basin", fontsize=8, color=PALETTE["gray"])

    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(0, 1.28)
    ax.set_xlabel("Search coordinate")
    ax.set_ylabel("Penalized energy")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    save(fig, "tunneling_barrier_nature.pdf")

    barrier_height = np.linspace(0.0, 4.0, 400)
    beta = 1.15
    gamma_d = 1.15
    thermal_escape = np.exp(-beta * barrier_height)
    wkb_slice = np.exp(-gamma_d * np.sqrt(barrier_height))

    fig, ax = plt.subplots(figsize=(4.1, 3.0))
    ax.plot(
        barrier_height,
        wkb_slice,
        color=PALETTE["blue"],
        linewidth=2.3,
        label="WKB-inspired tunneling",
    )
    ax.plot(
        barrier_height,
        thermal_escape,
        color=PALETTE["red"],
        linewidth=2.3,
        label="thermal escape",
    )
    ax.axvspan(0.0, 1.0, color=PALETTE["soft_red"], alpha=0.45, lw=0)
    ax.axvspan(1.0, 4.0, color=PALETTE["soft_blue"], alpha=0.40, lw=0)
    ax.axvline(1.0, color=PALETTE["gray"], linestyle=(0, (3, 3)), linewidth=0.9)

    label_box = dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="none", alpha=0.82)
    ax.text(
        0.62,
        0.85,
        "thermal-favored\nlow-barrier regime",
        color=PALETTE["red"],
        fontsize=8,
        ha="center",
        bbox=label_box,
    )
    ax.text(
        2.95,
        0.86,
        "quantum-favored\nhigh-barrier regime",
        color=PALETTE["blue"],
        fontsize=8,
        ha="center",
        bbox=label_box,
    )
    ax.text(
        2.62,
        0.62,
        "WKB-inspired tunneling",
        color=PALETTE["blue"],
        fontsize=8,
        ha="center",
        bbox=label_box,
    )
    ax.text(
        2.12,
        0.09,
        "thermal escape",
        color=PALETTE["red"],
        fontsize=8,
        ha="center",
        bbox=label_box,
    )

    ax.set_xlim(0, 4.0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Scaled barrier height")
    ax.set_ylabel("Acceptance probability")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=PALETTE["light_gray"], linewidth=0.7, alpha=0.7)
    save(fig, "tunneling_probability_nature.pdf")


def make_p0282_panels() -> None:
    benchmark = load_json(resolve_results_path("paper_benchmark_verified.json"))
    record = next(item for item in benchmark["results"] if item["instance_name"] == "p0282")

    labels = ["Standard\nPop-PDHG", "Quantum\nPop-PDHG"]
    violations = [
        record["standard_pdhg_result"]["primal_violation"],
        record["quantum_pdhg_result"]["primal_violation"],
    ]
    gaps = [
        abs(record["standard_pdhg_result"]["gap_to_gurobi"]),
        abs(record["quantum_pdhg_result"]["gap_to_gurobi"]),
    ]
    colors = [PALETTE["orange"], PALETTE["green"]]

    fig, ax = plt.subplots(figsize=(2.7, 2.7))
    ax.bar(labels, violations, color=colors, width=0.62)
    ax.axhline(1e-6, color=PALETTE["gray"], linestyle="--", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_ylabel("Maximum violation")
    ax.set_ylim(1e-7, 2e-1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=PALETTE["light_gray"], linewidth=0.7, alpha=0.7)
    save(fig, "p0282_feasibility_nature.pdf")

    fig, ax = plt.subplots(figsize=(2.7, 2.7))
    ax.bar(labels, gaps, color=colors, width=0.62)
    ax.set_ylabel("Absolute gap (%)")
    ax.set_ylim(0, max(gaps) * 1.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=PALETTE["light_gray"], linewidth=0.7, alpha=0.7)
    save(fig, "p0282_gap_nature.pdf")

    fig, ax = plt.subplots(figsize=(2.9, 2.8))
    x = [violations[0], violations[1], 1e-8]
    y = [gaps[0], gaps[1], 0.0]
    names = ["Standard", "Quantum", "Gurobi"]
    point_colors = [PALETTE["orange"], PALETTE["green"], PALETTE["blue"]]
    ax.scatter(x, y, s=[70, 70, 55], color=point_colors, zorder=3)
    ax.plot(x[:2], y[:2], color=PALETTE["gray"], linestyle="--", linewidth=1.0, zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("Maximum violation")
    ax.set_ylabel("Absolute gap (%)")
    ax.set_xlim(5e-8, 2e-1)
    ax.set_ylim(-0.01, max(y[:2]) * 1.45)
    for xv, yv, name, color, dx, dy in [
        (x[0], y[0], names[0], point_colors[0], 1.12, 0.012),
        (x[1], y[1], names[1], point_colors[1], 1.10, -0.012),
        (x[2], y[2], names[2], point_colors[2], 1.45, 0.010),
    ]:
        ax.text(xv * dx, yv + dy, name, color=color, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color=PALETTE["light_gray"], linewidth=0.7, alpha=0.7)
    save(fig, "p0282_tradeoff_nature.pdf")


def summarize_sensitivity(parameter: str):
    data = load_json(RESULTS_DIR / "parameter_sensitivity_results.json")["results"]
    summary = {}
    for instance in ["p0033", "knapsack_50"]:
        subset = [r for r in data if r["instance"] == instance and r["parameter"] == parameter]
        vals = sorted({r["param_value"] for r in subset}, key=lambda value: float(value))
        xs, ys = [], []
        for value in vals:
            runs = [r for r in subset if r["param_value"] == value]
            xs.append(float(value))
            summary[instance, value] = runs
            ys.append(runs)
        yield instance, vals, summary


def make_sensitivity_figures() -> None:
    data = load_json(resolve_results_path("parameter_sensitivity_results.json"))["results"]

    fig, ax = plt.subplots(figsize=(4.1, 3.0))
    for instance, color, marker in [
        ("p0033", PALETTE["blue"], "o"),
        ("knapsack_50", PALETTE["orange"], "s"),
    ]:
        subset = [r for r in data if r["instance"] == instance and r["parameter"] == "tunnel_interval"]
        vals = sorted({float(r["param_value"]) for r in subset})
        rates = []
        for value in vals:
            runs = [r for r in subset if float(r["param_value"]) == value]
            rates.append(np.mean([100 * r["tunnel_success_rate"] for r in runs]))
        ax.plot(vals, rates, color=color, marker=marker, linewidth=2.0, markersize=5, label=instance)
    ax.set_xlabel("Tunnel interval")
    ax.set_ylabel("Accepted-jump rate (%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color=PALETTE["light_gray"], linewidth=0.7, alpha=0.7)
    ax.legend(frameon=False)
    save(fig, "sensitivity_tunnel_interval_nature.pdf")

    fig, ax = plt.subplots(figsize=(4.1, 3.0))
    for instance, color, marker in [
        ("p0033", PALETTE["blue"], "o"),
        ("knapsack_50", PALETTE["orange"], "s"),
    ]:
        subset = [r for r in data if r["instance"] == instance and r["parameter"] == "population_size"]
        vals = sorted({float(r["param_value"]) for r in subset})
        runtimes = []
        for value in vals:
            runs = [r for r in subset if float(r["param_value"]) == value]
            runtimes.append(np.mean([r["runtime"] for r in runs]))
        ax.plot(vals, runtimes, color=color, marker=marker, linewidth=2.0, markersize=5, label=instance)
    ax.set_xlabel("Population size")
    ax.set_ylabel("Runtime (s)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color=PALETTE["light_gray"], linewidth=0.7, alpha=0.7)
    ax.legend(frameon=False)
    save(fig, "sensitivity_population_size_nature.pdf")


def make_sec_extended_benchmark() -> None:
    data = load_json(resolve_results_path("sec_extended_benchmark_v2.json", "sec_extended_benchmark.json"))["instance_summaries"]
    labels = [short_instance_label(row["instance_name"]) for row in data]
    standard = [row["standard_pdhg"]["mean_primal_violation"] for row in data]
    quantum = [row["quantum_pdhg"]["mean_primal_violation"] for row in data]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    ax.bar(x - width / 2, standard, width=width, color=PALETTE["orange"], label="Standard Pop-PDHG")
    ax.bar(x + width / 2, quantum, width=width, color=PALETTE["green"], label="Quantum Pop-PDHG")
    ax.set_yscale("log")
    ax.set_ylabel("Mean maximum violation")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=PALETTE["light_gray"], linewidth=0.7, alpha=0.7)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    save(fig, "sec_official_benchmark_nature.pdf")


def make_sec_metaheuristic_ranks() -> None:
    stats = load_json(resolve_results_path("sec_statistics_v3_6inst.json", "sec_statistics.json"))["meta"]["average_ranks"]
    labels = [row["solver"] for row in stats]
    values = [row["average_rank"] for row in stats]
    colors = []
    for label in labels:
        if label == "quantum_pdhg":
            colors.append(PALETTE["green"])
        elif label == "standard_pdhg":
            colors.append(PALETTE["orange"])
        elif label == "shade":
            colors.append(PALETTE["purple"])
        elif label == "de":
            colors.append(PALETTE["blue"])
        elif label == "ga":
            colors.append(PALETTE["red"])
        else:
            colors.append(PALETTE["gray"])

    pretty = {
        "quantum_pdhg": "Quantum Pop-PDHG",
        "standard_pdhg": "Standard Pop-PDHG",
        "shade": "SHADE",
        "de": "DE",
        "ga": "GA",
        "slpso": "SL-PSO",
    }

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    ax.barh(y, values, color=colors, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels([pretty.get(label, label) for label in labels])
    ax.set_xlabel("Average Friedman rank")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color=PALETTE["light_gray"], linewidth=0.7, alpha=0.7)
    for yi, value in zip(y, values):
        ax.text(value + 0.05, yi, f"{value:.2f}", va="center", ha="left", fontsize=8)
    save(fig, "sec_metaheuristic_ranks_nature.pdf")


def make_sec_metaheuristic_heatmap() -> None:
    payload = load_json(resolve_results_path("sec_metaheuristic_benchmark_v3_6inst.json", "sec_metaheuristic_benchmark.json"))
    summary_rows = payload["summary_rows"]
    solver_order = [
        row["solver"]
        for row in load_json(resolve_results_path("sec_statistics_v3_6inst.json", "sec_statistics.json"))["meta"]["average_ranks"]
    ]
    instances = sorted(
        {row["instance_name"] for row in summary_rows},
        key=lambda name: next(
            item["n_vars"] for item in payload["instances"] if item["instance_name"] == name
        ),
    )

    matrix = np.zeros((len(solver_order), len(instances)), dtype=float)
    for i, solver in enumerate(solver_order):
        for j, instance in enumerate(instances):
            row = next(
                item
                for item in summary_rows
                if item["solver"] == solver and item["instance_name"] == instance
            )
            matrix[i, j] = np.log10(1.0 + row["mean_primal_violation"])

    fig, ax = plt.subplots(figsize=(6.5, 3.7))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(instances)))
    ax.set_xticklabels([short_instance_label(name) for name in instances], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(solver_order)))
    ax.set_yticklabels(
        [
            {
                "quantum_pdhg": "Quantum",
                "standard_pdhg": "Standard",
                "shade": "SHADE",
                "de": "DE",
                "ga": "GA",
                "slpso": "SL-PSO",
            }.get(name, name)
            for name in solver_order
        ]
    )

    best_rows = np.argmin(matrix, axis=0)
    for col, best_row in enumerate(best_rows):
        ax.add_patch(
            Rectangle(
                (col - 0.5, best_row - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor="white",
                linewidth=1.3,
            )
        )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\log_{10}(1 + \mathrm{mean\ violation})$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, "sec_metaheuristic_heatmap_nature.pdf")


def main() -> None:
    setup_style()
    make_graphical_abstract()
    make_framework()
    make_measurement_schedule()
    make_tunneling_figures()
    make_p0282_panels()
    make_sensitivity_figures()
    make_sec_extended_benchmark()
    make_sec_metaheuristic_ranks()
    make_sec_metaheuristic_heatmap()


if __name__ == "__main__":
    main()
