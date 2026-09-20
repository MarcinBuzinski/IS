"""Pirminė duomenų apžvalga ir paveikslai kolokviumo planui.

Sugeneruoja:
  doc/figures/fig_quality_hist.png   – kokybės balų pasiskirstymas (raudonas / baltas)
  doc/figures/fig_corr.png           – požymių Pearson koreliacija su kokybės balu
  doc/figures/fig_pipeline.png       – sprendimo etapų schema
  doc/figures/fig_flowchart.png      – veiksmų eiliškumo (algoritmo) schema
  doc/figures/fig_eps_loss.png       – ε-nejautri nuostolio funkcija (SVR) palyginimui
  build/stats.json                   – skaičiai, naudojami dokumento lentelėse
  build/stats.md                     – tie patys skaičiai Markdown lentelėmis (kopijavimui / patikrai)

Paleidimas:  python scripts/explore_data.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, RepeatedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
FIG = ROOT / "doc" / "figures"
BUILD = ROOT / "build"
FIG.mkdir(parents=True, exist_ok=True)
BUILD.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "figure.dpi": 200})
RED, WHITE = "#8b1e3f", "#c9a227"


def load() -> dict[str, pd.DataFrame]:
    return {
        "red": pd.read_csv(RAW / "winequality-red.csv", sep=";"),
        "white": pd.read_csv(RAW / "winequality-white.csv", sep=";"),
    }


def quality_hist(data: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0), sharey=False)
    for ax, (name, color, title) in zip(
        axes, [("red", RED, "Raudonas vynas (n = 1599)"), ("white", WHITE, "Baltas vynas (n = 4898)")]
    ):
        vc = data[name]["quality"].value_counts().sort_index()
        pct = 100 * vc / vc.sum()
        ax.bar(vc.index, vc.values, color=color, width=0.7)
        for x, v, p in zip(vc.index, vc.values, pct):
            ax.text(x, v, f"{v}\n({p:.1f}%)", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(3, 10))
        ax.set_xlabel("Ekspertų kokybės balas (quality)")
        ax.set_ylabel("Mėginių skaičius")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, vc.max() * 1.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_quality_hist.png")
    plt.close(fig)


def corr_plot(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    corr = pd.DataFrame(
        {name: d.corr(numeric_only=True)["quality"].drop("quality") for name, d in data.items()}
    )
    order = corr.abs().mean(axis=1).sort_values().index
    corr = corr.loc[order]
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    y = np.arange(len(corr))
    ax.barh(y - 0.18, corr["red"], height=0.36, color=RED, label="raudonas")
    ax.barh(y + 0.18, corr["white"], height=0.36, color=WHITE, label="baltas")
    ax.set_yticks(y)
    ax.set_yticklabels(corr.index)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Pearson koreliacija su kokybės balu")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_corr.png")
    plt.close(fig)
    return corr


def pipeline_fig() -> None:
    stages = [
        ("1. Duomenų\ngavimas", "UCI CSV ×2,\nSHA-256, metaduomenys"),
        ("2. Paruošimas", "tipas, dublikatų grupės,\nlog1p, standartizavimas,\nskaidymas"),
        ("3. Modelių\nmokymas", "baseline, SVR, MLP,\nRBF, RF; nested CV"),
        ("4. Vertinimas", "MAE, Acc_T, makro-F1,\npainiavos matrica,\nabliacija, testai"),
        ("5. Naudojimas", "CLI / API: balas,\nklasė, tolerancijos\nvėliava, stebėsena"),
    ]
    fig, ax = plt.subplots(figsize=(9.0, 2.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    w, h = 1.7, 1.9
    xs = np.linspace(0.15, 10 - w - 0.15, len(stages))
    for i, (x, (title, body)) in enumerate(zip(xs, stages)):
        box = FancyBboxPatch(
            (x, 0.5), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
            fc="#f4f1ea" if i != 2 else "#e8dcc3", ec="#4a4a4a", lw=1.0,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, 0.5 + h - 0.32, title, ha="center", va="top", fontsize=9.5, weight="bold")
        ax.text(x + w / 2, 0.5 + 0.12, body, ha="center", va="bottom", fontsize=7.6)
        if i < len(stages) - 1:
            ax.add_patch(
                FancyArrowPatch((x + w + 0.03, 1.45), (xs[i + 1] - 0.03, 1.45),
                                arrowstyle="-|>", mutation_scale=12, lw=1.0, color="#4a4a4a")
            )
    fig.tight_layout()
    fig.savefig(FIG / "fig_pipeline.png")
    plt.close(fig)


def flowchart_fig() -> None:
    """Algoritmo (veiksmų eiliškumo) schema: mokymo kontūras kairėje, naudojimo kontūras dešinėje."""
    fig, ax = plt.subplots(figsize=(9.0, 7.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10.4)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#f4f1ea", fs=7.8, bold=False, shape="round"):
        style = "round,pad=0.02,rounding_size=0.10" if shape == "round" else "square,pad=0.02"
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=style, fc=fc, ec="#4a4a4a", lw=0.9))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal")

    def diamond(x, y, w, h, text, fs=7.4):
        pts = [(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)]
        ax.add_patch(plt.Polygon(pts, closed=True, fc="#fbe7c6", ec="#4a4a4a", lw=0.9))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    def arrow(p, q, text=None, tx=0.12):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, lw=0.9, color="#4a4a4a",
                                     shrinkA=0, shrinkB=0))
        if text:
            ax.text((p[0] + q[0]) / 2 + tx, (p[1] + q[1]) / 2, text, fontsize=7, ha="left", va="center")

    # --- Mokymo kontūras (kairė kolona) ---
    lx = 2.6
    ax.text(lx, 10.15, "A. MOKYMO IR VERTINIMO KONTŪRAS (vykdomas vieną kartą)", ha="center", fontsize=8.5, weight="bold")
    steps = [
        (9.45, "1. Nuskaityti CSV (raudonas, baltas);\npatikrinti SHA-256, stulpelius, diapazonus"),
        (8.45, "2. Pridėti požymį „type“; sudaryti dublikatų\ngrupes (identiškos 11 požymių eilutės)"),
        (7.45, "3. Stratifikuotas grupinis skaidymas 80/20;\ntestinė aibė užšaldoma"),
        (6.45, "4. Kiekvienam modeliui m ∈ {vidurkis, LR,\nSVR, MLP, RBF, RF}: įdėtinė CV\n(vidinė 5-fold HPO, išorinė 10-fold × 3)"),
        (5.35, "5. Statistinis palyginimas per 30 išorinių\nklosčių: Wilcoxon + Holm; H1–H4 tikrinimas"),
    ]
    for y, t in steps:
        box(lx, y, 4.6, 0.78 if "\n\n" not in t else 0.95, t)
    for (y1, _), (y2, _) in zip(steps[:-1], steps[1:]):
        arrow((lx, y1 - 0.39), (lx, y2 + 0.39))
    diamond(lx, 4.15, 3.4, 1.0, "H1 pasitvirtino\n(SVR < 0,95·LR, p < 0,05)?")
    arrow((lx, 5.35 - 0.39), (lx, 4.15 + 0.5))
    box(1.4, 3.05, 2.05, 0.75, "6a. Pagrindinis\nmodelis = SVR-RBF", fc="#e8dcc3")
    box(3.8, 3.05, 2.05, 0.75, "6b. Paprasčiausias\nne blogesnis modelis", fc="#e8dcc3")
    arrow((lx - 0.9, 4.15 - 0.27), (1.4, 3.05 + 0.38), "taip", tx=-0.55)
    arrow((lx + 0.9, 4.15 - 0.27), (3.8, 3.05 + 0.38), "ne", tx=0.15)
    box(lx, 2.0, 4.6, 0.78, "7. Permokyti su visa mokymo aibe (80 %);\nvienkartinis vertinimas testinėje aibėje (20 %):\nMAE, Acc_T, makro-F1, painiavos matrica")
    arrow((1.4, 3.05 - 0.38), (1.4, 2.0 + 0.39))
    arrow((3.8, 3.05 - 0.38), (3.8, 2.0 + 0.39))
    box(lx, 0.95, 4.6, 0.78, "8. Abliacija, klaidų analizė, jautrumo analizė;\nmodelio artefaktas (scaler + SVR, joblib) + ataskaita")
    arrow((lx, 2.0 - 0.39), (lx, 0.95 + 0.39))

    # --- Naudojimo kontūras (dešinė kolona) ---
    rx = 7.55
    ax.text(rx, 10.15, "B. NAUDOJIMO KONTŪRAS (kiekvienam naujam mėginiui)", ha="center", fontsize=8.5, weight="bold")
    inf = [
        (9.45, "Įvestis: 11 matavimų + tipas (JSON/CSV eilutė)"),
        (8.45, "Validacija: trūkstamos reikšmės, fizikiniai\ndiapazonai (pvz., pH 2,5–4,5) → klaida/įspėjimas"),
        (7.45, "log1p pasirinktiems požymiams;\nstandartizavimas z = (x − μ) / σ (μ, σ iš mokymo)"),
        (6.45, "SVR sprendimo funkcija\nf(x) = Σ(αᵢ − αᵢ*)·K(xᵢ, x) + b"),
        (5.35, "Postprocesas: q̂ = clip(round(f(x)), 3, 9);\nklasė c(q̂) ∈ {žema, vidutinė, aukšta}"),
        (4.15, "Patikimumo požymis: atstumas iki\napvalinimo ribos |f(x) − q̂| ir Acc_T=1 istorija"),
        (3.05, "Jei yra ekspertų balas y: vėliava,\nkai |f(x) − y| > 1 → pakartotinė degustacija"),
        (2.0, "Išvestis: {score_cont, score_int, class,\nconfidence, flags} + įrašas į stebėsenos žurnalą"),
    ]
    for y, t in inf:
        box(rx, y, 4.3, 0.78, t, fc="#eef2f7" if y not in (6.45,) else "#d9e4f0")
    for (y1, _), (y2, _) in zip(inf[:-1], inf[1:]):
        arrow((rx, y1 - 0.39), (rx, y2 + 0.39))
    box(rx, 0.95, 4.3, 0.78, "Stebėsena: požymių dreifas (PSI), MAE pagal\nmėnesį; permokymas, kai MAE ↑ > 10 % nuo bazinio", fc="#eef2f7")
    arrow((rx, 2.0 - 0.39), (rx, 0.95 + 0.39))

    # jungtis: mokymo artefaktas -> naudojimas (laužtė per tarpą tarp kolonų)
    cx = (lx + 2.3 + rx - 2.15) / 2
    ax.plot([lx + 2.3, cx, cx], [0.95, 0.95, 6.45], color="#8b1e3f", lw=0.9, ls="--")
    ax.add_patch(FancyArrowPatch((cx, 6.45), (rx - 2.15, 6.45), arrowstyle="-|>", mutation_scale=10,
                                 lw=0.9, color="#8b1e3f", linestyle="--", shrinkA=0, shrinkB=0))
    ax.text(cx + 0.09, 3.7, "modelio artefaktas", fontsize=6.8, color="#8b1e3f", ha="left", va="center", rotation=90)

    fig.tight_layout()
    fig.savefig(FIG / "fig_flowchart.png")
    plt.close(fig)


def eps_loss_fig() -> None:
    r = np.linspace(-2, 2, 400)
    eps = 0.5
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.plot(r, r**2, "--", color="#777777", label="kvadratinė (MLP, MNK)")
    ax.plot(r, np.abs(r), ":", color="#777777", label="absoliutinė (MAE)")
    ax.plot(r, np.maximum(0, np.abs(r) - eps), color=RED, lw=2.2, label=r"$\varepsilon$-nejautri, $\varepsilon=0{,}5$ (SVR)")
    ax.axvspan(-eps, eps, color=RED, alpha=0.08)
    ax.set_xlabel(r"liekana $r = y - f(x)$ (balais)")
    ax.set_ylabel("nuostolis")
    ax.set_ylim(0, 2.2)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_eps_loss.png")
    plt.close(fig)


def baseline_numbers(data: dict[str, pd.DataFrame]) -> dict:
    """Orientacinės baseline reikšmės (10-fold × 3; grupinė CV pagal dublikatus)."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    models = {
        "vidurkis": DummyRegressor(strategy="mean"),
        "mediana": DummyRegressor(strategy="median"),
        "tiesine_regresija": make_pipeline(StandardScaler(), LinearRegression()),
    }
    for name, d in data.items():
        X = d.drop(columns="quality").to_numpy()
        y = d["quality"].to_numpy()
        groups = pd.factorize(pd.util.hash_pandas_object(d.drop(columns="quality"), index=False))[0]
        out[name] = {}
        for mname, m in models.items():
            s_plain = -cross_val_score(
                m, X, y, cv=RepeatedKFold(n_splits=10, n_repeats=3, random_state=42),
                scoring="neg_mean_absolute_error", n_jobs=-1,
            )
            s_group = -cross_val_score(
                m, X, y, groups=groups, cv=GroupKFold(n_splits=10),
                scoring="neg_mean_absolute_error", n_jobs=-1,
            )
            out[name][mname] = {
                "mae_kfold": round(float(s_plain.mean()), 3),
                "mae_kfold_std": round(float(s_plain.std()), 3),
                "mae_groupkfold": round(float(s_group.mean()), 3),
                "mae_groupkfold_std": round(float(s_group.std()), 3),
            }
    return out


def main() -> None:
    data = load()
    quality_hist(data)
    corr = corr_plot(data)
    pipeline_fig()
    flowchart_fig()
    eps_loss_fig()

    stats: dict = {"features": list(data["red"].columns[:-1])}
    md = ["# Automatiškai suskaičiuoti duomenų faktai\n"]
    for name, d in data.items():
        vc = d["quality"].value_counts().sort_index()
        dups = int(d.duplicated().sum())
        conflicts = int((d.groupby(list(d.columns[:-1]))["quality"].nunique() > 1).sum())
        stats[name] = {
            "n": int(len(d)),
            "quality_counts": {int(k): int(v) for k, v in vc.items()},
            "quality_mean": round(float(d["quality"].mean()), 3),
            "quality_std": round(float(d["quality"].std()), 3),
            "duplicates": dups,
            "duplicates_pct": round(100 * dups / len(d), 1),
            "unique_rows": int(len(d.drop_duplicates())),
            "conflicting_duplicate_groups": conflicts,
            "skew": {k: round(float(v), 2) for k, v in d.drop(columns="quality").skew().items()},
            "describe": {
                c: {k: round(float(v), 3) for k, v in d[c].describe()[["mean", "std", "min", "50%", "max"]].items()}
                for c in d.columns
            },
        }
        md.append(f"\n## {name}: n={len(d)}, dublikatai={dups} ({stats[name]['duplicates_pct']}%), konfliktų={conflicts}\n")
        md.append("| balas | n | % |\n|---|---|---|")
        for k, v in vc.items():
            md.append(f"| {k} | {v} | {100*v/len(d):.1f} |")
        md.append("\n| požymis | vid. | st.nuok. | min | mediana | max | asimetrija |\n|---|---|---|---|---|---|---|")
        for c in d.columns:
            s = stats[name]["describe"][c]
            sk = stats[name]["skew"].get(c, "")
            md.append(f"| {c} | {s['mean']} | {s['std']} | {s['min']} | {s['50%']} | {s['max']} | {sk} |")
    stats["corr_with_quality"] = {k: {i: round(float(v), 3) for i, v in col.items()} for k, col in corr.items()}
    md.append("\n## Koreliacija su kokybe\n\n| požymis | raudonas | baltas |\n|---|---|---|")
    for i, row in corr.iterrows():
        md.append(f"| {i} | {row['red']:.3f} | {row['white']:.3f} |")

    stats["baseline"] = baseline_numbers(data)
    md.append("\n## Baseline MAE (10-fold×3 / GroupKFold-10)\n\n| tipas | metodas | MAE k-fold | MAE grupinė |\n|---|---|---|---|")
    for name, ms in stats["baseline"].items():
        for mname, v in ms.items():
            md.append(f"| {name} | {mname} | {v['mae_kfold']} ± {v['mae_kfold_std']} | {v['mae_groupkfold']} ± {v['mae_groupkfold_std']} |")

    (BUILD / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    (BUILD / "stats.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
