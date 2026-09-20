"""Sugeneruoja egzamino ataskaitą: TNR 12 pt, tik juoda, iki ~12 psl."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Emu, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "IS_Egzaminas_Marcin_Bužinski_ESKfm_26.docx"
FIG = RESULTS / "figures_bw"
BLACK = RGBColor(0, 0, 0)


def set_run_font(run, size=12, bold=False, italic=False):
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = BLACK
    # nuimti temines / paryškinimo spalvas
    rPr = run._element.get_or_add_rPr()
    for tag in ("w:highlight", "w:shd"):
        el = rPr.find(qn(tag))
        if el is not None:
            rPr.remove(el)
    rPr.append(OxmlElement("w:color"))
    rPr.find(qn("w:color")).set(qn("w:val"), "000000")


def add_p(doc, text, *, size=12, bold=False, italic=False, align="justify", space_after=6, space_before=0, first_line=True):
    p = doc.add_paragraph()
    p.alignment = {
        "justify": WD_ALIGN_PARAGRAPH.JUSTIFY,
        "center": WD_ALIGN_PARAGRAPH.CENTER,
        "left": WD_ALIGN_PARAGRAPH.LEFT,
        "right": WD_ALIGN_PARAGRAPH.RIGHT,
    }[align]
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(space_before)
    pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
    pf.line_spacing = 1.15
    if first_line and align == "justify":
        pf.first_line_indent = Cm(0.75)
    else:
        pf.first_line_indent = Cm(0)
    run = p.add_run(text)
    set_run_font(run, size=size, bold=bold, italic=italic)
    return p


def add_h(doc, text, level=1):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf = p.paragraph_format
    pf.space_before = Pt(10 if level == 1 else 8)
    pf.space_after = Pt(6)
    pf.line_spacing = 1.15
    pf.first_line_indent = Cm(0)
    pf.keep_with_next = True
    run = p.add_run(text)
    if level == 1:
        set_run_font(run, size=12, bold=True)
    else:
        set_run_font(run, size=12, bold=True, italic=True)
    return p


def shade_none(cell):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    for old in tcPr.findall(qn("w:shd")):
        tcPr.remove(old)
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), "FFFFFF")
    shd.set(qn("w:val"), "clear")
    tcPr.append(shd)
    # juodi rėmeliai
    tcBorders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:color"), "000000")
        tcBorders.append(el)
    tcPr.append(tcBorders)


def add_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    for j, h in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.first_line_indent = Cm(0)
        run = p.add_run(h)
        set_run_font(run, size=11, bold=True)
        shade_none(cell)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.rows[i + 1].cells[j]
            cell.text = ""
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.first_line_indent = Cm(0)
            run = p.add_run(str(val))
            set_run_font(run, size=11, bold=False)
            shade_none(cell)
    if col_widths:
        for row in table.rows:
            for j, w in enumerate(col_widths):
                row.cells[j].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return table


def caption(doc, text):
    add_p(doc, text, size=11, italic=True, align="center", space_after=8, first_line=False)


def add_figure(doc, path: Path, width_cm=14.5):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.first_line_indent = Cm(0)
    run = p.add_run()
    run.add_picture(str(path), width=Cm(width_cm))
    set_run_font(run, size=12)


def force_styles(doc):
    for name in ("Normal", "Heading 1", "Heading 2", "Heading 3", "Caption", "Title"):
        try:
            st = doc.styles[name]
        except KeyError:
            continue
        st.font.name = "Times New Roman"
        st.font.size = Pt(12)
        st.font.color.rgb = BLACK
        st.font.highlight_color = None
        rPr = st.element.get_or_add_rPr()
        rFonts = rPr.find(qn("w:rFonts"))
        if rFonts is None:
            rFonts = OxmlElement("w:rFonts")
            rPr.append(rFonts)
        rFonts.set(qn("w:ascii"), "Times New Roman")
        rFonts.set(qn("w:hAnsi"), "Times New Roman")
        rFonts.set(qn("w:eastAsia"), "Times New Roman")
        rFonts.set(qn("w:cs"), "Times New Roman")
        color = rPr.find(qn("w:color"))
        if color is None:
            color = OxmlElement("w:color")
            rPr.append(color)
        color.set(qn("w:val"), "000000")
        color.set(qn("w:themeColor"), "")
        # pašalinti themeColor jei yra
        if "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}themeColor" in color.attrib:
            del color.attrib[qn("w:themeColor")]


def make_figures(cmp: pd.DataFrame, conf_red, conf_white):
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "text.color": "black",
            "axes.titlecolor": "black",
        }
    )
    order = ["mean", "linear", "svr", "mlp", "rbf", "rf"]
    labels = ["vidurkis", "tiesinė", "SVR", "MLP", "RBF", "miškas"]
    colors = ["#7f7f7f", "#4a4a4a", "#8b1e3f", "#2d6a4f", "#c9a227", "#4a6fa5"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), sharey=True)
    for ax, wine, title in zip(axes, ["red", "white"], ["Raudonas vynas", "Baltas vynas"]):
        d = cmp[cmp["wine"] == wine].set_index("model").loc[order]
        x = np.arange(len(order))
        ax.bar(x, d["mae"].to_numpy(), color=colors, edgecolor="black", linewidth=0.6)
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.set_title(title)
        ax.set_ylabel("MAE")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for i, v in enumerate(d["mae"]):
            ax.text(i, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=7, color="black")
        ax.set_ylim(0, 0.82)
    fig.tight_layout()
    p1 = FIG / "mae_color.png"
    fig.savefig(p1, dpi=200, facecolor="white")
    plt.close(fig)

    def conf_plot(mat, title, path):
        arr = np.asarray(mat, dtype=float)
        fig, ax = plt.subplots(figsize=(4.4, 3.6))
        im = ax.imshow(arr, cmap="Blues", vmin=0, vmax=max(arr.max(), 1))
        labs = ["žema", "vidutinė", "aukšta"]
        ax.set_xticks(range(3), labels=labs)
        ax.set_yticks(range(3), labels=labs)
        ax.set_xlabel("Prognozuota klasė")
        ax.set_ylabel("Tikra klasė")
        ax.set_title(title)
        thresh = arr.max() * 0.55 if arr.max() > 0 else 0.5
        for i in range(3):
            for j in range(3):
                val = int(arr[i, j])
                ax.text(
                    j,
                    i,
                    str(val),
                    ha="center",
                    va="center",
                    color="white" if arr[i, j] >= thresh else "black",
                    fontsize=10,
                )
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(path, dpi=200, facecolor="white")
        plt.close(fig)

    p2 = FIG / "conf_red_color.png"
    conf_plot(conf_red, "SVR, raudonas vynas, 3 klasės", p2)
    p3 = FIG / "conf_white_color.png"
    conf_plot(conf_white, "SVR, baltas vynas, 3 klasės", p3)
    return p1, p2, p3


def fmt(x, n=3):
    return f"{x:.{n}f}".replace(".", ",")


def pct(x):
    return f"{100 * x:.1f} %".replace(".", ",")


def build():
    cmp = pd.read_csv(RESULTS / "comparison.csv")
    rep = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
    red, white = rep["wines"]["red"], rep["wines"]["white"]
    fig_mae, fig_cr, fig_cw = make_figures(
        cmp, red["metrics"]["svr"]["confusion_class"], white["metrics"]["svr"]["confusion_class"]
    )

    doc = Document()
    force_styles(doc)
    sec = doc.sections[0]
    sec.top_margin = Cm(2.0)
    sec.bottom_margin = Cm(2.0)
    sec.left_margin = Cm(2.5)
    sec.right_margin = Cm(2.5)
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)

    # antraštė
    add_p(doc, "Vilniaus Gedimino technikos universitetas", align="center", first_line=False, space_after=0)
    add_p(doc, "Intelektualiosios sistemos. Magistro studijos", align="center", first_line=False, space_after=0)
    add_p(doc, "Galutinio egzamino ataskaita", size=14, bold=True, align="center", first_line=False, space_before=10, space_after=4)
    add_p(
        doc,
        "Vyno kokybės vertinimas pagal fizikinius ir cheminius matavimus",
        size=12,
        bold=True,
        align="center",
        first_line=False,
        space_after=4,
    )
    add_p(
        doc,
        "Marcin Bužinski, ESKfm-26. 2026-09-20. Visi skaičiai paimti iš results/metrics.json po komandos python -m winequality.run_experiment.",
        size=11,
        italic=True,
        align="center",
        first_line=False,
        space_after=10,
    )

    add_h(doc, "1. Problema, įvestis ir išvestis")
    add_p(
        doc,
        "Gamintojas nori iš 11 standartinių laboratorinių matavimų prognozuoti ekspertų suteiktą vyno kokybės balą. "
        "Duomenys – UCI Wine Quality (Cortez ir kt., 2009; DOI 10.1016/j.dss.2009.05.016): 1599 raudono ir 4898 balto "
        "Vinho Verde mėginiai, trūkstamų reikšmių nėra. Taikinys quality yra medianinis aklas vertinimas skalėje 0–10, "
        "šiame rinkinyje stebimi balai 3–9. Sunkumai: subjektyvus žymėjimas (vienas balas – triukšminga imtis), "
        "ordinalinė skalė, stiprus klasių disbalansas (dominuoja 5–6) ir 15,0 % / 19,1 % tikslių eilučių dublikatų, "
        "kurie atsitiktiniame skaidyme nutekina testą.",
    )
    add_p(
        doc,
        "Techninė užduotis: regresija su tolesniu apvalinimu į sveiką balą ir tris verslo klases. "
        "Įvestis – 11 realiųjų požymių pirminiais vienetais ir vyno tipas (raudonas arba baltas). "
        "Išvestis apibrėžta 1 lentelėje. Modelis nekeičia degustatorių – tai antros nuomonės ir rūšiavimo "
        "įrankis, kai |f(x) − y| > 1.",
        space_after=4,
    )
    add_p(doc, "1 lentelė. Sistemos įvestis ir išvestis.", first_line=False, space_after=4)
    add_table(
        doc,
        ["Elementas", "Tipas", "Interpretacija"],
        [
            ["Įvestis x", "11 realiųjų + tipas", "Laboratorijos eilutė; pH už 2,5–4,5 atmetama"],
            ["score_cont", "realusis f(x)", "Tikėtinas ekspertų balas, pvz. 6,3 arčiau 6 nei 7"],
            ["score_int", "sveikasis 3–9", "q̂ = min(9, max(3, round(f)))"],
            ["class", "žema / vidutinė / aukšta", "≤4 / 5–6 / ≥7"],
            ["confidence", "[0; 1]", "1 − 2|f − q̂|; 0 – ant apvalinimo ribos"],
            ["flags", "sąrašas", "extrapolation, disagreement, extreme"],
        ],
    )
    caption(doc, "1 lentelė. Sutartis: winequality/postprocess.py ir predict.py.")

    add_h(doc, "2. Duomenų paruošimo grandinė ir skaidymas")
    add_p(
        doc,
        "Grandinė: (1) atsisiuntimas iš UCI ir SHA-256 patikra (winequality/data_io.py); "
        "(2) dublikatų grupės – SHA-256 iš 11 požymių, identiškos eilutės niekada neatsiduria skirtingose pusėse "
        "(winequality/splits.py, StratifiedGroupKFold, sėkla SPLIT_SEED = 0); "
        "(3) 20 % testo aibė užšaldoma, 80 % lieka mokymui; "
        "(4) log1p asimetriškiems požymiams (cukrus, chloridai, laisvas ir bendras SO2, sulfatai) ir z-standartizavimas "
        "tik pagal mokymo klostę (winequality/transform.py: x′j = ln(1+xj), zj = (x′j − μj)/σj). "
        "Raudonas ir baltas vynas mokomi atskirai. Gauta: raudonas 1279/320, baltas 3919/979; "
        "persidengiančių grupių tarp mokymo ir testo – 0.",
    )
    add_p(
        doc,
        "Hiperparametrai parenkami RandomizedSearchCV su GroupKFold (3 klostės) tik mokymo bloke. "
        "Užšaldytas testas liestas vieną kartą, po visų metodų mokymo. "
        "Modelių sėkla DEFAULT_SEED = 0. SVR – 12 atsitiktinių konfigūracijų, MLP, RBF tinklas ir miškas – po 8.",
    )

    add_h(doc, "3. Modeliai ir formulės ryšys su kodu")
    add_p(
        doc,
        "Privalomi baseline: vidurkio prognozė ŷ = ȳ_mok (winequality/baselines.py, mean_predictor) ir "
        "tiesinė regresija f(z) = w⊤z + b po to paties log1p+z paruošimo (linear_pipeline, OLS). "
        "Intelektualieji metodai pagal kolokviumo planą: SVR su Gauso branduoliu (pagrindinis), "
        "MLP, RBF tinklas ir atsitiktinis miškas.",
        space_after=4,
    )
    add_p(
        doc,
        "SVR naudojimo formulė įgyvendinta tiesiogiai, ne tik komentaru. Branduolys "
        "K(zi, z) = exp(−γ‖zi − z‖²) yra funkcija rbf_kernel (winequality/svr_model.py, 29–39 eil.). "
        "Dualinė prognozė f(z) = Σi∈SV (αi − αi*) K(zi, z) + b yra predict_from_dual (42–55 eil.); "
        "sklearn dual_coef_ ir yra (α − α*). Naudojimo kelias formula_predict (96–108 eil.) pirmiausia taiko "
        "Log1pStandardScaler, tada šią sumą. Teste ir eksperimente max |formulė − sklearn.predict| buvo "
        "8,5·10⁻¹⁴ (raudonas) ir 1,6·10⁻¹² (baltas). Mokymas (LIBSVM/SMO) minimizuoja ½‖w‖² + C Σ(ξi+ξi*) "
        "su ε-nejautriomis nelygybėmis. Parinkti hiperparametrai – 2 lentelėje.",
        space_after=4,
    )
    add_p(
        doc,
        "MLP: f(z) gaunamas sluoksniuotu persiuntimu h = act(W1 z + b1), išėjimas w2⊤h + b2; "
        "Adam, L2 ir ankstyvasis stabdymas iš 10 % mokymo (ne testo). RBF tinklas (LD3, winequality/rbf.py): "
        "centrai k-vidurkiais, φk(z) = exp(−‖z−ck‖²/(2σk²)) funkcijoje hidden_activations, "
        "svoriai ridge uždara formule, prognozė predict_from_rbf. Miškas: medžių vidurkis, laiptuota prognozė "
        "be ekstrapoliacijos.",
        space_after=4,
    )
    add_p(doc, "2 lentelė. Vidinės GroupKFold paieškos nugalėtojai (testas nenaudotas).", first_line=False, space_after=4)
    add_table(
        doc,
        ["Modelis", "Raudonas", "Baltas"],
        [
            ["SVR C; ε; γ", "2,10; 0,05; 0,069", "2,10; 0,05; 0,069"],
            ["MLP", "tanh, 64, α=3,5·10⁻³", "ReLU, 16, α=1,5·10⁻³"],
            ["RBF tinklas", "K=80, kaim. 2, λ=0,24", "K=80, kaim. 2, λ=0,24"],
            ["Miškas", "400 medžių, gylis laisvas", "400 medžių, gylis laisvas"],
        ],
    )
    caption(doc, "2 lentelė. Šaltinis: results/metrics.json, laukai *_tune.")

    add_h(doc, "4. Metrikos")
    add_p(
        doc,
        "Visoms sistemoms tas pats rinkinys (winequality/metrics.py): MAE = (1/M) Σ |yi − f(xi)| (pagrindinė); "
        "RMSE; Acc_T = dalis, kai |yi − f| ≤ T, T ∈ {0,5; 1,0}; kvadratinė svertinė kappa κw; "
        "po apvalinimo – makro-F1 ir painiavos matricos 7×7 (balai) bei 3×3 (klasės). "
        "Klasės ribos paliktos ≤4 / 5–6 / ≥7, kad makro-F1 būtų skaičiuojamas ne tuščioms kraštinėms klasėms.",
    )

    add_h(doc, "5. Rezultatai")
    add_p(
        doc,
        "3 lentelė. Užšaldytos testo aibės metrikos. Mažesnis MAE – geriau. Šaltinis: results/comparison.csv.",
        first_line=False,
        space_after=4,
    )
    headers = ["Vynas", "Modelis", "MAE", "Acc 0,5", "Acc 1,0", "κw", "makro-F1"]
    name = {"mean": "vidurkis", "linear": "tiesinė", "svr": "SVR-RBF", "mlp": "MLP", "rbf": "RBF tinklas", "rf": "miškas"}
    rows = []
    for wine, wlt in [("raudonas", "red"), ("baltas", "white")]:
        for m in ["mean", "linear", "svr", "mlp", "rbf", "rf"]:
            r = cmp[(cmp["wine"] == wlt) & (cmp["model"] == m)].iloc[0]
            rows.append(
                [
                    wine,
                    name[m],
                    fmt(r["mae"]),
                    fmt(r["acc_0.5"]),
                    fmt(r["acc_1.0"]),
                    fmt(r["kappa_w"]),
                    fmt(r["macro_f1"]),
                ]
            )
    add_table(doc, headers, rows)
    caption(doc, "3 lentelė. Palyginimas tuo pačiu grupiniu 80/20 skaidymu.")

    add_figure(doc, fig_mae, 15.0)
    caption(doc, "1 pav. Testo MAE pagal modelį ir vyno tipą.")

    add_p(
        doc,
        "Raudonam vynui mažiausią MAE turi MLP (0,472) ir SVR (0,473), tiesinė – 0,491, miškas – 0,485, "
        "RBF tinklas – 0,497, vidurkis – 0,684. Baltam: miškas 0,532, SVR 0,534, RBF tinklas 0,552, "
        "MLP 0,572 (beveik tiesinė 0,573), vidurkis 0,673. Hipotezė H1 (SVR MAE ≤ 0,95×tiesinės) "
        "baltam galioja (gerinimas 6,9 %), raudonam – ne (tik 3,5 %). Tai neigiamas, bet korektiškas rezultatas: "
        "protokolas laikytas, slenkstis raudonai imčiai buvo griežtas. H2 (SVR − RF ≤ 0,02 MAE) galioja abiem "
        "(−0,011 ir +0,002), todėl pagal planą pagrindinis metodas lieka SVR: miškas nepranoko 0,02 slenksčio, "
        "o jo prognozė laiptuota. RBF tinklas baltam yra tarp tiesinės ir SVR – kaip ir tikėtasi iš fiksuotų "
        "centrų palyginimo su adaptyviais atraminiais vektoriais. MLP raudonam lygiavertis SVR, baltam – ne; "
        "lentelinė asimetriška imtis MLP nepalanki.",
    )

    add_figure(doc, fig_cr, 8.2)
    caption(doc, "2 pav. SVR 3×3 painiavos matrica, raudonas vynas.")
    add_figure(doc, fig_cw, 8.2)
    caption(doc, "3 pav. SVR 3×3 painiavos matrica, baltas vynas.")
    add_p(
        doc,
        "2 pav. matyti klasės disbalanso kaina: visos 12 žemų raudono vyno testo mėginių priskirti vidutinei klasei; "
        "aukšta atpažįstama 19 iš 44. Baltam (3 pav.) žemų teisingai 4 iš 37, aukštų 92 iš 212. "
        "Makro-F1 todėl SVR raudonam (0,470) žemesnis nei miško (0,548) ir MLP (0,536), nors MAE geresnis ar lygus. "
        "Tai ordinalinės regresijos ir retų klasių konfliktas: MAE baudžia visus balus vienodai, makro-F1 – retas klases. "
        "Pasirinktas MAE, nes taikinys paliktas balais, o klasės – pagalbinės.",
        space_after=4,
    )

    add_h(doc, "6. Klaidų pavyzdžiai ir jautrumas")
    add_p(
        doc,
        "Pagal tikrą balą SVR raudonam: 3 balų mėginiai (n=2) pervertinami vidutiniškai +2,10, 8 balų (n=4) nuvertinami "
        "−1,59; 5–6 balų MAE 0,33–0,46. Baltam ta pati trauka į vidurkį: 3 balai +2,09, 9 balai −2,02. "
        "Tai regresija į vidurkį ir žymių triukšmas, ne mokymo klaida siaurąja prasme.",
        space_after=4,
    )
    add_p(doc, "4 lentelė. SVR MAE ir poslinkis pagal tikrąjį balą (užšaldytas testas).", first_line=False, space_after=4)
    add_table(
        doc,
        ["Vynas", "Balas", "n", "MAE", "Vid. f", "Poslinkis"],
        [
            ["raudonas", "3", "2", "2,10", "5,10", "+2,10"],
            ["raudonas", "5", "136", "0,33", "5,29", "+0,29"],
            ["raudonas", "6", "128", "0,46", "5,81", "−0,19"],
            ["raudonas", "8", "4", "1,59", "6,41", "−1,59"],
            ["baltas", "3", "6", "2,09", "5,09", "+2,09"],
            ["baltas", "6", "439", "0,39", "5,90", "−0,10"],
            ["baltas", "8", "35", "1,40", "6,60", "−1,40"],
            ["baltas", "9", "1", "2,02", "6,98", "−2,02"],
        ],
    )
    caption(doc, "4 lentelė. Kraštai traukiami į 5–6; vidurinės klasės MAE priimtinas.")
    add_p(doc, "5 lentelė. Didžiausios |yi − f| SVR klaidos užšaldytame teste.", first_line=False, space_after=4)
    add_table(
        doc,
        ["Vynas", "y", "f", "liekana", "Alkoholis, tūrio %", "Lakiosios rūgštys"],
        [
            ["raudonas", "4", "6,29", "−2,29", "11,0", "0,53"],
            ["raudonas", "3", "5,28", "−2,28", "9,7", "0,98"],
            ["raudonas", "8", "6,08", "+1,92", "13,4", "0,62"],
            ["baltas", "3", "6,07", "−3,07", "11,5", "0,32"],
            ["baltas", "4", "6,58", "−2,58", "12,9", "0,64"],
            ["baltas", "8", "5,54", "+2,46", "10,5", "0,15"],
        ],
    )
    caption(doc, "5 lentelė. Šaltinis: results/error_examples_{red,white}.csv.")
    add_p(
        doc,
        "Klaidos ≥ 2 balų dažniausiai kraštiniuose taikiniuose. Jautrumo bandymas (požymis keičiamas nuo −1 iki +1 "
        "mokymo st. nuok., kiti – vidurkyje): alkoholis kelia f (raudonas Δf = +0,80, baltas +0,41), "
        "lakiosios rūgštys mažina (atitinkamai −0,44 ir −0,34). Kryptys sutampa su enologine teorija ir "
        "Pirsono koreliacijomis (alkoholis +0,48 / +0,44, lakiosios rūgštys raudonam −0,39).",
    )

    add_h(doc, "7. Abliacija ir atsparumas")
    add_p(
        doc,
        "Abliacija A1 (tas pats skaidymas): tiesinio branduolio SVR MAE 0,488 (raudonas) ir 0,571 (baltas), "
        "Gauso – 0,473 ir 0,534. Netiesinis branduolys duoda visą H1 gerinimą baltam; vien ε nuostolis su tiesiniu "
        "branduoliu tiesinės regresijos nepranoksta pakankamai. Tai pagrindžia Gauso branduolio pasirinkimą.",
        space_after=4,
    )
    add_p(
        doc,
        "Grupės poslinkio / nutekėjimo bandymas (A6): atsitiktinis skaidymas į testą įleidžia 74 raudono ir 274 balto "
        "dublikatų grupes, SVR MAE krenta iki 0,429 ir 0,507 (palyginti su 0,473 ir 0,534 grupiniame skaidyme). "
        "Apie 0,04 MAE „pranašumo“ būtų nutekėjimas, ne modelio kokybė. Todėl grupinis skaidymas paliekamas privalomas. "
        "Triukšmo bandymas: prie testo požymių pridėtas N(0, 0,1·std) – raudono MAE auga 0,012 (0,473 → 0,486), "
        "balto 0,002; Acc_1,0 krenta 0,9 p. p. ir 1,6 p. p. Modelis triukšmui vidutiniškai atsparus, jautriau reaguoja "
        "mažesnė raudono imtis.",
    )

    add_h(doc, "8. Kodėl SVR paliktas pagrindiniu ir kokios ribos")
    add_p(
        doc,
        "SVR „laimi“ ne visais MAE stulpeliais: raudonam MLP 0,001 geresnis, baltam miškas 0,002 geresnis. "
        "Pagal iš anksto užfiksuotą H2 taisyklę šie skirtumai per maži keisti modeliui. SVR vis tiek pranoksta "
        "abu privalomus baseline, turi patikrinamą formulę kode, iškilų mokymą ir ε zoną, atitinkančią ±0,5 balo "
        "apvalinimą. RBF tinklas patvirtina, kad fiksuoti centrai silpnesni už atraminius vektorius. "
        "Ribos: triukšmingas taikinys be individualių degustatorių – nesumažinama paklaidos grindų nežinoma; "
        "kraštinės klasės n = keliolika, jų tikslumo interpretuoti negalima; duomenys – vienas regionas ir laikotarpis, "
        "kitam gamintojui reikės kalibracijos. Diegiant SVR naudojamas kaip antra nuomonė, ne kaip įstatyminis balas.",
    )

    add_h(doc, "9. Atkūrimas")
    add_p(
        doc,
        "Priklausomybės: Python 3.12, numpy, pandas, scikit-learn, scipy, matplotlib, joblib, pytest "
        "(requirements.txt). Duomenys nepridedami: pirma komanda pati atsisiunčia CSV ir tikrina SHA-256 "
        "(raudonas 4a402cf0…, baltas 76c3f809…). Sėklos: SPLIT_SEED = 0, DEFAULT_SEED = 0 faile winequality/config.py. "
        "Viena pagrindinio eksperimento komanda (projekto šaknyje, aktyviame .venv):",
        space_after=4,
    )
    add_p(doc, "python -m winequality.run_experiment", align="left", first_line=False, space_after=6, italic=True)
    add_p(
        doc,
        "Išvestis: results/metrics.json, results/comparison.csv, results/figures/, artifacts/*.joblib. "
        "Vienetų testai: python -m pytest tests -q (tarp jų formulės ir sklearn prognozių sutapimas). "
        "Pavienė prognozė: python -m winequality.predict --kind red --json sample.json. "
        "Užrašinės naršyklėje: python -m winequality.paleisti_uzrasines.",
    )

    add_h(doc, "10. Dirbtinio intelekto naudojimo žurnalas")
    add_p(
        doc,
        "Naudotas Cursor Grok 4.6 (2026-09-20). Svarbiausios užklausos: kolokviumo plano perdavimas ir spragų "
        "patikra; egzamino (ne tik plano) apimties patikslinimas; baseline, SVR su formulėmis ir antro metodo "
        "įgyvendinimas; vėliau – MLP ir RBF tinklo įtraukimas; Jupyter užrašinės; ši ataskaita.",
        space_after=4,
    )
    add_p(
        doc,
        "Priimti pasiūlymai: egzamino dydžio protokolas (užšaldytas 80/20, vidinis GroupKFold, ne pilnas 10×3×60 CV); "
        "SVR kaip pagrindinis, palyginimui RF, MLP ir RBF tinklas; dualinė SVR formulė kaip atskira funkcija; "
        "abliacija A1, jautrumas ir atsparumas triukšmui bei grupiniam nutekėjimui. "
        "Atmesti: CORN, SMOTE, Wilcoxon/Holm ir serviso/stebėsenos sluoksnis; hiperparametrų paieška teste; "
        "kolokviumo orientacinio SVR MAE 0,455/0,516 naudojimas kaip egzamino rezultatas.",
        space_after=4,
    )
    add_p(
        doc,
        "Aptiktos AI klaidos / nepatikrintos prielaidos. (1) Prielaida, kad sklearn SVR.dual_coef_ yra (α − α*) "
        "ir predict_from_dual gali pakeisti SVR.predict – tikrinta test_svr_formula_matches_sklearn ir "
        "eksperimento max-abs slenksčiu 10⁻⁶. (2) Prielaida, kad sklearn.utils.fixes.loguniform egzistuoja "
        "dabartinėje scikit-learn – nepasitikėta, kode scipy.stats.loguniform. (3) Pirmasis eksperimento paleidimas "
        "lūžo, nes Log1pStandardScaler keitė konstruktoriaus argumentą ir sklearn clone() to nepriėmė – pataisyta "
        "paliekant parametrus nepakeistus. (4) Kolokviumo skaičiai nenaudoti kaip galutiniai: ataskaitoje tik "
        "results/metrics.json po paleistos komandos.",
    )

    add_h(doc, "11. Išvados")
    add_p(
        doc,
        "Įgyvendinta atkuriama grandinė, du baseline ir keturi planiniai intelektualieji metodai, lyginami tuo pačiu "
        "grupiniu skaidymu. SVR dualinė formulė susieta su konkrečiomis kodo eilutėmis. H1 raudonam nepasitvirtino, "
        "baltam pasitvirtino; H2 galioja, todėl diegti paliekamas SVR. Didžiausia grėsmė išvadoms – dublikatų nutekėjimas "
        "ir subjektyvus taikinys; pirmoji suvaldyta grupėmis, antroji lieka duomenų savybė, todėl modelis tinka "
        "rūšiavimui ir pakartotinei degustacijai, ne ekspertų pakeitimui.",
    )

    add_p(doc, "Literatūra (atrinkta)", bold=True, align="left", first_line=False, space_before=8, space_after=4)
    refs = [
        "1. Cortez, P., Cerdeira, A., Almeida, F., Matos, T., Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. Decision Support Systems, 47(4), 547–553. https://doi.org/10.1016/j.dss.2009.05.016",
        "2. Kapoor, S., Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. Patterns, 4(9), 100804. https://doi.org/10.1016/j.patter.2023.100804",
        "3. Karal, Ö. (2023). Robust and optimal epsilon-insensitive kernel-based regression for general noise models. Engineering Applications of Artificial Intelligence, 120, 105841. https://doi.org/10.1016/j.engappai.2023.105841",
        "4. Wu, J., Wang, Y.-G. (2022). A working likelihood approach to support vector regression with a data-driven insensitivity parameter. International Journal of Machine Learning and Cybernetics, 14(3), 929–945. https://doi.org/10.1007/s13042-022-01672-x",
        "5. Grinsztajn, L., Oyallon, E., Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? NeurIPS 2022.",
        "6. Bodington, J. (2022). Stochastic error and biases remain in blind wine ratings. Journal of Wine Economics, 17(4), 345–351. https://doi.org/10.1017/jwe.2022.53",
    ]
    for t in refs:
        add_p(doc, t, size=11, align="left", first_line=False, space_after=3)

    doc.save(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    build()
