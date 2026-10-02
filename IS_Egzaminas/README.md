# Vyno kokybės prognozė

Python sprendimas pagal atnaujintą įgyvendinimo planą. Raudonas ir baltas Vinho Verde vynas modeliuojami atskirai. Pagrindinis metodas yra XGBOrdinal. Palyginimui naudojami CatBoost, CORN ir SVR. Baseline yra mokymo vidurkis ir tiesinė regresija.

Duomenys: UCI Wine Quality, Cortez ir kt. 2009, DOI 10.24432/C56S3T. Failų SHA-256 tikrinamas atsisiuntimo metu.

## Aplinka

Python 3.11 arba naujesnė. Sėklos: skaidymo ir modelių sėkla `0`. Papildomai užrašytos sėklos `0, 1, 2`.

```text
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

`torch` CPU ratui, jei reikia atskiros nuorodos:

```text
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Viena paleidimo komanda

Iš šio katalogo, net jei `data/raw` tuščias:

```text
python -m winequality.run_experiment --config configs/main.yaml
```

Komanda atsisiunčia CSV, užšaldo 80/20 grupinį skaidymą, hiperparametrus renka tik mokymo bloke ir testą vertina vieną kartą. Plano biudžetas: XGBOrdinal 50, CatBoost 50, CORN 30, SVR 40 bandymai, kiekvienas per 5 vidines klostes, abiem vyno tipams.

Trumpa patikra, kurios skaičiai nėra galutinis tyrimas:

```text
python -m winequality.run_experiment --config configs/smoke.yaml
```

## Prognozė

Po pilno eksperimento:

```text
python -m winequality.predict --kind red --json sample.json
```

Išvestis: `score_cont`, `score_int`, klasė (`žema` / `vidutinė` / `aukšta`), balų tikimybės, jei modelis tikimybinis, ir perspėjimai `extrapolation`, `missing_input`, `low_confidence`, `disagreement`, `extreme`.

Trūkstama įvestis pakeičiama mokymo mediana ir pažymima `missing_input`. Mokymo rinkinyje trūkstamų langelių nėra, todėl trūkstamumo indikatorius pagrindiniame modelyje nemokomas: iš visų nulių sudarytas požymis nesuteiktų skaidinio.

## Katalogai

- `winequality/` — modeliai, metrikos, paieška, prognozė
- `configs/main.yaml` — plano biudžetas
- `data/splits/` — užšaldyti indeksai
- `results/` — `metrics.json`, `comparison.csv`, paveikslai, HPO žurnalai
- `tests/` — formulių ir nutekėjimo testai

Sprendimo taisyklė užrašyta iš anksto: jei H1 nepasitvirtina, lieka tiesinė regresija; jei H1 pasitvirtina, bet H2 ne, praktinis modelis yra CatBoost; kitu atveju lieka XGBOrdinal.

## Testai

```text
python -m pytest
```
