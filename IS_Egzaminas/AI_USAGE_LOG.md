# AI naudojimo žurnalas

## 2026-10-02. Sprendimo programavimas

Užklausa: šiame kataloge suprogramuoti sprendimą pagal atnaujintą įgyvendinimo planą.

Gautas pasiūlymas: Python paketas su XGBOrdinal, CatBoost, CORN, SVR, vidurkio ir tiesinės regresijos baseline, grupiniu skaidymu, train-only Optuna paieška ir iš anksto užrašytomis H1–H4 taisyklėmis.

Priimta: modulių struktūra, formulės ir viena paleidimo komanda `python -m winequality.run_experiment --config configs/main.yaml`.

Atmesta pirmoji XGBOrdinal versija su `scale_pos_weight`. Ji sumažino P(y > 3 | x) ir nutempė tikėtiną balą. Formulė reikalauja nesvertos slenksčio tikimybės, todėl svoris pašalintas. Patikra: po pataisos raudono vyno testo MAE su fiksuotais medžiais buvo 0,480, slenksčių vidurkiai [0,996; 0,967; 0,550; 0,143; 0,006; 0,000].

Atmesta pirmoji CORN versija, kurioje viena epocha buvo vienas visos aibės žingsnis. 20 epochų modelis liko prie inicializacijos (vidutinė prognozė 4,11). Pakeista į mini-batch. Patikra: po 20 epochų MAE 0,508, tikimybių masė ant 5 ir 6 balų.

Patikros įrodymas: `python -m pytest` — 11 testų, visi praėjo. `configs/smoke.yaml` paleidimas baigėsi per 16 s ir sukūrė `results/smoke/metrics.json`. Vieno bandymo XGBOrdinal neįvykdė H1 (MAE pagerėjo mažiau nei 5 %, κw prieaugis mažesnis nei 0,05), todėl taisyklė paliko tiesinę regresiją. Tai nėra pilno biudžeto rezultatas.

## 2026-10-02. Pagerinimai ir ataskaita

Užklausa: įgyvendinti pagerinimus, paleisti modelius, atnaujinti palyginimą ir parengti galutinę ataskaitą.

Priimta: log1p, trys santykiai, dublikatų ir kraštinių balų svoriai, bendras Frank–Hall CatBoost ir LightGBM, monotoniniai apribojimai. Paleista `configs/pagerinta.yaml`.

Atmesta: Python asimetrinis CatBoost nuostolis. Biblioteka jo neperkėlė į C++ skaičiavimą, paieška strigo. Vietoj jo liko įprasta MAE ir imties svoriai. Patikra: paleidimas įrašė `results/pagerinta/comparison.csv`. MAE liko didesnė už ankstesnio CatBoost (raudonam 0,451, baltam 0,515). κw išaugo. Prielaida, kad pakeitimai sumažins MAE, nepasitvirtino.

Patikrinta prielaida apie dublikatų nutekėjimą: grupių ir identiškų požymių eilučių sankirta tarp mokymo ir testo yra 0. Vidurkio ir tiesinės MAE perskaičiuotos ir sutapo su byla. CatBoost ir XGBOrdinal, pertraukti tais pačiais parametrais, sutapo su įrašyta testo MAE iki paskutinio skaitmens.
