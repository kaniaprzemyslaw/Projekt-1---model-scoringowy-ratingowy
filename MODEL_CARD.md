# Model Card – model ratingowy PD (Projekt 1, zbiór 5)

> Wszystkie liczby pochodzą z pełnego przebiegu notebooka `projekt1_scoring_rating.ipynb` (seed 42; wersje pakietów w `requirements.txt`).

## 1. Cel i zakres

| | |
|---|---|
| Przeznaczenie | Ocena ryzyka niewypłacalności przedsiębiorstwa: prawdopodobieństwo defaultu (PD), klasa ratingowa i rekomendacja decyzji (przyjęcie / odrzucenie) na podstawie danych finansowych. |
| Użytkownicy | Analityk kredytowy / komitet kredytowy – model wspiera decyzję, nie zastępuje oceny eksperckiej. |
| Zdarzenie | `default` = 1 – zdarzenie ryzyka w horyzoncie przyjętym w opisie zmiennych. Dane nie zawierają dat, więc horyzontu nie da się zweryfikować w danych. |
| Poziom PD | PD skorygowane do długoterminowej tendencji centralnej 4% (założenie biznesowe projektu). |
| Populacja | Przedsiębiorstwa ze sprawozdaniami finansowymi o strukturze zbioru 5 (jedna forma prawna, 76 działów PKD). |
| Poza zakresem | Firmy bez sprawozdań finansowych, inne formy prawne, automatyczne decyzje bez nadzoru człowieka, wycena ryzyka (cennik) bez dodatkowej kalibracji, wnioskowanie przyczynowe („zmiana wskaźnika X spowoduje spadek ryzyka”). |

## 2. Dane

| | |
|---|---|
| Źródło | `zbiór_5.csv` + dokument opisu zmiennych; dane przekrojowe, bez dat. |
| Liczebność | 3000 firm, 178 defaultów (DR = 5,93%). |
| Zmienne | 139 wskaźników `wsk_*`, 78 pozycji kwotowych, 5 kolumn kodowych (PKD, forma własności, forma prawna, schematy sprawozdań). |
| Jakość danych | Brak duplikatów wierszy i powtórzonych odcisków firm. `±inf` we wskaźnikach = dzielenie przez zero (np. brak zobowiązań krótkoterminowych) – obsługiwane jako osobne wartości, nie jako braki. Kontrole tożsamości bilansowych, wartości niemożliwych i ogonów – rozdział 6 notebooka. |
| Wykluczone cechy | `szczegolnaFormaPrawna_Symbol` (jeden poziom); 7 cech z ≥ 50% braków; cechy quasi-stałe (≥ 99% jednej wartości); 9 duplikatów kolumn. |
| Cechy do weryfikacji | `wsk_index_aktywne`, `wsk_index_skr_zaw`, `wsk_index_nowe` – wskaźniki sektorowe przypisane do kodu PKD, najsilniejsze cechy obu modeli. Okresu pomiaru nie da się ustalić z danych; muszą pochodzić sprzed daty oceny (rozdział 5.4). |

## 3. Podział i walidacja

* Podział warstwowy po `default` (seed 42): train 1800 (107 defaultów), validation 600 (35), test 600 (36). Brak powtórzeń podmiotów → podział bez grup (kod obsługuje `StratifiedGroupKFold`).
* Stratified 5-fold CV na train: wybór cech, binningu i hiperparametrów; w każdym foldzie całe przygotowanie cech jest dopasowywane od nowa.
* Validation: kalibracja, korekta do 4%, próg decyzyjny, granice klas ratingowych.
* Test: wyłącznie ocena końcowa.
* Kontrola przecieku: asercje rozłączności zbiorów i foldów, test permutacji etykiet (AUC po permutacji ≈ 0,52), test wrażliwości na `wsk_index_*`.
* Diagram schematu walidacji: `artifacts/fig/schemat_walidacji.png`; podziały i foldy: `artifacts/splits.json`.

## 4. Modele

| | Karta (model interpretowalny) | LightGBM (black-box) |
|---|---|---|
| Wejście | 8 cech WoE (binning monotoniczny, IV ≥ 0,02, \|r\| < 0,70, limit 8 cech) | 212 kolumn: wskaźniki z cappingiem p1–p99, pozycje kwotowe po `signed_log1p`, flagi braków / `inf`, one-hot kategorii |
| Algorytm | regresja logistyczna; selekcja L1 → VIF ≤ 5 → L2 z kontrolą znaku (β < 0) | gradient boosting drzew (LightGBM 4.7) |
| Tuning | siatka C (L1, L2) w 5-fold CV, reguła 1-SE: `C_l1` = 0,1, `C_final` = 0,1 | Random Search, 40 konfiguracji w 5-fold CV, kryterium log-loss, reguła 1-SE: 134 drzewa, `learning_rate` 0,054, `num_leaves` 4, `max_depth` 5, `min_child_samples` 41, `subsample` 0,94, `colsample_bytree` 0,60, `reg_lambda` 41,9, `reg_alpha` 0,026 |
| Złożoność | 8 cech + wyraz wolny, wszystkie zależności monotoniczne | 134 drzewa × maks. 4 liście, 212 kolumn wejściowych; minimum log-lossu w CV przy 134 drzewach |
| Ograniczenia | monotoniczność z binningu, znak β | regularyzacja, bez ograniczeń monotoniczności |
| Przeuczenie (AUC train – OOF) | 0,790 – 0,715 = 0,075 | 0,868 – 0,722 = 0,146 |

**Model rekomendowany: karta WoE + regresja logistyczna.** LightGBM nie daje istotnie lepszego rozróżniania: ΔAUC (LightGBM − karta) na validation = −0,021, 95% przedział bootstrap [−0,052; +0,013]; na teście AUC 0,832 vs 0,828. Karta ma 8 cech zamiast 212, dokładne wyjaśnienia (dekompozycja logitu), mniejsze przeuczenie i lepszy calibration slope na teście (1,25 vs 1,48). LightGBM pozostaje modelem porównawczym (challenger).

## 5. Jakość

| Metryka | Karta – CV na train (śr. ± std) | Karta – test | LightGBM – CV | LightGBM – test |
|---|---|---|---|---|
| AUC | 0,716 ± 0,065 | 0,828 | 0,724 ± 0,068 | 0,832 |
| Gini | 0,431 | 0,656 | 0,448 | 0,664 |
| PR-AUC | 0,216 ± 0,050 | 0,316 | 0,241 ± 0,064 | 0,277 |
| KS | 0,379 ± 0,090 | 0,556 | 0,394 ± 0,115 | 0,592 |
| Log-loss | 0,207 ± 0,013 | 0,185 | 0,206 ± 0,010 | 0,189 |
| Brier | 0,053 ± 0,002 | 0,050 | 0,053 ± 0,002 | 0,051 |

Metryki testu dla PD przed korektą do 4% (`artifacts/metrics_test.csv`). Zbiór testowy ma 36 defaultów, więc AUC na teście ma duży błąd losowy; AUC w poszczególnych foldach CV wahało się od 0,61 do 0,78 (karta) i od 0,61 do 0,77 (LightGBM).

Bez `wsk_index_*` AUC w CV spada o 0,063 (karta) i 0,051 (LightGBM) (rozdziały 9.3 i 16.1).

## 6. Kalibracja i dostosowanie do PD = 4%

**Kalibracja statystyczna** (porównanie metod: 20 × 5-fold CV wewnątrz validation, rozdział 17.2):

| Metoda | Karta: log-loss | Karta: Brier | Karta: slope | LightGBM: log-loss | LightGBM: Brier | LightGBM: slope |
|---|---|---|---|---|---|---|
| brak | 0,1863 | 0,0499 | 1,11 | 0,1942 | 0,0514 | 1,24 |
| intercept | 0,1864 | 0,0500 | 1,10 | 0,1942 | 0,0513 | 1,23 |
| Platt | 0,1875 | 0,0504 | 0,97 | 0,1946 | 0,0518 | 0,96 |
| izotoniczna | 0,1930 | 0,0505 | 0,67 | 0,2041 | 0,0533 | 0,59 |
| beta | 0,1891 | 0,0512 | 0,93 | 0,1963 | 0,0526 | 0,90 |

Dla obu modeli wybrano wariant bez dodatkowej kalibracji kształtu: żadna metoda nie obniżyła log-lossu o więcej niż 0,5%. Platt koryguje slope do ≈ 0,96, ale nie poprawia log-lossu ani Briera. Regresja izotoniczna (35 defaultów w validation) jest przeuczona: najwyższy log-loss i slope 0,67 / 0,59.

**Korekta do 4%:** przesunięcie logitu o δ wyznaczone na validation tak, by średnie PD = 4%.

| | Karta | LightGBM |
|---|---|---|
| DR train / validation / test | 5,94% / 5,83% / 6,00% | jw. |
| Średnie PD przed korektą (validation) | 5,87% | 5,40% |
| δ (logit), exp(δ) | −0,436; 0,647 | −0,330; 0,719 |
| Średnie PD przed / po korekcie – test | 6,24% / 4,26% | 6,07% / 4,50% |
| CITL na teście przed / po korekcie | −0,046 / 0,390 | −0,012 / 0,317 |
| CITL na teście przeważonym do DR 4%, przed / po korekcie | −0,465 / −0,029 | −0,430 / −0,101 |
| Calibration slope (test) | 1,25 | 1,48 |
| Brier / log-loss na teście przed → po korekcie | 0,0496 → 0,0506 / 0,1845 → 0,1880 | 0,0509 → 0,0519 / 0,1887 → 0,1912 |

Korekta do 4% nie zmienia rankingu (AUC bez zmian) i nie poprawia kalibracji statystycznej: względem częstości defaultów w próbie (6%) PD po korekcie są zaniżone (CITL 0,39 i 0,32), a Brier i log-loss na teście nieznacznie rosną. Korekta jest właściwa tylko przy założeniu, że portfel docelowy ma DR 4%, a rozkłady cech w klasach są takie jak w próbie – na teście przeważonym do 4% CITL spada z −0,47 do −0,03 (karta) i z −0,43 do −0,10 (LightGBM). Slope > 1 na teście oznacza, że PD obu modeli są zbyt skupione wokół średniej; na validation (35 defaultów) żadna metoda tego nie poprawiała. Krzywe reliability: `artifacts/fig/reliability_*.png`.

## 7. Skala ratingowa i próg decyzyjny

**Skala** (PD po korekcie do 4%; skala nadrzędna 0,5% / 1% / 2% / 4% / 8% / 15%; klasy łączone na validation do monotonicznej częstości defaultów i min. 15 obserwacji):

| Klasa (karta) | AA | A | BBB | BB | B | CCC |
|---|---|---|---|---|---|---|
| Granice PD | < 1% | 1–2% | 2–4% | 4–8% | 8–15% | ≥ 15% |
| Udział – test | 16,8% | 26,5% | 24,0% | 17,8% | 10,7% | 4,2% |
| Średnie PD – test | 0,7% | 1,5% | 2,9% | 5,5% | 10,6% | 22,8% |
| Obs. DR – test | 1,0% | 0,0% | 4,2% | 9,3% | 17,2% | 32,0% |

Średnie PD rośnie monotonicznie; obserwowana częstość na teście rośnie od klasy A (w AA 1 default na 101 firm, w A – 0 na 159; przedziały Wilsona się pokrywają). Klasa D (stwierdzony default) jest poza skalą modelu.

| Klasa (LightGBM) | A | BBB | BB | B | CCC |
|---|---|---|---|---|---|
| Granice PD | < 2% | 2–4% | 4–8% | 8–15% | ≥ 15% |
| Udział – test | 32,0% | 32,0% | 20,5% | 12,0% | 3,5% |
| Średnie PD – test | 1,4% | 2,9% | 5,7% | 10,9% | 19,0% |
| Obs. DR – test | 0,5% | 2,1% | 8,1% | 18,1% | 38,1% |

Skala LightGBM ma 5 klas (scalenie na validation); obserwowana częstość rośnie monotonicznie na validation i na teście. Te same klasy w obu modelach ma 46,7% firm testowych, różnica ≤ 1 stopnia – 98,5%.

**Próg:** marża 6% ekspozycji, LGD 45%; próg teoretyczny MARGIN / (MARGIN + LGD) = 11,8%. Progi wybrane na validation (maksimum zysku): karta – PD 14,9% w skali próby (10,1% w skali 4%), LightGBM – 6,9% (5,1%).

| Test | Odrzucenia | Wykryte defaulty | DR przyjętych | Zysk / klienta |
|---|---|---|---|---|
| karta – próg* z validation | 9,7% | 11 z 36 | 4,6% | 0,0330 |
| karta – odrzucenie ~15% (próg z validation) | 18,3% | 23 z 36 | 2,7% | 0,0379 |
| LightGBM – próg* z validation | 28,0% | 26 z 36 | 2,3% | 0,0347 |
| LightGBM – odrzucenie ~10% (próg z validation) | 14,7% | 19 z 36 | 3,3% | 0,0367 |
| przyjęcie wszystkich | 0% | 0 | 6,0% | 0,0294 |

Każdy wariant z progiem daje wyższy zysk niż przyjęcie wszystkich. Progi obu modeli różnią się (odrzucenia na validation 9,3% vs 22,8%), a na teście najwyższy zysk dają inne progi niż wybrane na validation – przy 35 defaultach w validation wybór progu jest obarczony dużą niepewnością. Zgodność decyzji karty i LightGBM na teście: 81,0% (kappa Cohena 0,41); LightGBM odrzuca 168 firm, karta 58, w tym 56 wspólnych.

## 8. Wyjaśnienia

**Globalnie – karta** (udział w ważności |β|·std WoE): `wsk_index_nowe` 26%, `wsk_index_skr_zaw` 15%, `wsk_struktury_kapitalu` 12%, `wsk_poziom_kapitalu_obrotowego_netto` 11%, `wsk_stopa_zysku_sprzedaz` 10%, `wsk_ebit` 10%, `wsk_efekt_dzwigni_fin_1` 9%, `wsk_ebitda_3` 6%. Wszystkie β < 0, maks. VIF 2,04. Kierunki zgodne z oczekiwaniem ekonomicznym dla 5 cech (EBIT, EBITDA, marża, kapitał obrotowy netto, wykreślenia firm w branży); dla 3 cech brak hipotezy a priori. Na validation kierunek zależności jest taki sam jak na train dla 8 z 8 cech.

**Globalnie – LightGBM (SHAP, średnie |SHAP| na validation):** `wsk_index_nowe` 0,36, `slog_Kapital_wlasny` 0,14, `wsk_struktury_kapitalu` 0,11, `wsk_index_skr_zaw` 0,11, `wsk_udzial_zapasy_zobowiazania` 0,09. Zgodność z kartą: Spearman rang 8 cech karty = 0,88, Jaccard top-5 = 0,43 (wspólne: oba wskaźniki sektorowe i `wsk_struktury_kapitalu`). `wsk_ebit` ma w LightGBM ważność ≈ 0 (ranga 120 z 212 kolumn).

**Lokalnie:** karta – dokładna dekompozycja logitu na wkłady cech (rozdział 13.3); LightGBM – SHAP waterfall i LIME dla 4 przypadków: wykryty default, niedefault o najniższym PD, przypadek graniczny (PD ≈ 85. percentyl), niewykryty default (rozdział 14.3). Dla wykrytego defaultu (PD 31,7% karta / 30,4% LightGBM) wszystkie trzy metody wskazują jako główny czynnik `wsk_index_nowe`; ustawienie trzech najsilniej działających wskaźników na medianę obniża PD do 5,1% (karta) i 7,8% (LightGBM) (what-if, rozdział 14.4).

**Analiza kontrfaktyczna (rozdział 18.4):** firma z validation (niedefault) w klasie BB karty, PD 5,5%. Najmniejsza zmiana spełniająca ograniczenia: wzrost `wsk_poziom_kapitalu_obrotowego_netto` z 29 199 do 211 313 (jednostki jak w danych; z 30. do 55. percentyla rozkładu train) obniża PD karty do 3,9% → klasa BBB. W LightGBM ta sama zmiana obniża PD tylko z 9,1% do 8,4% (klasa B bez zmian), a wskaźnika tego nie ma wśród 5 głównych czynników zwiększających ryzyko tej firmy według SHAP i LIME (według karty jest drugim). Wynik pokazuje zachowanie modelu, nie skutek przyczynowy.

**Wierność i stabilność wyjaśnień (rozdział 15):** dekompozycja karty jest dokładna (asercja). LightGBM:

| Eksperyment | Wynik |
|---|---|
| SHAP – addytywność (maks. błąd, log-odds) | 7·10⁻¹⁵ |
| LIME – lokalne R² (30 obserwacji) | mediana 0,20; 90% obserwacji z R² < 0,5 |
| LIME – \|PD lokalnego modelu − PD LightGBM\| | mediana 0,007 |
| LIME – 10 ziaren, 4 obserwacje | Spearman wag 0,85; Jaccard top-5 0,80; zgodność znaku top-3 100% |
| SHAP – bootstrap obserwacji validation | Spearman 1,00; Jaccard top-10 0,99 |
| SHAP – LightGBM uczony od nowa w 5 foldach | Spearman 0,55; Jaccard top-10 0,50 |
| Karta uczona od nowa w 5 foldach | Spearman 0,34; Jaccard top-10 0,65 |

Niskie R² LIME oznacza, że liniowe przybliżenie słabo opisuje LightGBM w otoczeniu obserwacji – wagi LIME należy traktować ostrożnie. Ranking SHAP jest stabilny względem próby objaśnianej, a zmienia się przy ponownym uczeniu modelu – niestabilność wynika z modelu, nie z metody wyjaśniającej. Wśród 10 najważniejszych cech SHAP są pary wskaźników o |ρ| > 0,8 (rozdział 15.3), między które model może różnie rozdzielać wpływ. Stabilność karty w foldach: znaki β stałe; Jaccard zbiorów cech 0,43 – we wszystkich foldach wybierane są tylko oba wskaźniki sektorowe.

**Ograniczenia wyjaśnień:** skorelowane wskaźniki dzielą między siebie wpływ w SHAP i LIME (rozdział 15.3); LIME zależy od losowania i dyskretyzacji; żadna z metod nie dowodzi zależności przyczynowej.

## 9. Audyt podgrup (test, karta)

| Grupa | n | Defaulty | Obs. DR | AUC karta | AUC LightGBM | Śr. PD karta | Śr. PD LightGBM |
|---|---|---|---|---|---|---|---|
| Aktywa – małe | 157 | 14 | 8,9% | 0,76 | 0,78 | 8,1% | 8,3% |
| Aktywa – średnie-niższe | 169 | 9 | 5,3% | 0,73 | 0,73 | 6,2% | 6,8% |
| Aktywa – średnie-wyższe | 130 | 8 | 6,2% | 0,91 | 0,92 | 5,0% | 4,6% |
| Aktywa – duże | 144 | 5 | 3,5% | 0,89 | 0,92 | 5,4% | 4,2% |
| Forma własności 214 | 397 | 18 | 4,5% | 0,80 | 0,80 | 5,9% | 5,9% |
| Forma własności 215 | 65 | 7 | 10,8% | 0,85 | 0,80 | 8,1% | 7,1% |
| Forma własności 216 | 52 | 6 | 11,5% | 0,95 | 0,97 | 7,7% | 6,3% |

Śr. PD – po kalibracji, przed korektą do 4%. Grupy mają po 5–18 defaultów, więc różnice AUC między nimi są orientacyjne. Oba modele mają podobne AUC w grupach. Karta zawyża PD dużych firm (5,4% vs DR 3,5%); oba modele zaniżają PD form własności 215 i 216 (DR ok. 11%, PD 6–8%). Pełna tabela: `artifacts/subgroup_audit_test.csv`.

## 10. Ograniczenia i ryzyka

* Mała liczba defaultów (107 / 35 / 36): szerokie przedziały ufności metryk, ograniczona liczba cech karty (EPV ≈ 12), niestabilna regresja izotoniczna.
* Dane przekrojowe bez dat: brak walidacji out-of-time; nie wiadomo, czy rating jest PIT czy TTC.
* Silna zależność od wskaźników sektorowych `wsk_index_*` o nieustalonym okresie pomiaru – ryzyko przecieku, jeśli pochodzą z okresu po dacie oceny.
* PD 4% to założenie biznesowe; przy innym DR portfela PD będą systematycznie zaniżone lub zawyżone.
* Parametry funkcji zysku (marża 6%, LGD 45%) są przykładowe – próg należy wyznaczyć ponownie dla rzeczywistej ekonomii produktu.
* Zmienne o potencjalnie wrażliwym charakterze: forma własności i branża – różnice między grupami wymagają monitorowania.

## 11. Plan monitoringu

| Obszar | Miara | Częstotliwość | Próg alarmowy |
|---|---|---|---|
| Stabilność populacji | PSI rozkładu PD, CSI 8 cech karty | kwartalnie | PSI / CSI > 0,10 – obserwacja, > 0,25 – przegląd modelu |
| Dyskryminacja | AUC, KS na nowych defaultach | rocznie / po ≥ 30 defaultach | spadek AUC > 0,05 względem walidacji |
| Kalibracja | średnie PD vs DR, Brier, ECE, test dwumianowy w klasach | rocznie | p < 0,05 w ≥ 2 klasach lub średnie PD poza przedziałem ufności DR |
| Skala | udział klas, monotoniczność DR w klasach | kwartalnie | koncentracja > 40% w jednej klasie, odwrócenie DR w sąsiednich klasach |
| Wyjaśnienia | ranking ważności (SHAP / \|β\|·std), Jaccard top-5 względem wdrożenia | półrocznie | Jaccard < 0,6 |
| Dane wejściowe | udział braków i `±inf`, nowe poziomy PKD / formy własności | przy każdym zasileniu | wzrost udziału braków > 5 p.p. |
| Decyzje | odsetek odrzuceń, DR przyjętych | kwartalnie | odchylenie > 50% od wartości z walidacji |

## 12. Replikowalność

Seed 42; podziały i foldy w `artifacts/splits.json`; dopasowane transformery, modele i kalibratory w `artifacts/*.joblib`; konfiguracja i suma kontrolna danych w `artifacts/feature_manifest.json`; wersje pakietów w `requirements.txt`. Instrukcja uruchomienia: `README.md`.

## 13. Użycie narzędzi AI

Zgodnie z regulaminem w projekcie korzystaliśmy pomocniczo z asystenta AI Claude (Anthropic). Zakres pomocy:

| Obszar | Pomoc AI |
|---|---|
| Weryfikacja zgodności z wymaganiami | sprawdzenie notebooka względem punktów 3.1–3.6 polecenia, wskazanie braków i niespójności (np. konwencja WoE i oczekiwany znak β), kontrola odczytu zapisanych artefaktów |
| Asercje i kontrole poprawności | propozycje asercji, m.in. rozłączność zbiorów train / validation / test, poprawność foldów CV, brak NaN / inf po transformacji |
| Testy | identyfikacja cech potrzebnych do stworzenia odcisków unikalnych obserwacji, pomoc przy testach przecieku danych i stabilności, m.in. test permutacji etykiet i test wrażliwości na wsk_index_* |
| Kod | poprawki błędów (m.in. indeksy foldów lokalne vs globalne, obsługa braków w AUC), połączenie notebooków przygotowania danych i modeli w jeden pipeline (w trakcie projektu działaliśmy na rozdzielonych notebookach)|
| Uporządkowanie i teksty | pomoc w uporządkowaniu struktury rozdziałów, redakcji opisów komórek oraz szkicu README.md i niniejszej karty