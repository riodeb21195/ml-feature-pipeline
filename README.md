# ML Feature Pipeline – Delivery Risk

Ein reproduzierbares ML-Engineering-Portfolio-Projekt: Es schätzt für neue Bestellungen das Risiko einer verspäteten Lieferung. Der technische Schwerpunkt ist **Point-in-Time-Correctness** – Features dürfen nur Informationen verwenden, die zum Kaufzeitpunkt bekannt waren.

> **Datengrenze:** Alle Bestellungen werden synthetisch erzeugt. Das Projekt demonstriert Engineering-Prinzipien, keine nachgewiesene Vorhersagequalität für reale Lieferketten.

## Architektur und Ablauf

```text
Historische Bestellungen (Labels bekannt)
       │
       ├── Data Quality Checks
       │
       └── Point-in-Time Feature Build ──> Training + zeitbasierte Evaluation
                                               │
                                               └── Modell + Run-Metadaten

Neue Bestellungen (ohne Lieferergebnis)
       │
       └── gemeinsam mit Historie Features berechnen
                    │
                    └── Batch Scoring ──> Risikoscore + Risikoklasse
                                               │
Später bekannte Lieferergebnisse ──────────────┴──> Performance Evaluation

Historische vs. aktuelle Eingabeverteilung ──> Drift Check
```

## Schnellstart

Getestete Umgebung: Python 3.12. Die Abhängigkeiten sind in `requirements.txt` festgelegt.

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# 1. Historische Daten erzeugen und validieren
python data_generation/generate_raw_orders.py
python monitoring/check_data_quality.py

# 2. Features berechnen und Modell trainieren
python features/build_features.py
python training/train_model.py

# 3. Neue, ungelabelte Bestellungen erzeugen und scoren
python data_generation/generate_new_orders.py --n-orders 200 --seed 2026
python scoring/predict.py

# 4. Eingabedaten auf Drift prüfen und Tests ausführen
python monitoring/check_drift.py
pytest tests/unit -v
```

Die wichtigsten Artefakte entstehen lokal und werden nicht ins Git eingecheckt:

- `data_generation/raw_orders.parquet`: historische Bestellungen mit Ergebnis
- `data_generation/features.parquet`: historische Features und Labels
- `data_generation/new_orders.parquet`: neue Bestellungen **ohne** tatsächliche Lieferdauer und Zielvariable
- `data_generation/scored_orders.parquet`: Scores für neue Bestellungen
- `training/runs/run_*.json`: Metriken und Trainingsmetadaten
- `models/latest_model.joblib`: zuletzt trainiertes Modell
- `monitoring/drift_history.jsonl`: Verlauf der Drift-Prüfungen

## Evaluation nach Bekanntwerden der Labels

Scoring und Evaluation sind getrennt. Wenn Lieferergebnisse für zuvor gescorte Bestellungen vorliegen, speichere eine Parquet-Datei mit mindestens `order_id` und `is_late_delivery` als `data_generation/resolved_orders.parquet`. Die Datei muss eindeutige `order_id`-Werte enthalten.

```bash
python monitoring/evaluate_predictions.py
```

Das Skript berechnet Brier Score, ROC-AUC (sofern beide Klassen vorliegen), Average Precision, Precision und Recall am konfigurierten Schwellenwert. Es wertet nur übereinstimmende Vorhersagen und Labels aus. Die Beispielpipeline simuliert keine tatsächlichen zukünftigen Zustellungen automatisch.

## Was die Pipeline absichert

- **Point-in-Time-Features:** Nur Outcomes mit `label_available_at < order_purchase_timestamp` fließen in historische Raten ein.
- **Noch offene Bestellungen:** Neue Bestellungen ohne Lieferergebnis werden beim Feature-Build unterstützt und können keine Labels beisteuern.
- **Zeitbasierter Split + Embargo:** Trainingslabels müssen vor Beginn des Testzeitraums bekannt gewesen sein.
- **Baseline und Kalibrierung:** Das Modell wird mit einer Dummy-Baseline verglichen; Wahrscheinlichkeiten werden zusätzlich geprüft.
- **Data Quality:** Schema, eindeutige IDs, Zeitstempel, positive Lieferzeiten und binäres Target werden validiert.
- **Drift vs. Performance:** KS-Drift ist ein Warnsignal, kein Beweis für sinkende Modellqualität. Automatisches Überschreiben des aktiven Modells ist deshalb deaktiviert.
- **CI:** GitHub Actions führt Linting, Unit-Tests und einen End-to-End-Smoke-Run aus.

## Projektstruktur

```text
ml-feature-pipeline/
├── data_generation/
│   ├── generate_raw_orders.py
│   └── generate_new_orders.py
├── features/
│   ├── build_features.py
│   └── feature_registry.yml
├── training/
│   └── train_model.py
├── scoring/
│   └── predict.py
├── monitoring/
│   ├── check_data_quality.py
│   ├── check_drift.py
│   └── evaluate_predictions.py
├── tests/unit/
├── .github/workflows/ci.yml
├── ARCHITECTURE.md
└── requirements.txt
```

## Grenzen und nächste Ausbaustufen

- Die Daten sind synthetisch und enthalten ein künstlich eingebautes Signal.
- Drift wird über eine zeitliche Aufteilung der verfügbaren Historie simuliert, nicht aus einem echten produktiven Traffic-Stream gemessen.
- Ein Modell sollte erst nach Vergleich auf einem zeitlich passenden, gelabelten Evaluationssatz freigegeben werden.
- Eine API, Feature-Store-Technologie oder Streaming-Plattform ist für diesen Batch-Anwendungsfall bewusst nicht erforderlich.
