# Architektur & Design-Entscheidungen

## Fachlicher Ablauf

1. Historische Bestellungen enthalten Kaufzeitpunkt und später bekanntes Lieferergebnis.
2. Der Feature-Build setzt `label_available_at` auf Kaufzeitpunkt plus tatsächliche Lieferdauer.
3. Historische Raten verwenden nur Ergebnisse, die vor dem Kaufzeitpunkt der jeweiligen Bestellung bekannt waren.
4. Der zeitbasierte Trainingssplit verwendet ein Embargo, damit keine nach Testbeginn bekannten Trainingslabels einfließen.
5. Neue Bestellungen werden ohne Lieferergebnis erzeugt. Historie und neue Bestellungen werden gemeinsam verarbeitet, neue Zeilen steuern aber keine Outcomes bei.
6. Scoring speichert Risikowahrscheinlichkeit, Risikoklasse und Modell-Run-ID.
7. Erst nach Bekanntwerden der Labels bewertet `evaluate_predictions.py` die tatsächliche Modellleistung.

## Warum kein Feature-Store-Framework?

Es gibt ein Modell und einen Batch-Anwendungsfall ohne Millisekunden-Latenzanforderung. Die explizite Pandas-Implementierung macht Point-in-Time-Logik und Leakage-Tests direkt nachvollziehbar. Ein Feature Store wäre bei mehreren Modellen, gemeinsam genutzten Features oder Online-/Offline-Serving sinnvoller.

## Warum GradientBoostingClassifier?

Die Modellwahl ist nicht der Hauptgegenstand. scikit-learn bietet ein gut nachvollziehbares tabellarisches Modell mit wenig Infrastruktur. Der Fokus liegt auf Datenkorrektheit, reproduzierbarem Training, Evaluation und Monitoring.

## Warum JSON-Run-Metadaten statt MLflow?

Ein JSON-Datensatz pro Trainingslauf enthält Modellversion, Features, Seed, zeitliche Split-Grenze, Metriken, Baseline und Kalibrierung. Für ein einzelnes Portfolio-Projekt reicht das aus und benötigt keinen Tracking-Server.

## Drift ist nicht Performance

Der KS-Test vergleicht zeitlich getrennte Stichproben. Er kann Verteilungsänderungen anzeigen, aber nicht beweisen, dass die Vorhersagen schlechter wurden. Dafür sind später verfügbare Labels erforderlich. Darum ersetzt der Drift-Check nicht automatisch das aktive Modell; ein Kandidat muss gegen das aktuelle Modell evaluiert und bewusst freigegeben werden.

## Point-in-Time-Correctness

Eine Bestellung von gestern kann heute noch unterwegs sein. Obwohl ihr Kaufzeitpunkt vor dem aktuellen Kauf liegt, ist ihr Lieferergebnis noch unbekannt. Daher reicht eine Sortierung nach Kaufzeitpunkt nicht. Die Feature-Berechnung aggregiert anhand von `label_available_at`, und die Unit-Tests prüfen unter anderem offene Bestellungen, Zeitfenster und Veränderungen zukünftiger Zeilen.

## Grenzen

Die Daten sind synthetisch und enthalten ein künstliches Signal für Verspätungen. Ergebnisse dürfen nicht als reale Lieferprognoseleistung interpretiert werden. Die aktuelle Drift-Prüfung nutzt historische Zeitfenster als Simulation und ist kein Ersatz für Monitoring mit produktivem Traffic.
