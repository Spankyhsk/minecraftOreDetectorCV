# Minecraft Ore Detector CV - Handoff

Stand: 2026-06-14

Dieses Dokument beschreibt den aktuellen Arbeitsstand fuer die Uebergabe an den naechsten Bearbeiter. Das Projekt erkennt Minecraft-Erze in Screenshots mit klassischer Bildverarbeitung. Machine Learning wird nicht verwendet und soll auch nicht eingebaut werden.

## Kurzfassung des aktuellen Stands

- Ziel: Erze in Minecraft-Screenshots erkennen und als Bounding Boxes ausgeben.
- Methoden: klassische CV mit Preprocessing, HSV-Farbmasken, Kantenmasken, Konturen, Template Matching und heuristischen Plausibilitaetsfiltern.
- Aktueller Fokus: Recall fuer sichtbare Erze in Hoehlen verbessern, ohne False Positives explodieren zu lassen.
- Performance: nach dem Entfernen eines zu teuren Copper-Expander-Pfads wieder deutlich besser; ein Normalfall wie `test18.png` laeuft grob im Bereich weniger Sekunden.

## Wichtige aktuelle Beobachtungen

- `test18.png` ist der aktuell beste Real-World-Test fuer gemischte Erze in einer Hoehle.
- Copper wird dort korrekt erkannt.
- Gold war zuletzt durch eine zu grosse Box auffaellig und wurde mit einer engeren Plausibilitaetspruefung entschärft.
- Ein Regression-Fall bleibt offen: `test2.png` erkennt das deepslate Copper aktuell nicht mehr.
- Coal ist bekannt schwierig und wurde bewusst nicht als Hauptziel priorisiert.

## Pipeline-Uebersicht

Die Hauptpipeline sitzt in `src/pipeline.py`.

Ablauf:

1. Bildvorverarbeitung mit Helligkeitsanpassung in `src/preprocessing.py`
2. Umwandlung nach HSV
3. Kantenbild erzeugen
4. Pro Erz eine Farbmaske erzeugen
5. Masken ueber HUD-, Wasser- und Grossflaechenfilter bereinigen
6. Erzspezifische Kandidaten per Konturen finden
7. Fuer passende Erze Template Matching gegen die Template-Bank ausfuehren
8. Kandidaten per NMS und Plausibilitaetsregeln filtern
9. Zusatzlogik fuer Sonderfaelle:
   - Copper: edge-cluster fallback
   - Iron: color-cluster fallback
   - Diamond: Candidate-Expander und kleine Cluster-Merges
   - Gold: harte Groessenbegrenzung gegen riesige Falschboxen

## Wichtige Codepunkte

- `src/pipeline.py`
  - Zentrale Pipeline und finale ROI-Plausibilitaeten.
  - Copper-Fallback ueber `_detect_copper_edge_clusters(...)`.
  - Iron-Fallback ueber `_detect_iron_color_clusters(...)`.
  - Gold hat eine enge Groessen- und Farbrestriktion.
- `src/detection.py`
  - Konturbasierte Kandidatensuche.
  - Template-Matching.
  - Entscheidungslogik fuer Label, Farbe und Score.
  - Copper-gegen-Emerald/Diamond Schutz.
- `src/runtime_mask_filter.py`
  - HUD-, Wasser- und Grossflaechenfilter.
  - Copper-Regionen werden nicht pauschal geloescht, wenn sie als grosse echte Copper-Fläche plausibel wirken.
- `src/ore_candidate_detection.py`
  - Diamond-Candidate-Expander.
  - CoalPrimaryDetector.
  - Der fruehere Copper-Expander ist nicht mehr Teil des aktiven Pfads.

## Debug- und Analyse-Tools

### `src/annotate.py`

Zum Zeichnen von Ground-Truth-Boxen.

Beispiel:

```bash
python3 -m minecraft_ore_detector.evaluation.annotate --image test1.png
```

Bedienung:

- `1` bis `8`: Label waehlen
- Linke Maustaste ziehen: Box zeichnen
- Rechte Maustaste: vorhandene Box auswaehlen
- `z`: letzte Box entfernen
- `h`: Schwierigkeit markieren
- `i`: Box ignorieren
- `s`: speichern
- `q` oder `ESC`: speichern und schliessen

### `src/review_detections.py`

Zum manuellen Bewerten der Detektionen. Das ist wichtig, weil die automatische Auswertung allein bei schwierigen Hoehlenbildern nicht reicht.

Beispiel:

```bash
python3 -m minecraft_ore_detector.evaluation.review_detections
```

Bedienung:

- `y`: Detektion korrekt
- `n`: Detektion falsch
- `i`: ignorieren bzw. als akzeptabel markieren
- `s`: Entscheidung ueberspringen/loeschen
- `a` / `d`: vorheriger / naechster Fall
- `q` oder `ESC`: speichern und schliessen

Die Entscheidungen landen in `data/annotations/manual_review.json`.

### `src/evaluate.py`

Berechnet Metriken gegen Ground Truth und optional gegen die manuelle Review.

Beispiele:

```bash
python3 -m minecraft_ore_detector.evaluation.evaluate
python3 -m minecraft_ore_detector.evaluation.evaluate --review data/annotations/manual_review.json
python3 -m minecraft_ore_detector.evaluation.evaluate --include-hard --hard-iou 0.20
python3 -m minecraft_ore_detector.evaluation.evaluate --min-f1 0.65
```

Wichtig:

- `--review` bindet die manuellen Entscheidungen ein.
- `--include-hard` nimmt als `hard` markierte Boxen mit auf.
- `--hard-iou` steuert die strengere IoU fuer `hard`-Falle.

### `src/debug_visualization.py`

Erstellt ein visuelles Debug-Board pro Bild mit den internen Pipeline-Stufen.

Beispiele:

```bash
python3 -m minecraft_ore_detector.debug.visualization --image test17.png
python3 -m minecraft_ore_detector.debug.visualization --image test17.png --ore gold
python3 -m minecraft_ore_detector.debug.visualization --image test17.png --ore redstone
```

Was das Board zeigt:

- Originalbild
- Vorverarbeitetes Bild
- Kantenbild
- Farbmaske pro Erz
- finale Maske pro Erz
- Kandidaten und Roh-Detektionen
- Ground Truth
- Review-Overlay

Ausgabeort:

- `data/debug_visual/`

### `src/analyze_misses.py`

Das wichtigste Diagnosewerkzeug fuer Verpasser.

Beispiele:

```bash
python3 -m minecraft_ore_detector.debug.analyze_misses
python3 -m minecraft_ore_detector.debug.analyze_misses --image data/screenshots/test12.png --label copper
python3 -m minecraft_ore_detector.debug.analyze_misses --image data/screenshots/test18.png --label copper
```

Was es ausgibt:

- Bild, Label und GT-Box
- Stats fuer Original- und Processed-Crop
- Maskendichte in allen Pipeline-Stufen
- Template-Score fuer GT und erweiterte Box
- Kandidaten-Ranking nach IoU und Teilueberlappung
- Raw-Detections und finale Detektionen in der Naehe

Das Tool ist ideal, wenn man wissen will, ob ein Fehler in Farbe, Maske, Kandidatenbildung, Template-Matching oder Filterung entsteht.

### `src/debug_evaluation.py`

Konsolenbasierte Debug-/Ausgabevariante fuer Evaluation. Nuerzlich, wenn man ohne GUI schnell Metriken sehen will.

## Praktische Arbeitsweise fuer weitere Aenderungen

Empfohlener Zyklus:

1. Aenderung klein halten.
2. `src/analyze_misses.py` auf betroffenen Bildern ausfuehren.
3. `src/debug_visualization.py` anschauen, wenn die Boxen oder Masken unklar sind.
4. Manuell mit `src/review_detections.py` gegenpruefen.
5. Dann erst die Metrik mit `src/evaluate.py` werten.

Das Projekt ist bewusst auf nachvollziehbare klassische CV ausgelegt. Wenn eine Aenderung den Durchsatz oder die Interpretierbarkeit verschlechtert, ist sie wahrscheinlich der falsche Pfad.

## Vermeiden

- Kein Machine Learning.
- Keine neuen Abhaengigkeiten, die ML oder neuronale Netze einfuehren.
- Keine grossen brute-force Suchfenster, wenn dieselbe Information schon ueber einen kleineren, strukturierteren Pfad erreichbar ist.
- `src/config.py` lokal fuer eigene Testbilder verwenden, aber nicht blind als Projektstand committen.

