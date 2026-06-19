# Minecraft Ore Detector CV

Automatische Erz-Erkennung in Minecraft mittels klassischer Bildverarbeitung.

---

## Projektbeschreibung

Minecraft Ore Detector CV ist ein Computer-Vision-Projekt zur Erkennung von Minecraft-Erzen in Screenshots und Videos.

Das Projekt verzichtet bewusst auf Machine Learning und basiert ausschließlich auf klassischen Bildverarbeitungstechniken.

Aktueller Uebergabestand und Debug-Workflow:
[docs/hand_off.md](docs/hand_off.md)

## Installation

Das Projekt wird als lokales Python-Paket installiert:

```bash
python3 -m pip install -e .
```

Für den optionalen Live-Modus:

```bash
python3 -m pip install -e ".[live]"
```

---

## Ziel des Projekts

Das Ziel ist es, verschiedene Minecraft-Erze (z. B. Diamant, Gold, Eisen) automatisch in Bildern zu erkennen und visuell zu markieren.

---

## Verwendete Methoden

### Vorverarbeitung
- Bildglättung (Gaussian Blur)
- Farbraum-Transformation (RGB → HSV)

### Segmentierung
- Farbthresholding im HSV-Raum
- Maskenerstellung für spezifische Erzfarben

### Bildverarbeitung
- Morphologische Operationen (Erosion, Dilation, Opening, Closing)
- Konturerkennung
- Connected Components

### Erkennung
- Template Matching für zusätzliche Validierung

---

## Pipeline

Der Ablauf des Systems ist wie folgt:

Input Bild  
→ Vorverarbeitung  
→ HSV-Segmentierung  
→ Morphologische Filterung  
→ Konturerkennung  
→ Erkennung der Erze  
→ Visualisierung der Ergebnisse  

### Aktueller Ablauf im Code

1. `src/main.py` laden und starten.
2. `OreDetector` aus `src/pipeline.py` steuert die komplette Pipeline.
3. Screenshot-Helligkeit wird anhand des Referenzbilds normalisiert.
4. Für jede Overworld-Erzfamilie wird eine HSV-Maske mit Kantenmaske kombiniert.
5. Kandidaten werden gefunden, gefiltert und nahe Boxen werden gemerged.
6. Für jeden Erztyp wird die passende Template-Bank aus `data/templates/` geladen
   (z. B. `diamond_ore.png` und `diamond_deepslate_ore.png`).
7. Für jeden Kandidaten wird Multi-Scale-Template-Matching ausgeführt.
8. Überlappende Doppel-Treffer werden per NMS entfernt.

### Code-Struktur

- `src/main.py`: schlanker Einstiegspunkt für die Einzelbildverarbeitung.
- `src/live_detection.py`: wiederverwendet einen Detector für fortlaufende Frames.
- `src/pipeline.py`: schlanke Ablaufsteuerung der kompletten Pipeline.
- `src/ore_detection_processor.py`: interne Verarbeitungsschritte und Erzstrategien.
- `src/coal_fallback_detector.py`: aufeinander aufbauende Coal-Sonderstrategien.
- `src/copper_detector.py`: Copper-spezifische Zusatzstrategien.
- `src/iron_detector.py`: Iron-spezifische Zusatzstrategien.
- `src/gold_detector.py`: Gold-spezifische Erkennung großer Maskenbereiche.
- `src/diamond_postprocessor.py`: Diamond-Merging und Anpassung kleiner Clusterboxen.
- `src/config.py`: Pfade, Debug-Schalter und Matching-Thresholds.
- `src/runtime_mask_filter.py`: HUD-, Wasser- und Großflächenfilter.
- `src/ore_candidate_detection.py`: Sonderfälle für Coal und Diamond-Kandidaten.
- `src/template_repository.py`: Laden und Caching der Template-Banks.
- `src/segmentation.py`, `src/morphology.py`, `src/detection.py`: klassische Bildverarbeitungsbausteine.

### Hinweis zu Templates

Die Templates in `data/templates/` sind volle Screenshots (1080x1920).
Im Code wird automatisch ein zentraler Template-Ausschnitt extrahiert,
damit das Matching gegen kleine Kandidaten im Zielbild möglich ist.

### Debug / Headless-Auswertung

Für eine reine Konsolen-Auswertung ohne GUI:

```bash
python3 -m minecraft_ore_detector.debug.evaluation
```

### Ground Truth / Regression Tests

Manuelle Boxen werden in `data/annotations/ground_truth.json` gespeichert.
Zum Eintragen gibt es ein kleines Zeichenwerkzeug:

```bash
python3 -m minecraft_ore_detector.evaluation.annotate --image test1.png
```

Bedienung im Fenster:

- `1` bis `8`: Erzlabel wählen (`1=Coal`, `2=Copper`, `3=Diamond`, `4=Emerald`, `5=Gold`, `6=Iron`, `7=Lapis`, `8=Redstone`)
- Linke Maustaste ziehen: Box zeichnen
- Rechte Maustaste: vorhandene Box auswählen
- `z`: letzte Box entfernen
- `h`: ausgewählte Box als schwierig markieren; ohne Auswahl gilt `h` für neu gezeichnete Boxen
- `i`: ausgewählte Box ignorieren
- `s`: speichern
- `q` oder `ESC`: speichern und schließen

Die automatische Genauigkeitsmessung läuft danach mit:

```bash
python3 -m minecraft_ore_detector.evaluation.evaluate
```

Falls du die automatischen Treffer/Misses manuell gegenpruefen willst:

```bash
python3 -m minecraft_ore_detector.evaluation.review_detections
```

Das Review-Fenster zeigt nacheinander `TP`, `FP` und `FN`-Faelle:

- `y`: Fall als gut/korrekt markieren
- `n`: Fall als schlecht/falsch markieren
- `i`: Detektion ignorieren bzw. verpassten Fall als akzeptabel markieren
- `s`: manuelle Entscheidung loeschen/ueberspringen
- `a` / `d`: vorheriger / naechster Fall
- `q` oder `ESC`: speichern und schliessen

Die Entscheidungen landen in `data/annotations/manual_review.json`.
Die Evaluation nutzt sie mit:

```bash
python3 -m minecraft_ore_detector.evaluation.evaluate --review data/annotations/manual_review.json
```

Fuer visuelles Pipeline-Debugging kann ein Debug-Board pro Bild erzeugt werden:

```bash
python3 -m minecraft_ore_detector.debug.visualization --image test17.png
```

Das Board landet unter `data/debug_visual/` und zeigt Original, Vorverarbeitung,
Kanten, finale Treffer, Ground Truth, Review-Overlay sowie Farb-/Clean-Masken
inklusive Kandidaten pro Erz. Einzelne Erze koennen isoliert werden:

```bash
python3 -m minecraft_ore_detector.debug.visualization --image test17.png --ore gold
python3 -m minecraft_ore_detector.debug.visualization --image test17.png --ore redstone
```

Wenn nur die Kandidatenboxen ohne finale Detektionen sichtbar sein sollen:

```bash
python3 -m minecraft_ore_detector.debug.visualization --image test17.png --candidates-only
python3 -m minecraft_ore_detector.debug.visualization --image test17.png --ore gold --candidates-only
```

Standardmäßig werden `difficulty: "hard"` und `ignore: true` nicht streng bewertet.
Hard-Boxen können optional mit niedrigerer IoU in die Metrik aufgenommen werden:

```bash
python3 -m minecraft_ore_detector.evaluation.evaluate --include-hard --hard-iou 0.20
```

Optional kann ein Mindestwert für Regression-Checks gesetzt werden:

```bash
python3 -m minecraft_ore_detector.evaluation.evaluate --min-f1 0.65
```

Start der normalen Pipeline mit Visualisierung:

```bash
python3 -m minecraft_ore_detector.app.main
```

---

## Geplante Features

- Bounding Boxes mit Labels
- Videoanalyse in Echtzeit
- Statistische Auswertung der Erze im Bild
- FPS-Optimierung für Live-Verarbeitung

---

## Hinweise

Dieses Projekt wurde im Rahmen eines 2D-Computer-Vision-Moduls entwickelt und nutzt ausschließlich klassische Bildverarbeitung ohne neuronale Netze oder andere Machine-Learning-Verfahren.
