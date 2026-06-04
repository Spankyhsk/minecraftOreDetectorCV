# Minecraft Ore Detector CV

Automatische Erz-Erkennung in Minecraft mittels klassischer Bildverarbeitung.

---

## Projektbeschreibung

Minecraft Ore Detector CV ist ein Computer-Vision-Projekt zur Erkennung von Minecraft-Erzen in Screenshots und Videos.

Das Projekt verzichtet bewusst auf Machine Learning und basiert ausschließlich auf klassischen Bildverarbeitungstechniken.

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
2. Screenshot wird mit CLAHE + Blur vorverarbeitet.
3. Für jede Overworld-Erzfamilie wird eine HSV-Maske mit Kantenmaske kombiniert.
4. Kandidaten werden gefunden, gefiltert und nahe Boxen werden gemerged.
5. Für jeden Erztyp wird die passende Template-Bank aus `data/templates/` geladen
   (z. B. `diamond_ore.png` und `diamond_deepslate_ore.png`).
6. Für jeden Kandidaten wird Multi-Scale-Template-Matching ausgeführt.
7. Überlappende Doppel-Treffer werden per NMS entfernt.

### Hinweis zu Templates

Die Templates in `data/templates/` sind volle Screenshots (1080x1920).
Im Code wird automatisch ein zentraler Template-Ausschnitt extrahiert,
damit das Matching gegen kleine Kandidaten im Zielbild möglich ist.

### Debug / Headless-Auswertung

Für eine reine Konsolen-Auswertung ohne GUI:

```bash
python3 src/eval_debug.py
```

Start der normalen Pipeline mit Visualisierung:

```bash
python3 src/main.py
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
