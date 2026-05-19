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

---

## Geplante Features

- Erkennung von mehreren Erzarten
- Bounding Boxes mit Labels
- Videoanalyse in Echtzeit
- Statistische Auswertung der Erze im Bild
- FPS-Optimierung für Live-Verarbeitung

---

## Hinweise

Dieses Projekt wurde im Rahmen eines 2D-Computer-Vision-Moduls entwickelt und nutzt ausschließlich klassische Bildverarbeitung ohne neuronale Netze oder andere Machine-Learning-Verfahren.
