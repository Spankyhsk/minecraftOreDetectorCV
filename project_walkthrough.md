# Projekt-Dokumentation: Minecraft Ore Detector CV

Dieses Dokument beschreibt die Architektur, Funktionsweise und Methodik des computergestützten Bildverarbeitungssystems **VoxelVision (Minecraft Ore Detector CV)**. Das System dient zur automatischen Erkennung von Minecraft-Erzen in Screenshots und verzichtet bewusst auf Deep Learning zugunsten klassischer 2D-Bildverarbeitungstechniken.

---

## 1. Systemarchitektur & Pipeline

Das System arbeitet als serielle Pipeline. Ein Eingabebild durchläuft definierte Phasen der Vorverarbeitung, Segmentierung, Kandidateneingrenzung, Validierung per Template-Matching und abschließenden Filterung.

```mermaid
graph TD
    A[Input Screenshot] --> B[Vorverarbeitung: Helligkeitsnormalisierung]
    B --> C[Maskenerstellung: HSV-Farbe + Canny-Kante]
    C --> D[Morphologische Bereinigung: Opening & Closing]
    D --> E[Kandidatensuche: Konturen + Bounding Box Merging]
    E --> F[Template-Matching: Multi-Scale auf Grau- und Kantenwerten]
    F --> G[Plausibilitätsprüfung: HSV-Farb-Kompatibilität]
    G --> H[Non-Maximum Suppression (NMS)]
    H --> I[Visualisierung & GUI-Ausgabe]
```

---

## 2. Detaillierte Modulbeschreibung

### A. Vorverarbeitung (`minecraft_ore_detector.imaging.preprocessing`)
Die Vorverarbeitung bereitet das rohe Screenshot-Bild für die mathematische Segmentierung vor.
*   **Helligkeitsnormalisierung:**
    Der Median des HSV-Helligkeitskanals wird an ein Referenzbild angeglichen. Dadurch werden sehr helle und sehr dunkle Szenen vergleichbarer, ohne einen adaptiven CLAHE-Kontrastausgleich vorzutäuschen.
*   **Kein zusätzlicher Blur:**
    Die aktuelle Hauptpipeline glättet das vorverarbeitete Farbbild nicht. Nur die Kantenberechnung verwendet intern einen kleinen Gauß-Filter.
*   **HSV-Konvertierung:**
    Für die Farbsegmentierung wird das Bild in den HSV-Farbraum (Hue, Saturation, Value) transformiert. Im Gegensatz zu RGB/BGR trennt HSV den Farbton von der Helligkeit, was die Erkennung unter wechselnden Lichtbedingungen (z. B. Fackellicht oder Schatten) massiv erleichtert.

### B. Segmentierung und Bereinigung (`minecraft_ore_detector.imaging`)
Hier werden relevante Farbinformationen extrahiert.
*   **HSV-Thresholding:**
    Jedes Erz besitzt ein definiertes Farbprofil im HSV-Farbraum. Die Schwellenwerte sind bewusst tolerant gewählt, um möglichst alle Erze zu erfassen (hoher Recall). Redstone stellt eine Besonderheit dar: Da Rot im HSV-Farbraum am Rand liegt (Schnittstelle 0/180), werden hier zwei getrennte Bereiche über eine ODER-Verknüpfung kombiniert.
*   **Kantenerkennung (Canny Edge):**
    Ein Canny-Filter ermittelt signifikante Helligkeitsübergänge. Die erzeugten Kanten helfen, die geometrischen Grenzen von Erzblöcken zu rekonstruieren.
*   **Hybride Maskierung:**
    Farbmaske und Kantenmaske werden bitweise ver-ODER-t (Ausnahme: Kohle). Die Kanten stützen Bereiche, in denen die Farbinformation aufgrund von Schatteneinfall verloren ging.
*   **Morphologische Operationen:**
    Mit einem $3 \times 3$ Kernel wird erst ein *Opening* (entfernt isolierte Störpixel) und anschließend ein *Closing* (schließt kleine Risse und Hohlräume innerhalb von Clustern) ausgeführt.

### C. Kandidatensuche (`minecraft_ore_detector.detection.candidate_finder`)
Aus der bereinigten Binärmaske werden konkrete Kandidatenboxen extrahiert.
*   **Konturanalyse:**
    `cv2.findContours` erfasst zusammenhängende weiße Flächen.
*   **Geometrisches Filtern:**
    Die umschließenden Rechtecke (Bounding Boxes) werden anhand von Größengrenzen (abhängig von der Bildauflösung) gefiltert. Zu kleine Flächen (Artefakte) oder zu große Flächen (Decken/Böden) werden verworfen.
*   **Farb-Dichte-Validierung:**
    Es wird geprüft, wie hoch der Anteil von Pixeln der echten Erzfarbe innerhalb des Kandidatenrechtecks ist. Kantenreiche Steinstrukturen ohne Farbpigmente werden so frühzeitig aussortiert.
*   **Bounding Box Merging (`_merge_nearby_boxes`):**
    Erze treten im Spiel oft in Adern auf. Durch perspektivische Verzerrungen oder Texturlücken entstehen manchmal mehrere kleine Bounding Boxes für einen einzigen Block. Liegen Boxen näher als 18 Pixel beieinander, werden sie zu einer Region verschmolzen.

### D. Erz-Matching und Plausibilität (`minecraft_ore_detector.detection`)
Dies ist der rechenintensivste und wichtigste Schritt zur Validierung der Kandidaten.
*   **Template-Generierung (`load_template`):**
    Die Template-Datenbank enthält Vollbild-Screenshots. Das System schneidet beim Laden automatisch den zentrierten Erzblock aus, indem es die neutrale Wandfarbe (Beton) detektiert, herausfiltert und die Bounding Box des zentrierten Objekts extrahiert.
*   **Helligkeits-Split (Stone vs. Deepslate):**
    In Minecraft generieren Erze entweder in hellem Stein (Stone) oder in dunklem Tiefenschiefer (Deepslate). Das System berechnet die mittlere Helligkeit des Kandidatenareals. Liegt dieser unter dem Schwellenwert 95.0, werden für das Template-Matching nur Deepslate-Templates geladen, andernfalls nur normale Stone-Templates. Dies halbiert die Rechenzeit und verhindert Verwechslungen.
*   **Multi-Scale Template-Matching (`match_template_multiscale`):**
    Da Erzblöcke unterschiedlich weit entfernt sein können, wird das Template in 8 Stufen skaliert ($0.05$ bis $1.25$). Für jede Skalierungsstufe wird ein Template-Matching auf dem Graustufenbild sowie auf den Canny-Kantenbildern durchgeführt (letzteres ist extrem robust gegen Helligkeitsschwankungen).
*   **Farb-Kompatibilitäts-Score (`_color_compatibility`):**
    Zusätzlich zum rein strukturellen Matching berechnet das System einen Farb-Plausibilitätswert. Es vergleicht die mittlere Farbe des Ausschnitts im HSV-Raum mit den erztypischen Referenzwerten.
*   **Score-Gewichtung:**
    Der finale Score wird gewichtet berechnet: $85\%$ Formähnlichkeit (Template-Matching-Score) + $15\%$ Farbplausibilität. Nur wenn dieser kombinierte Wert den erzspezifischen Grenzwert überschreitet, gilt das Erz als erkannt.

### E. Überlappungsbereinigung & Ausgabe
*   **Non-Maximum Suppression (NMS) (`non_max_suppression`):**
    Da verschiedene Erz-Detektoren (z. B. Gold und Eisen) denselben Block erfassen könnten oder mehrere Skalierungsstufen anschlagen, filtert NMS überlappende Boxen. Basierend auf der **Intersection over Union (IoU)** werden Boxen mit einer Überlappung von $\ge 25\%$ verglichen; nur die Box mit dem höchsten Übereinstimmungs-Score wird behalten.
*   **Visualisierung (`minecraft_ore_detector.presentation.visualization`):**
    Die Ergebnisse werden mit OpenCV gezeichnet. Ein Tkinter-Aufruf ermittelt die Bildschirmauflösung des Host-Systems, um das Bild dynamisch und ohne Verzerrung an das Display anzupassen. Im Debug-Modus werden zusätzlich alle Roh-Kandidaten (blau) eingezeichnet.

---

## 3. Kritische Analyse & Verbesserungspotenzial (Vorschau für den nächsten Schritt)

Obwohl die Pipeline eine sehr hohe Erkennungsrate auf den Testbildern aufweist, gibt es Schwachstellen, die für eine exzellente Hochschulabgabe verbessert werden sollten:

1.  **Statische HSV-Grenzwerte (`segmentation.py`):**
    Die Farbgrenzen in `ORE_CONFIG` sind statisch im Code hinterlegt. Bei veränderten Helligkeiten (z.B. sehr dunkle Höhlen ohne Fackellicht oder veränderte Shader/Texture-Packs) kann die Erkennung einbrechen.
2.  **Performance des Multi-Scale Matchings (`detection.py`):**
    Die Skalierung des Templates über 8 Stufen hinweg und das doppelte Matching (Graustufen + Canny) für jeden Kandidaten ist rechenintensiv. Für eine geplante Echtzeit-Videoanalyse (siehe `README.md`) ist dies zu langsam.
3.  **Fehlender Helligkeits-Split bei Kohle:**
    Kohle nutzt aktuell eine globale Template-Suche über das gesamte Bild, falls keine Kandidaten gefunden werden. Dies führt bei großen Suchbereichen zu Performance-Einbußen.
4.  **Harte Bounding-Box-Zusammenführung (`_merge_nearby_boxes`):**
    Die Zusammenführung nutzt einen festen Pixel-Abstand (`gap=18`). Bei hoher Auflösung (4K) oder geringer Auflösung (720p) verhält sich dieser Abstand anders. Eine Normalisierung basierend auf der Bildauflösung wäre robuster.
5.  **HUD und UI Erkennung:**
    Das System filtert zwar den unteren Rand in Templates aus, aber im Suchbild selbst könnten HUD-Elemente (z. B. Gegenstände im Inventar) fälschlicherweise als Erze segmentiert werden.

---

## 4. Ausführung des Projekts

*   **Normale Pipeline mit Visualisierung:**
    ```bash
    python3 -m minecraft_ore_detector.app.main
    ```
*   **Headless Testauswertung (reine Textausgabe):**
    ```bash
    python3 -m minecraft_ore_detector.debug.evaluation
    ```
*   **HSV- und Farbdiagnose annotierter Boxen:**
    ```bash
    python3 -m minecraft_ore_detector.debug.color_diagnostics
    ```
