# Kursaufgabe: MNIST Ziffernklassifikation

Dieses Projekt implementiert eine Klassifizierungsaufgabe für handgeschriebene Ziffern im Rahmen des Kurses "Sprachliche Informationsverarbeitung". Es verwendet Gaußsche Naive Bayes- und Entscheidungsbaum-Klassifikatoren, um Ziffern aus dem MNIST-Datensatz zu klassifizieren.

## Projektübersicht

Dieses Projekt vergleicht zwei verschiedene Machine-Learning-Algorithmen (**Gaussian Naive Bayes** und **Decision Tree**) zur Klassifizierung handgeschriebener Ziffern. Die Implementierung nutzt Python und die Bibliothek `scikit-learn`.

## Wie das Training funktioniert

Der Trainingsprozess wird durch das Skript `src/nb_dt_training/main.py` gesteuert und läuft in folgenden Schritten ab:

1.  **Daten laden**:
    *   Die Trainingsdaten (`train.csv`) und Testdaten (`test.csv`) werden aus dem Ordner `data/mnist/` eingelesen.
    *   Jede Zeile im CSV repräsentiert ein Bild: Die erste Spalte ist das korrekte Label (Ziffer 0-9), die restlichen 784 Spalten sind die Pixelwerte (28x28 Pixel Graustufen).
    *   Dafür wird die Hilfsfunktion `read_dataset` aus `utils.py` verwendet.
    *   Optional wird ein zufälliges Beispielbild zur Überprüfung ausgegeben.

2.  **Vorverarbeitung**:
    *   Die eingelesenen Listen werden in `numpy`-Arrays konvertiert, da `scikit-learn` diese für effiziente Berechnungen benötigt.

3.  **Modell 1: Naive Bayes**:
    *   Ein `GaussianNB`-Klassifikator wird initialisiert.
    *   **Training (`fit`)**: Das Modell lernt die Wahrscheinlichkeitsverteilungen der Pixelwerte für jede Ziffer basierend auf den Trainingsdaten (`x_train`, `y_train`).
    *   **Vorhersage (`predict`)**: Das trainierte Modell klassifiziert die Bilder aus dem Testdatensatz.

4.  **Modell 2: Decision Tree**:
    *   Ein `DecisionTreeClassifier` wird initialisiert (mit `random_state=42` für reproduzierbare Ergebnisse).
    *   **Training (`fit`)**: Der Entscheidungsbaum lernt Regeln, um die Ziffern basierend auf den Pixelwerten zu unterscheiden.
    *   **Vorhersage (`predict`)**: Auch dieses Modell klassifiziert anschließend die Testdaten.

5.  **Evaluation**:
    *   Für beide Modelle werden **Precision**, **Recall** und der **F1-Score** berechnet.
    *   Dabei wird der `weighted` Average verwendet, um eventuelle Ungleichgewichte in der Häufigkeit der Ziffern zu berücksichtigen.

6.  **Ergebnisse speichern**:
    *   Die berechneten Metriken werden in einer CSV-Datei unter `results/evaluation.csv` gespeichert.

## Projektstruktur

*   **`src/nb_dt_training/main.py`**: Das Hauptprogramm. Steuert den gesamten Ablauf von Datenladen bis Speichern der Ergebnisse.
*   **`src/nb_dt_training/utils.py`**: Enthält Hilfsfunktionen:
    *   `read_dataset(filename)`: Liest die CSV-Dateien ein.
    *   `display_item(item)`: Gibt eine Ziffer als ASCII-Art in der Konsole aus (zur visuellen Überprüfung).
*   **`pyproject.toml`**: Definiert die Projektabhängigkeiten (`scikit-learn`, `numpy`) und den Befehl `train`.
*   **`data/mnist/`**: Enthält die Rohdaten (`train.csv`, `test.csv`).
*   **`results/evaluation.csv`**: Hier werden die Leistungsmetriken der Modelle gespeichert.

## Ausführung

Um die Klassifikation und Evaluation durchzuführen, verwende `uv`:

```bash
uv run train
```
