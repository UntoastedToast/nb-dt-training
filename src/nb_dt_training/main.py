"""
Hauptskript für das Training und die Evaluation der MNIST-Klassifikatoren.

Steuert den gesamten Ablauf: Daten laden, Modelle trainieren und die Leistung bewerten.
"""

import os
import csv
import time
import random
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

# Eigene Tools
from nb_dt_training.utils import read_dataset, display_item

def train_and_evaluate(model, x_train, y_train, x_test, y_test, model_name):
    """
    Führt das Training für ein Modell durch und evaluiert anschließend die Vorhersagen.
    """
    print(f"Training {model_name}...")
    start_time = time.time()
    
    # Hier findet der Lernprozess statt
    model.fit(x_train, y_train)
    
    duration = time.time() - start_time
    print(f"Training beendet in {duration:.2f} Sekunden.")

    print(f"Vorhersage mit {model_name}...")
    y_pred = model.predict(x_test)

    # 'weighted' wird verwendet, um eventuelle Ungleichgewichte in der Klassenverteilung zu berücksichtigen.
    # Das liefert einen repräsentativeren Durchschnitt als ein einfaches Mittel.
    # 'zero_division=0' verhindert Warnungen, falls eine Ziffer gar nicht vorhergesagt wurde.
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    return [model_name, f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"]

def main():
    # Pfade werden relativ zum Arbeitsverzeichnis gesetzt, damit das Skript portabel bleibt
    base_dir = os.getcwd()
    train_path = os.path.join(base_dir, "data", "mnist", "train.csv")
    test_path = os.path.join(base_dir, "data", "mnist", "test.csv")
    results_dir = os.path.join(base_dir, "results")
    results_path = os.path.join(results_dir, "evaluation.csv")

    os.makedirs(results_dir, exist_ok=True)

    print(f"Lade Trainingsdaten von {train_path}...")
    x_train, y_train = read_dataset(train_path)
    
    print(f"Lade Testdaten von {test_path}...")
    x_test, y_test = read_dataset(test_path)

    print(f"Anzahl Trainingselemente: {len(x_train)}")
    print(f"Anzahl Testelemente: {len(x_test)}")

    # Zur Sicherheit wird ein zufälliges Beispiel ausgegeben.
    # So ist direkt ersichtlich, ob die Daten korrekt eingelesen wurden.
    if x_train:
        item = random.randint(0, len(x_train) - 1)
        print(f"Dies ist Element {item} (Label: {y_train[item]}):")
        display_item(x_train[item])

    # Listen werden in NumPy-Arrays konvertiert, da Scikit-learn damit deutlich effizienter arbeitet.
    print("Konvertiere Daten in NumPy-Arrays...")
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    results = [["Model", "Precision", "Recall", "F1 Score"]]

    # 1. Kandidat: Gaussian Naive Bayes
    # Start mit Naive Bayes als Baseline. Es ist schnell und liefert oft erste brauchbare Ergebnisse
    # ohne komplexes Hyperparameter-Tuning.
    nb_model = GaussianNB()
    nb_results = train_and_evaluate(nb_model, x_train, y_train, x_test, y_test, "Naive Bayes")
    results.append(nb_results)

    # 2. Kandidat: Decision Tree
    # Als Vergleich dient ein Entscheidungsbaum, der explizite Regeln aus den Pixelwerten ableitet.
    dt_model = DecisionTreeClassifier(random_state=42) 
    dt_results = train_and_evaluate(dt_model, x_train, y_train, x_test, y_test, "Decision Tree")
    results.append(dt_results)

    print(f"Speichere Evaluationsergebnisse in {results_path}...")
    with open(results_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(results)
    
    print("Fertig.")

if __name__ == "__main__":
    main()