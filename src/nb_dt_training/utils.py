"""Hilfsfunktionen zum Einlesen und Anzeigen von MNIST-Daten.

Dieses Modul kümmert sich um die grundlegende Datenverarbeitung:
Es liest die CSV-Dateien ein und bietet eine einfache Möglichkeit, die Ziffern im Terminal zu visualisieren.
"""

import csv

def read_dataset(filename):
    """
    Liest den MNIST-Datensatz (CSV) ein.
    Erwartet das Format: Label, Pixel1, Pixel2, ..., Pixel784
    """
    x = []
    y = []
    
    with open(filename, newline='', encoding='utf-8') as csvfile:
        next(csvfile) # Kopfzeile überspringen
        
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            y.append(int(row[0])) # Die erste Spalte enthält das Label (die Ziffer)
            
            # Der Rest sind die Pixelwerte. Sie werden direkt in Integers umgewandelt.
            x.append([int(pixel) for pixel in row[1:]])
            
    return (x, y)

def display_item(item):
    """
    Gibt ein einzelnes MNIST-Bild (28x28 Pixel) als Text in der Konsole aus.
    Das ist hilfreich, um stichprobenartig zu prüfen, ob die Daten korrekt geladen wurden.
    """
    cell = 0
    for _r in range(0, 28): 
        for _c in range(0, 28): 
            # Breite 3 sorgt für eine halbwegs quadratische Darstellung im Terminal
            print(f"{item[cell]:3.0f}", end="")
            cell = cell + 1
        print()