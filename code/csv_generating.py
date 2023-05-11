import csv
import os

# Specificare la cartella di cui si vuole ottenere i nomi dei file
cartella = "dataset/FRLL_test/FRLL-Morphs/morph_webmorph"

# Ottenere una lista dei nomi dei file nella cartella
nomi_file = os.listdir(cartella)

# Creare una lista contenente il percorso completo di ciascun file nella cartella
percorso_file = [os.path.join(cartella, nome) for nome in nomi_file]

# Aprire un file CSV in modalità scrittura
with open("dataset/FRLL_test/test_morph_webmorph.csv", "w", newline="") as csvfile:
    # Creare un oggetto writer per scrivere nel file CSV
    writer = csv.writer(csvfile)

    # Scrivere i nomi dei campi nel file CSV
    writer.writerow(["image_path"])

    # Scrivere ciascun percorso di file come una riga nel file CSV
    for percorso in percorso_file:
        writer.writerow([percorso])