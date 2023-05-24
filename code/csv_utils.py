import csv
import os
import pandas as pd


def csv_generation():
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


def check_and_add(file1, file2):
    data_tesisti = {}
    data_train = {}

    with open(file1, 'r') as file1:
        reader1 = csv.reader(file1)
        for row in reader1:
            if row[0] != 'image_path':
                key = row[0]  # valore nella prima colonna
                data_tesisti[key] = row[2:]  # memorizza i dati dalla terza colonna in poi

    with open(file2, 'r') as file2:
        reader2 = csv.reader(file2)
        for row in reader2:
            if len(row[0].split('/')) > 1:
                key = row[0].split('/')[3]  # valore nella prima colonna
                data_train[key] = row[0:]  # memorizza i dati dalla prima colonna in poi

    merged_data = {}
    for key in data_train:
        if key in data_tesisti:
            merged_data[key] = data_train[key] + data_tesisti[key]

    with open('output/feature_extraction/model_webmorph/merge_csv/merged.csv', 'w', newline='') as merged_file:
        writer = csv.writer(merged_file)
        for key in merged_data:
            row = merged_data[key]
            writer.writerow(row)


# Utilizzo della funzione verifica_e_aggiungi
check_and_add('dataset/csv_tesisti/train_tesisti.csv', 'output/feature_extraction/model_webmorph/train/train.csv')
