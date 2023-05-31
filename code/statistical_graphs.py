import seaborn as sns
import matplotlib.pyplot as plt

#######################################################################################
# Accuracy
models_accuracy = ['Amsl', 'Amsl', 'Amsl', 'Amsl',
                   'Facemorpher', 'Facemorpher', 'Facemorpher', 'Facemorpher',
                   'OpenCV', 'OpenCV', 'OpenCV', 'OpenCV',
                   'StyleGAN', 'StyleGAN', 'StyleGAN', 'StyleGAN',
                   'Webmorph', 'Webmorph', 'Webmorph', 'Webmorph']

sub_categories_accuracy = ['Tesisti', 'Nostri (not merged)', 'Merged', 'Smile (included)',
                           'Tesisti', 'Nostri (not merged)', 'Merged', 'Smile (included)',
                           'Tesisti', 'Nostri (not merged)', 'Merged', 'Smile (included)',
                           'Tesisti', 'Nostri (not merged)', 'Merged', 'Smile (included)',
                           'Tesisti', 'Nostri (not merged)', 'Merged', 'Smile (included)']

# Tesisti, Nostri, Merged, Smile
accuracy = [44, 98.32, 99.47, 97.89,
            57, 83.52, 98.72, 98.79,
            55, 87.94, 92.97, 93.20,
            54, 96.35, 96.98, 96.75,
            51, 92.29, 95.54, 96.37]

# ho preso solo l'eer del classificatore migliore
# in base alla tabella riportata
eer_pca = [0, 0.98, 0.29, 1.45,
           0, 9.62, 2.04, 3.92,
           0, 7.04, 3.81, 4.02,
           0, 2.94, 4.33, 4.53,
           0, 4.50, 2.42, 2.14]

#######################################################################################

# Error Equal rate (EER)
models_apcr = ['Amsl', 'Amsl', 'Amsl',
               'Facemorpher', 'Facemorpher', 'Facemorpher',
               'OpenCV', 'OpenCV', 'OpenCV',
               'StyleGAN', 'StyleGAN', 'StyleGAN',
               'Webmorph', 'Webmorph', 'Webmorph']

sub_categories_apcr = ['Tesisti', 'Nostri', 'Paper',
                       'Tesisti', 'Nostri', 'Paper',
                       'Tesisti', 'Nostri', 'Paper',
                       'Tesisti', 'Nostri', 'Paper',
                       'Tesisti', 'Nostri', 'Paper']

# Tesisti, Nostri, Paper
eer = [52.32, 12.74, 15.18,
       5.23, 0.49, 3.87,
       5.15, 0.57, 4.39,
       0.0, 9.82, 8.99,
       5.24, 7.53, 12.35]

# BPCR (0.10%) @ APCR =
bpcr01 = [66.43, 100.00, 76.47,
          5.56, 100.00, 63.24,
          5.48, 100.00, 64.22,
          0.0, 100.00, 79.90,
          5.56, 100.00, 90.69]

# BPCR (1.00%) @ APCR =
bpcr1 = [63.95, 68.78, 49.51,
         5.40, 0.49, 23.53,
         5.24, 0.57, 26.47,
         0.0, 41.73, 42.16,
         5.40, 43.90, 80.39]

# BPCR (10.00%) @ APCR =
bpcr10 = [59.81, 15.86, 21.08,
          5.23, 0.16, 0.49,
          5.15, 0.57, 1.96,
          0.0, 9.98, 8.82,
          5.15, 6.96, 15.20]

# BPCR (20.00%) @ APCR =
bpcr20 = [58.20, 7.82, 11.76,
          5.07, 0.16, 0.49,
          4.99, 0.57, 1.47,
          0.0, 5.16, 4.41,
          5.15, 4.59, 7.84]

#######################################################################################

# Creazione del grafico a colonne con sottocategorie
sns.barplot(x=models_accuracy, y=accuracy, hue=sub_categories_accuracy,
            color='blue', saturation=0.7, alpha=0.8)

# Aggiunta di etichette
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy')

# Modifica della grandezza delle colonne
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Visualizzazione del grafico
plt.show()
