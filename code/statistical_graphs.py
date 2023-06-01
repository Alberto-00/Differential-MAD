import seaborn as sns
import matplotlib.pyplot as plt

#######################################################################################
# Accuracy
models = ['Amsl', 'Amsl', 'Amsl', 'Amsl',
          'Facemorpher', 'Facemorpher', 'Facemorpher', 'Facemorpher',
          'OpenCV', 'OpenCV', 'OpenCV', 'OpenCV',
          'StyleGAN', 'StyleGAN', 'StyleGAN', 'StyleGAN',
          'Webmorph', 'Webmorph', 'Webmorph', 'Webmorph']

sub_categories = ['Tesisti', 'Nostri (not merged)', 'Nostri (merged)', 'Smile (included)',
                  'Tesisti', 'Nostri (not merged)', 'Nostri (merged)', 'Smile (included)',
                  'Tesisti', 'Nostri (not merged)', 'Nostri (merged)', 'Smile (included)',
                  'Tesisti', 'Nostri (not merged)', 'Nostri (merged)', 'Smile (included)',
                  'Tesisti', 'Nostri (not merged)', 'Nostri (merged)', 'Smile (included)']

models_accuracy_smile = ['Amsl', 'Facemorpher', 'OpenCV', 'StyleGAN', 'Webmorph']

sub_categories_accuracy_smile = ['Smile (included)', 'Smile (included)', 'Smile (included)',
                                 'Smile (included)', 'Smile (included)']

# Tesisti, Nostri, Merged, Smile
accuracy_smile = [97.89, 98.79, 94.11, 96.75, 95.85]

accuracy = [95.52, 98.32, 99.29, 97.89,
            98.18, 83.52, 98.72, 98.79,
            98.79, 88.29, 92.52, 94.11,
            98.71, 96.21, 97.05, 96.75,
            94.63, 91.16, 95.54, 95.85]


#######################################################################################
# Equal Erro Rate (EER) compared with paper
models_eer_paper = ['Amsl', 'Amsl',
                    'Facemorpher', 'Facemorpher',
                    'OpenCV', 'OpenCV',
                    'StyleGAN', 'StyleGAN',
                    'Webmorph', 'Webmorph']

sub_categories_eer_paper = ['Nostri', 'Paper',
                            'Nostri', 'Paper',
                            'Nostri', 'Paper',
                            'Nostri', 'Paper',
                            'Nostri', 'Paper']

eer_paper = [12.74, 15.18,
             0.49, 3.87,
             0.57, 4.39,
             9.82, 8.99,
             7.53, 12.35]

#######################################################################################
# BPCR (0.10%) @ APCR =
bpcr01 = [58.82, 100, 100, 100,
          63.73, 100, 100, 100,
          71.08, 100, 100, 100,
          79.90, 100, 100, 100,
          74.51, 100, 100, 100]

# BPCR (1.00%) @ APCR =
bpcr1 = [33.33, 49.02, 0, 27.94,
         26.47, 0, 52.94, 100,
         22.06, 0, 0, 0,
         42.16, 37.74, 16.70, 15.69,
         53.92, 1.47, 0.98, 1.96]

# BPCR (10.00%) @ APCR =
bpcr10 = [9.80, 0, 0, 0,
          0.49, 0, 52.94, 100,
          0.49, 0, 0, 0,
          8.82, 0, 0, 0,
          9.80, 0, 0, 0]

# BPCR (20.00%) @ APCR =
bpcr20 = [4.41, 0, 0, 0,
          0, 0, 52.94, 100,
          0.49, 0, 0, 0,
          4.41, 0, 0, 0,
          2.94, 0, 0, 0]

#######################################################################################

# Creazione del grafico a colonne con sottocategorie
sns.barplot(x=models, y=bpcr20, hue=sub_categories,
            palette='pastel', saturation=0.7, alpha=0.8)

# Aggiunta di etichette
plt.xlabel('Models')
plt.ylabel('%')
plt.title('BPCR (20.00%) @ APCR =')

# Modifica della grandezza delle colonne
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Visualizzazione del grafico
plt.show()
