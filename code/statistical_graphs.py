import seaborn as sns
import matplotlib.pyplot as plt

#######################################################################################
models = ['Amsl', 'Amsl', 'Amsl',
          'Facemorpher', 'Facemorpher', 'Facemorpher',
          'OpenCV', 'OpenCV', 'OpenCV',
          'StyleGAN', 'StyleGAN', 'StyleGAN',
          'Webmorph', 'Webmorph', 'Webmorph']

sub_categories = ['Tesisti', 'Nostri (not merged)', 'Nostri (merged)',
                  'Tesisti', 'Nostri (not merged)', 'Nostri (merged)',
                  'Tesisti', 'Nostri (not merged)', 'Nostri (merged)',
                  'Tesisti', 'Nostri (not merged)', 'Nostri (merged)',
                  'Tesisti', 'Nostri (not merged)', 'Nostri (merged)']

sub_categories_smile = ['Tesisti', 'Nostri (not merged)', 'Nostri (Smile)',
                        'Tesisti', 'Nostri (not merged)', 'Nostri (Smile)',
                        'Tesisti', 'Nostri (not merged)', 'Nostri (Smile)',
                        'Tesisti', 'Nostri (not merged)', 'Nostri (Smile)',
                        'Tesisti', 'Nostri (not merged)', 'Nostri (Smile)']

# Accuracy - NO SMILE
accuracy = [95.52, 98.32, 99.29,
            98.18, 83.52, 98.72,
            98.79, 88.29, 92.52,
            98.71, 96.21, 97.05,
            94.63, 91.16, 95.54]


# Accuracy - SMILE
accuracy_smile = [91.46, 98.32, 97.89,
                  91.23, 83.52, 98.79,
                  91.78, 88.29, 94.11,
                  91.72, 96.21, 96.75,
                  87.92, 91.16, 95.85]


#######################################################################################
# eer compared with paper
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


# EER - NO SMILE
eer = [9.80, 0.98, 0.39,
       2.20, 9.62, 2.04,
       2.42, 6.83, 4.05,
       8.99, 3.43, 4.29,
       9.86, 5.16, 2.42]


# EER - SMILE
eer_smile = [48.91, 0.98, 1.45,
             47.70, 9.62, 3.92,
             49.14, 6.83, 3.48,
             45.66, 3.43, 4.73,
             47.82, 5.16, 2.46]

#######################################################################################
# *********  NO SMILE *********************

# BPCR (0.10%) @ APCR =
bpcr01 = [58.82, 100, 100,
          63.73, 100, 100,
          71.08, 100, 100,
          79.90, 100, 100,
          74.51, 100, 100]

# BPCR (1.00%) @ APCR =
bpcr1 = [33.33, 49.02, 0,
         26.47, 0, 52.94,
         22.06, 0, 0,
         42.16, 37.74, 16.70,
         53.92, 1.47, 0.98]

# BPCR (10.00%) @ APCR =
bpcr10 = [9.80, 0, 0,
          0.49, 0, 52.94,
          0.49, 0, 0,
          8.82, 0, 0,
          9.80, 0, 0]

# BPCR (20.00%) @ APCR =
bpcr20 = [4.41, 0, 0,
          0, 0, 52.94,
          0.49, 0, 0,
          4.41, 0, 0,
          2.94, 0, 0]


# *********  SMILE *********************

# BPCR (0.10%) @ APCR =
bpcr01_smile = [50.00, 100, 100,
                50.00, 100, 100,
                50.00, 100, 100,
                50.00, 100, 100,
                50.00, 100, 100]

# BPCR (1.00%) @ APCR =
bpcr1_smile = [50.00, 49.02, 27.94,
                50.00, 0, 100,
                49.50, 0, 0,
                49.50, 37.74, 15.69,
                49.50, 1.47, 1.96]

# BPCR (10.00%) @ APCR =
bpcr10_smile = [49.50, 0, 0,
                48.52, 0, 100,
                48.52, 0, 0,
                48.52, 0, 0,
                48.52, 0, 0]

# BPCR (20.00%) @ APCR =
bpcr20_smile = [48.52, 0, 0,
                48.52, 0, 100,
                48.52, 0, 0,
                48.52, 0, 0,
                48.52, 0, 0]

#######################################################################################

# Creazione del grafico a colonne con sottocategorie
ax = sns.barplot(x=models, y=eer, hue=sub_categories,
            palette='pastel', saturation=0.7, alpha=0.8)

# Aggiungi il valore sulle barre
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2, height + 1, f'{height:.2f}', ha="center", fontsize=12)


# Grandezza grafico
fig = plt.gcf()
fig.set_size_inches(15, 11)

# Aggiunta di etichette
plt.xlabel('Models', fontsize=20)
plt.ylabel('EER', fontsize=20)
plt.title('Equal Error Rate (EER)', fontsize=20)

# Modifica della grandezza delle colonne
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Posiziona la leggenda fuori dal grafico
plt.legend(loc="upper right", bbox_to_anchor=(1.0, 1.13), fontsize=12)

# Visualizzazione del grafico
plt.show()

