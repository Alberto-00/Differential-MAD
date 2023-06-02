import seaborn as sns
import matplotlib.pyplot as plt

#######################################################################################
models = ['Amsl', 'Amsl', 'Amsl',
          'Facemorpher', 'Facemorpher', 'Facemorpher',
          'OpenCV', 'OpenCV', 'OpenCV',
          'StyleGAN', 'StyleGAN', 'StyleGAN',
          'Webmorph', 'Webmorph', 'Webmorph']

sub_categories_no_smile = ['Approccio Geometrico (not Smile)', 'AoM not Smile', 'AoM & Approccio Geometrico (not Smile)',
                           'Approccio Geometrico (not Smile)', 'AoM not Smile', 'AoM & Approccio Geometrico (not Smile)',
                           'Approccio Geometrico (not Smile)', 'AoM not Smile', 'AoM & Approccio Geometrico (not Smile)',
                           'Approccio Geometrico (not Smile)', 'AoM not Smile', 'AoM & Approccio Geometrico (not Smile)',
                           'Approccio Geometrico (not Smile)', 'AoM not Smile', 'AoM & Approccio Geometrico (not Smile)']

sub_categories_smile = ['Approccio Geometrico (Smile)', 'AoM Smile', 'AoM & Approccio Geometrico (Smile)',
                        'Approccio Geometrico (Smile)', 'AoM Smile', 'AoM & Approccio Geometrico (Smile)',
                        'Approccio Geometrico (Smile)', 'AoM Smile', 'AoM & Approccio Geometrico (Smile)',
                        'Approccio Geometrico (Smile)', 'AoM Smile', 'AoM & Approccio Geometrico (Smile)',
                        'Approccio Geometrico (Smile)', 'AoM Smile', 'AoM & Approccio Geometrico (Smile)']

# Accuracy - NO SMILE
accuracy_no_smile = [95.52, 95.54, 99.29,
                     98.18, 81.50, 98.72,
                     98.79, 84.14, 92.52,
                     98.71, 95.39, 97.05,
                     94.63, 88.67, 95.54]


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

sub_categories_eer_paper = ['AoM', 'Paper',
                            'AoM', 'Paper',
                            'AoM', 'Paper',
                            'AoM', 'Paper',
                            'AoM', 'Paper']

eer_paper = [12.74, 15.18,
             0.49, 3.87,
             0.57, 4.39,
             9.82, 8.99,
             7.53, 12.35]


# EER - NO SMILE
eer_no_smile = [9.80, 3.38, 0.39,
                2.20, 10.02, 2.04,
                2.42, 8.59, 4.05,
                8.99, 3.39, 4.29,
                9.86, 6.14, 2.42]


# EER - SMILE
eer_smile = [48.91, 0.98, 1.45,
             47.70, 9.62, 3.92,
             49.14, 6.83, 3.48,
             45.66, 3.43, 4.73,
             47.82, 5.16, 2.46]

#######################################################################################
# *********  NO SMILE *********************

# BPCER (0.10%) @ APCER
bpcr01 = [58.82, 100, 100,
          63.73, 100, 100,
          71.08, 100, 100,
          79.90, 100, 100,
          74.51, 100, 100]

# BPCER (1.00%) @ APCER
bpcr1 = [33.33, 49.02, 0,
         26.47, 0, 52.94,
         22.06, 18.63, 0,
         42.16, 81.37, 16.70,
         53.92, 38.24, 0.98]

# BPCER (10.00%) @ APCER
bpcr10 = [9.80, 0, 0,
          0.49, 0, 52.94,
          0.49, 0, 0,
          8.82, 0, 0,
          9.80, 0, 0]

# BPCER (20.00%) @ APCER
bpcr20 = [4.41, 0, 0,
          0, 0, 52.94,
          0.49, 0, 0,
          4.41, 0, 0,
          2.94, 0, 0]


# *********  SMILE *********************

# BPCER (0.10%) @ APCER
bpcr01_smile = [50.00, 100, 100,
                50.00, 100, 100,
                50.00, 100, 100,
                50.00, 100, 100,
                50.00, 100, 100]

# BPCER (1.00%) @ APCER
bpcr1_smile = [50.00, 49.02, 27.94,
               50.00, 0, 100,
               49.50, 0, 0,
               49.50, 37.74, 15.69,
               49.50, 1.47, 1.96]

# BPCER (10.00%) @ APCER
bpcr10_smile = [49.50, 0, 0,
                48.52, 0, 100,
                48.52, 0, 0,
                48.52, 0, 0,
                48.52, 0, 0]

# BPCER (20.00%) @ APCER
bpcr20_smile = [48.52, 0, 0,
                48.52, 0, 100,
                48.52, 0, 0,
                48.52, 0, 0,
                48.52, 0, 0]

#######################################################################################

# Creazione del grafico a colonne con sottocategorie
ax = sns.barplot(x=models, y=bpcr20_smile, hue=sub_categories_smile,
            palette='pastel', saturation=0.7, alpha=0.8)

# Aggiungi il valore sulle barre
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2, height + 1, f'{height:.2f}%', ha="center", fontsize=10)


# Grandezza grafico
fig = plt.gcf()
fig.set_size_inches(15, 11)

# Aggiunta di etichette
plt.xlabel('Models', fontsize=20)
plt.ylabel('BPCER (20.00%) @ APCER', fontsize=20)
plt.title("BPCER (20.00%) @ APCER = (con 'smile')", fontsize=20)

# Modifica della grandezza delle colonne
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Posiziona la leggenda fuori dal grafico
plt.legend(loc="upper right", bbox_to_anchor=(1.0, 1.16), fontsize=12)

# Visualizzazione del grafico
plt.show()

