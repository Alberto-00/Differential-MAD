import seaborn as sns
import matplotlib.pyplot as plt

#######################################################################################
models = ['Amsl', 'Amsl', 'Amsl', 'Amsl',
          'Facemorpher',  'Facemorpher', 'Facemorpher', 'Facemorpher',
          'OpenCV', 'OpenCV', 'OpenCV', 'OpenCV',
          'StyleGAN', 'StyleGAN', 'StyleGAN', 'StyleGAN',
          'Webmorph', 'Webmorph', 'Webmorph', 'Webmorph']



sub_categories_no_smile = ['Paper', 'Approccio Geometrico (not Smile)', 'AoM not Smile',
                           'AoM & Approccio Geometrico (not Smile)',
                           'Paper', 'Approccio Geometrico (not Smile)', 'AoM not Smile',
                           'AoM & Approccio Geometrico (not Smile)',
                           'Paper','Approccio Geometrico (not Smile)', 'AoM not Smile',
                           'AoM & Approccio Geometrico (not Smile)',
                           'Paper', 'Approccio Geometrico (not Smile)', 'AoM not Smile',
                           'AoM & Approccio Geometrico (not Smile)',
                           'Paper', 'Approccio Geometrico (not Smile)', 'AoM not Smile',
                           'AoM & Approccio Geometrico (not Smile)']

sub_categories_smile = ['Paper', 'Approccio Geometrico (Smile)', 'AoM Smile', 'AoM & Approccio Geometrico (Smile)',
                        'Paper', 'Approccio Geometrico (Smile)', 'AoM Smile', 'AoM & Approccio Geometrico (Smile)',
                        'Paper', 'Approccio Geometrico (Smile)', 'AoM Smile', 'AoM & Approccio Geometrico (Smile)',
                        'Paper', 'Approccio Geometrico (Smile)', 'AoM Smile', 'AoM & Approccio Geometrico (Smile)',
                        'Paper', 'Approccio Geometrico (Smile)', 'AoM Smile', 'AoM & Approccio Geometrico (Smile)']

# Accuracy - NO SMILE
accuracy_no_smile = [95.52, 17.52, 95.76,
                     98.18, 95.54, 99.29,
                     98.79, 93.76, 97.51,
                     98.71, 70.17, 96.15,
                     94.63, 24.04, 95.46]

# Accuracy - SMILE
accuracy_smile = [91.46, 22.36, 97.76,
                  91.23, 98.32, 98.79,
                  91.78, 97.91, 98.79,
                  91.72, 75.04, 98.56,
                  87.92, 30.95, 98.72]

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
eer_no_smile = [15.18, 9.80, 43.17, 3.66,
                3.87, 2.20, 2.41, 0.39,
                4.39, 2.42, 3.38, 2.70,
                8.99, 8.99, 16.16, 3.43,
                12.35, 9.86, 41.15, 3.80]

# EER - SMILE
eer_smile = [15.18, 48.91, 42.46, 4.77,
             3.87, 47.70, 0.98, 3.92,
             4.39, 49.14, 1.22, 3.92,
             8.99, 45.66, 14.57, 4.06,
             12.35, 47.82, 40.29, 3.97]

#######################################################################################
# *********  NO SMILE *********************

# BPCER (0.10%) @ APCER
bpcr01 = [58.82, 100, 100,
          63.73, 100, 100,
          71.08, 100, 100,
          79.90, 100, 100,
          74.51, 100, 100]

# BPCER (1.00%) @ APCER
bpcr1 = [33.33, 100, 100,
         26.47, 100, 0,
         22.06, 100, 100,
         42.16, 100, 100,
         53.92, 100, 100]

# BPCER (10.00%) @ APCER
bpcr10 = [9.80, 1.96, 0.98,
          0.49, 0, 0,
          0.49, 0, 52.94,
          8.82, 0.98, 0.98,
          9.80, 1.96, 52.94]

# BPCER (20.00%) @ APCER
bpcr20 = [4.41, 1.96, 0.98,
          0, 0, 0,
          0.49, 0, 52.94,
          4.41, 0, 0.98,
          2.94, 0.98, 52.94]

# *********  SMILE *********************

# BPCER (0.10%) @ APCER
bpcr01_smile = [50.00, 100, 100,
                50.00, 100, 100,
                50.00, 100, 100,
                50.00, 100, 100,
                50.00, 100, 100]

# BPCER (1.00%) @ APCER
bpcr1_smile = [50.00, 100, 9.31,
               50.00, 0.49, 100,
               49.50, 0.49, 100,
               49.50, 2.94, 6.37,
               49.50, 100, 10.78]

# BPCER (10.00%) @ APCER
bpcr10_smile = [49.50, 2.94, 5.88,
                48.52, 0, 100,
                48.52, 0, 100,
                48.52, 0.98, 6.37,
                48.52, 1.47, 10.78]

# BPCER (20.00%) @ APCER
bpcr20_smile = [48.52, 1.47, 5.88,
                48.52, 0, 100,
                48.52, 0, 100,
                48.52, 0, 6.37,
                48.52, 0.98, 10.78]

#######################################################################################

# *********  Validation *********************
models_v = ['Amsl', 'Amsl', 'Amsl', 'Amsl', 'Amsl',
           'Facemorpher', 'Facemorpher',  'Facemorpher', 'Facemorpher', 'Facemorpher',
          'OpenCV', 'OpenCV', 'OpenCV', 'OpenCV', 'OpenCV',
          'StyleGAN', 'StyleGAN', 'StyleGAN', 'StyleGAN', 'StyleGAN',
          'Webmorph', 'Webmorph', 'Webmorph', 'Webmorph', 'Webmorph']

lines_name_v = ['Amsl', 'Facemorpher', 'OpenCV', 'StyleGAN','Webmorph',
                'Amsl', 'Facemorpher', 'OpenCV', 'StyleGAN','Webmorph',
                'Amsl', 'Facemorpher', 'OpenCV', 'StyleGAN','Webmorph',
                'Amsl', 'Facemorpher', 'OpenCV', 'StyleGAN','Webmorph',
                'Amsl', 'Facemorpher', 'OpenCV', 'StyleGAN','Webmorph']


eer_validation_smile = [42.46, 50, 50, 49.51, 49.91,
                        0.98, 9.62, 10.64, 3.64, 6.26,
                        1.22, 24.82, 25.22, 10.38, 16.40,
                        14.57, 35.68, 35.02, 22.05, 28.89,
                        40.29, 49.96, 49.96, 50, 49.84]

eer_validation_no_smile = [43.17, 50, 50, 49.53, 49.91,
                           2.41, 10.02, 11.37, 4.13, 7.45,
                           3.38, 25.10, 25.79, 11.11, 17.09,
                           16.16, 35.84, 35.35, 25.53, 29.54,
                           41.15, 49.96, 49.96, 50.04, 49.84]

# Creazione del grafico a colonne con sottocategorie
ax = sns.barplot(x=models, y=eer_no_smile, hue=sub_categories_no_smile,
                 palette='pastel', saturation=0.7, alpha=0.8)

# Aggiungi il valore sulle barre
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2, height + 1, f'{height:.2f}%', ha="center", fontsize=10)

# Grandezza grafico
fig = plt.gcf()
fig.set_size_inches(15, 11)

# Aggiunta di etichette
plt.xlabel('Tests', fontsize=20)
plt.ylabel('EER (%)', fontsize=20)
plt.title("Equal Error Rate(no 'smile')", fontsize=20)

# Modifica della grandezza delle colonne
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Posiziona la leggenda fuori dal grafico
plt.legend(loc="upper right", bbox_to_anchor=(1.0, 1.16), fontsize=12)

# Visualizzazione del grafico
plt.show()


sns.lineplot(x=models_v, y=eer_validation_no_smile, hue=lines_name_v)
# Grandezza grafico
fig = plt.gcf()
fig.set_size_inches(15, 11)

# Aggiunta di etichette
plt.xlabel('Validation', fontsize=20)
plt.ylabel('EER (%)', fontsize=20)
plt.title("EER a variare del Validation (no_smile)", fontsize=20)

# Modifica della grandezza delle colonne
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Posiziona la leggenda fuori dal grafico
plt.legend(loc="upper right", bbox_to_anchor=(1.0, 1.19), fontsize=12)

# Visualizzazione del grafico
plt.show()
