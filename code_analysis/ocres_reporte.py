# =============================================================================
#       PREÁMBULO
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
# import scipy as sc
import pandas as pd
# import matplotlib.image as mpimg
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler

# import os

from source.functions import *

#%%

# =============================================================================
#   ANÁLISIS PCA
# =============================================================================

# Todos los elementos que aparecen en todas las muestras
elem_name = ['Fe', 'Ca', 'Ti', 'Cu', 'Zn', 'Sr', 'Mn', 'Zr', 'Rb', 'K']

df_ocres = pd.read_csv('data_analysis/data_PCA/df_pca_2nd_sel.csv', index_col=0)

eigval, eigvec, pc_scores, cumulative_exp_var = PCA(df_ocres, elem_name)

#%%
### ------------ Veamos la varianza acumulada

plt.plot(np.array(range(len(elem_name)))+1, cumulative_exp_var, marker='*',color='red', ls=':')
plt.grid()
plt.title('Var explicada acumulada')
plt.show()


print(cumulative_exp_var[5])
print(cumulative_exp_var[6])

#%%

# Componentes principales a comparar:
n = 2
m = 5

plt.scatter(pc_scores.iloc[0:2][n-1], pc_scores.iloc[0:2][m-1], label='OPL1', c='blue')
plt.scatter(pc_scores.iloc[2:6][n-1], pc_scores.iloc[2:6][m-1], label='OPL2', c='red')
plt.scatter(pc_scores.iloc[6:8][n-1], pc_scores.iloc[6:8][m-1], label='OPL3', c='green')
plt.scatter(pc_scores.iloc[8:9][n-1], pc_scores.iloc[8:9][m-1], label='OPL4', c='gold')
plt.scatter(pc_scores.iloc[9:12][n-1], pc_scores.iloc[9:12][m-1], label='OPL5', c='cyan')
plt.scatter(pc_scores.iloc[12:13][n-1], pc_scores.iloc[12:13][m-1], label='OPL6', c='magenta')
plt.scatter(pc_scores.iloc[13:15][n-1], pc_scores.iloc[13:15][m-1], label='OPL7', c='lime')
plt.scatter(pc_scores.iloc[15:18][n-1], pc_scores.iloc[15:18][m-1], label='OPL8', c='gray')
plt.scatter(pc_scores.iloc[18:19][n-1], pc_scores.iloc[18:19][m-1], label='OPL9', c='purple')
plt.scatter(pc_scores.iloc[19:21][n-1], pc_scores.iloc[19:21][m-1], label='OPL10', c='orange')
plt.scatter(pc_scores.iloc[21:23][n-1], pc_scores.iloc[21:23][m-1], label='OPL12', c='pink')
plt.scatter(pc_scores.iloc[23:24][n-1], pc_scores.iloc[23:24][m-1], label='OPL11', c='brown')

for i in range(len(elem_name)):
  plt.scatter(eigvec[i][n-1], eigvec[i][m-1], color='black')
  plt.text(eigvec[i][n-1]-0.5, eigvec[i][m-1], elem_name[i], color='black')

plt.xlabel('PC '+str(n))
plt.ylabel('PC '+str(m))
plt.grid()
plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
plt.tight_layout()