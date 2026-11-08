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
import data_analysis.armado_data_pca as reducido

#%%

# =============================================================================
#   ANÁLISIS PCA - todos los objetos
# =============================================================================

# Todos los elementos que aparecen en todas las muestras
elem_name = ['Fe', 'Ca', 'Ti', 'Cu', 'Zn', 'Sr', 'Mn', 'Zr', 'Rb', 'K']

df_pca_ocres = pd.read_csv('data_analysis/data_PCA/df_pca_2nd_sel.csv', index_col=0)

eigval, eigvec, pc_scores, cumulative_exp_var = PCA(df_pca_ocres, elem_name)

#%%
### ------------ Veamos la varianza acumulada

plt.plot(np.array(range(len(elem_name)))+1, cumulative_exp_var, marker='*',color='red', ls=':')
plt.grid()
plt.title('Var explicada acumulada')
plt.show()


#%%

# Componentes principales a comparar:
n = 1
m = 2

plt.figure(figsize=(7,5.5))

plt.scatter(pc_scores.iloc[0:2][n-1], pc_scores.iloc[0:2][m-1], label='OLP1', c='blue')
plt.scatter(pc_scores.iloc[2:6][n-1], pc_scores.iloc[2:6][m-1], label='OLP2', c='red')
plt.scatter(pc_scores.iloc[6:8][n-1], pc_scores.iloc[6:8][m-1], label='OLP3', c='green')
plt.scatter(pc_scores.iloc[8:9][n-1], pc_scores.iloc[8:9][m-1], label='OLP4', c='gold')
plt.scatter(pc_scores.iloc[9:12][n-1], pc_scores.iloc[9:12][m-1], label='OLP5', c='cyan')
plt.scatter(pc_scores.iloc[12:13][n-1], pc_scores.iloc[12:13][m-1], label='OLP6', c='magenta')
plt.scatter(pc_scores.iloc[13:15][n-1], pc_scores.iloc[13:15][m-1], label='OLP7', c='lime')
plt.scatter(pc_scores.iloc[15:18][n-1], pc_scores.iloc[15:18][m-1], label='OLP8', c='gray')
plt.scatter(pc_scores.iloc[18:19][n-1], pc_scores.iloc[18:19][m-1], label='OLP9', c='purple')
plt.scatter(pc_scores.iloc[19:21][n-1], pc_scores.iloc[19:21][m-1], label='OLP10', c='orange')
plt.scatter(pc_scores.iloc[23:24][n-1], pc_scores.iloc[23:24][m-1], label='OLP11', c='brown')
plt.scatter(pc_scores.iloc[21:23][n-1], pc_scores.iloc[21:23][m-1], label='OLP12', c='pink')


for i in range(len(elem_name)):
  plt.scatter(eigvec[i][n-1], eigvec[i][m-1], color='black')
  plt.text(eigvec[i][n-1]-0.3, eigvec[i][m-1]+0.1, elem_name[i], color='black')

plt.xlabel('PC '+str(n))
plt.ylabel('PC '+str(m))
plt.grid()
plt.legend(bbox_to_anchor=(1, 0.5), loc='center left')
plt.tight_layout()
plt.title('Todos los objetos - todos los elementos')

#%%

# =============================================================================
#       ANÁLISIS PCA - Sin los objetos con soporte
# =============================================================================

eigval, eigvec, pc_scores, cumulative_exp_var = PCA(reducido.df_pca_sin3, reducido.elem_name_sin3)

#%%
### ------------ Veamos la varianza acumulada

plt.plot(np.array(range(len(reducido.elem_name_sin3)))+1, cumulative_exp_var, marker='*',color='red', ls=':')
plt.grid()
plt.title('Var explicada acumulada')
plt.show()


#%%

# Componentes principales a comparar:
n = 3
m = 4

plt.figure(figsize=(7,5.5))

plt.scatter(pc_scores.iloc[0:1][n-1], pc_scores.iloc[0:1][m-1], label='OLP4', c='gold')
plt.scatter(pc_scores.iloc[1:4][n-1], pc_scores.iloc[1:4][m-1], label='OLP5', c='cyan')
plt.scatter(pc_scores.iloc[4:5][n-1], pc_scores.iloc[4:5][m-1], label='OLP6', c='magenta')
plt.scatter(pc_scores.iloc[5:7][n-1], pc_scores.iloc[5:7][m-1], label='OLP7', c='lime')
plt.scatter(pc_scores.iloc[7:10][n-1], pc_scores.iloc[7:10][m-1], label='OLP8', c='gray')
plt.scatter(pc_scores.iloc[10:11][n-1], pc_scores.iloc[10:11][m-1], label='OLP9', c='purple')
plt.scatter(pc_scores.iloc[11:13][n-1], pc_scores.iloc[11:13][m-1], label='OLP10', c='orange')
plt.scatter(pc_scores.iloc[13:14][n-1], pc_scores.iloc[13:14][m-1], label='OLP11', c='brown')
plt.scatter(pc_scores.iloc[14:16][n-1], pc_scores.iloc[14:16][m-1], label='OLP12', c='pink')


for i in range(len(reducido.elem_name_sin3)):
  plt.scatter(eigvec[i][n-1], eigvec[i][m-1], color='black')
  plt.text(eigvec[i][n-1]-0.4, eigvec[i][m-1], reducido.elem_name_sin3[i], color='black')

plt.xlabel('PC '+str(n))
plt.ylabel('PC '+str(m))
plt.grid()
plt.legend(bbox_to_anchor=(1, 0.5), loc='center left')
plt.tight_layout()
plt.title('Objetos sin soporte - todos los elementos')

#%%

# =============================================================================
#       ANÁLISIS PCA - Sin los objetos con soporte, sin Rb y Zr
# =============================================================================

eigval, eigvec, pc_scores, cumulative_exp_var = PCA(reducido.df_pca_sin3_v2, reducido.elem_name_sin3_v2)

#%%
### ------------ Veamos la varianza acumulada

plt.plot(np.array(range(len(reducido.elem_name_sin3_v2)))+1, cumulative_exp_var, marker='*',color='red', ls=':')
plt.grid()
plt.title('Var explicada acumulada')
plt.show()

#%%

# Componentes principales a comparar:
n = 3
m = 4


plt.scatter(pc_scores.iloc[0:1][n-1], pc_scores.iloc[0:1][m-1], label='OLP4', c='gold')
plt.scatter(pc_scores.iloc[1:4][n-1], pc_scores.iloc[1:4][m-1], label='OLP5', c='cyan')
plt.scatter(pc_scores.iloc[4:5][n-1], pc_scores.iloc[4:5][m-1], label='OLP6', c='magenta')
plt.scatter(pc_scores.iloc[5:7][n-1], pc_scores.iloc[5:7][m-1], label='OLP7', c='lime')
plt.scatter(pc_scores.iloc[7:10][n-1], pc_scores.iloc[7:10][m-1], label='OLP8', c='gray')
plt.scatter(pc_scores.iloc[10:11][n-1], pc_scores.iloc[10:11][m-1], label='OLP9', c='purple')
plt.scatter(pc_scores.iloc[11:13][n-1], pc_scores.iloc[11:13][m-1], label='OLP10', c='orange')
plt.scatter(pc_scores.iloc[13:14][n-1], pc_scores.iloc[13:14][m-1], label='OLP11', c='brown')
plt.scatter(pc_scores.iloc[14:16][n-1], pc_scores.iloc[14:16][m-1], label='OLP12', c='pink')


for i in range(len(reducido.elem_name_sin3_v2)):
  plt.scatter(eigvec[i][n-1], eigvec[i][m-1], color='black')
  plt.text(eigvec[i][n-1]-0.5, eigvec[i][m-1], reducido.elem_name_sin3_v2[i], color='black')

plt.xlabel('PC '+str(n))
plt.ylabel('PC '+str(m))
plt.grid()
plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
plt.tight_layout()
plt.title('Objetos sin soporte - sin Rb y Zr')

#%%

# =============================================================================
#       ANÁLISIS PCA - Sin los objetos con soporte
# =============================================================================

eigval, eigvec, pc_scores, cumulative_exp_var = PCA(reducido.df_pca_sin3_v3, reducido.elem_name_sin3_v3)

#%%
### ------------ Veamos la varianza acumulada

plt.plot(np.array(range(len(reducido.elem_name_sin3_v3)))+1, cumulative_exp_var, marker='*',color='red', ls=':')
plt.grid()
plt.title('Var explicada acumulada')
plt.show()

#%%

# Componentes principales a comparar:
n = 3
m = 4


plt.scatter(pc_scores.iloc[0:1][n-1], pc_scores.iloc[0:1][m-1], label='OLP4', c='gold')
plt.scatter(pc_scores.iloc[1:4][n-1], pc_scores.iloc[1:4][m-1], label='OLP5', c='cyan')
plt.scatter(pc_scores.iloc[4:5][n-1], pc_scores.iloc[4:5][m-1], label='OLP6', c='magenta')
plt.scatter(pc_scores.iloc[5:7][n-1], pc_scores.iloc[5:7][m-1], label='OLP7', c='lime')
plt.scatter(pc_scores.iloc[7:10][n-1], pc_scores.iloc[7:10][m-1], label='OLP8', c='gray')
plt.scatter(pc_scores.iloc[10:11][n-1], pc_scores.iloc[10:11][m-1], label='OLP9', c='purple')
plt.scatter(pc_scores.iloc[11:13][n-1], pc_scores.iloc[11:13][m-1], label='OLP10', c='orange')
plt.scatter(pc_scores.iloc[13:14][n-1], pc_scores.iloc[13:14][m-1], label='OLP11', c='brown')
plt.scatter(pc_scores.iloc[14:16][n-1], pc_scores.iloc[14:16][m-1], label='OLP12', c='pink')


for i in range(len(reducido.elem_name_sin3_v3)):
  plt.scatter(eigvec[i][n-1], eigvec[i][m-1], color='black')
  plt.text(eigvec[i][n-1]-0.5, eigvec[i][m-1], reducido.elem_name_sin3_v3[i], color='black')

plt.xlabel('PC '+str(n))
plt.ylabel('PC '+str(m))
plt.grid()
plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
plt.tight_layout()
plt.title('Objetos sin soporte - 3ra selección de elementos')