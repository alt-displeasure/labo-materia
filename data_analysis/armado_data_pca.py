# =============================================================================
#       PREÁMBULO
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os


from source.functions import *

#%%

# =============================================================================
#       CARGA DE DATOS
# =============================================================================

data = {}
names=[]
for file in glob.glob("data_analysis/ocres/*.*"):

  #Tomamos solo los csv
  if os.path.basename(file)[-1]=='v':

    #Guardamos el nombre de la medición
    name = os.path.splitext(os.path.basename(file))[0].split(".")[0]
    names.append(name)

    df = pd.read_csv(file, skiprows=20)

    #Calibrar y pasar a keV
    e = calib(df['Channel#'])/1000

    I = df['Intensity']

    #Renombro
    d = pd.DataFrame({'Energy' : e, 'Intensity' : I})

    data[name] = d
    
# Ordenamos la lista 

names=sorted(names, key=lambda x: (
    int(x.split()[0][3:]),  # primer número
    int(x.split()[1])       # segundo número
))
    
#%%

# Los picos que vamos a usar para identificar cada elemento, y algunos secundarios para tener de referencia
# Todo está en keV
Pb = 10.551
Pb2 = 12.614
Hg = 9.989
Fe = 6.405
Fe2 = 7.059
Ca = 3.692
Ti = 4.512
Cu = 8.046
Ni = 7.48
Zn = 8.637
Zn2 = 9.57
Rh = 20.216
As = 10.543
Sr = 14.165
Mn = 5.900
Fe_bis = Fe*2
Zr = 15.775
Rb = 13.396
K = 3.314

#%%

# =============================================================================
#       ARMADO DATAFRAME PARA PCA
# =============================================================================
    
# Todos los elementos encontrados 
elem = [Fe, Ca, Ti, Cu, Zn, Sr, Mn, Zr, Rb, K]
elem_name = ['Fe', 'Ca', 'Ti', 'Cu', 'Zn', 'Sr', 'Mn', 'Zr', 'Rb', 'K']

# Seleccionamos las lineas
I_elem = {}

for i in range(len(elem)):
  I = []

  for j in names[8:]:

    # Busco el rodio para normalizar
    X_Rh, Y_Rh = select_element2(data[j], Rh)

    # Selecciono las coordenadas del pico del elemento buscado
    X, Y = select_element2(data[j], elem[i])
    # Guardo la inensidad normalizada
    I.append(Y/Y_Rh)

  # Añado las intensidades al diccionario, con clave símbolo del elemento
  I_elem[elem_name[i]] = I

df_pca = pd.DataFrame(I_elem, index=names[8:])
