"""

       FUNCIONES ÚTILES
       
"""

import numpy as np
import pandas as pd
import scipy as sc

# ---------------------------------------
# Minimizar distancia a un punto

def minimize(array, x):
    
    """
    Encontrar el punto del array que minimiza la distancia al x dado
    """
    
    index = np.argmin(np.abs(array - x))
    return index, array[index]

# ---------------------------------------
# Calibrar el espectro, de canales a eV

def calib(x):
  m = 20.2627 #ev per channel
  b = -7.6785 #eV
  return x*m + b

# ---------------------------------------
# Seleccionar elemento de un espectro (2 versiones de la función)

def select_element2(df, element):
  """

  df: un dataframe de Pandas, con 'Energy' e 'Intensity' sus dos entradas.

  element: la energía del elemento que queremos encontrar, dada en keV

  Busca los picos con find_peaks, y devuelve Las coordenadas X e Y del gráfico
  de la medición donde se encuentra el elemento seleccionado.

  """

  X = df['Energy']
  Y = df['Intensity']

  # Encuentra los picos
  p, peaks = sc.signal.find_peaks(Y, height = 100, distance=18)

  # Les asigna energías
  energies = np.array([X[i] for i in p])

  # Encuentra la más cercana
  index, energy = minimize(energies, element)

  coord = p[index]

  X_elem = X[coord]
  Y_elem = Y[coord]

  # Le agrego una tolerancia
  if np.abs(X_elem - element) < 0.4:
    return X_elem, Y_elem
  else:
    return np.nan, np.nan


# ---------------------------------------


def select_element(peaks, energies, element ,tol):
  """

  E, I: listas de energía e intensidad

  element: un string del símbolo del elemento que queremos

  Busca los picos con find_peaks, y devuelve Las coordenadas X e Y del gráfico
  de la medición donde se encuentra el elemento seleccionado.

  """

  # En estas listas vamos a guardar las energías que corresponderían a las energías de los picos del elemento en cuestión
  E_elem = []
  I_elem = []
  ind_elem = []

  for e in ref[element]['Lines']:
    #Para cada valor de energía del elemento, encuentro el pico más cercano
    index, energy = minimize(energies, e)
    if np.abs(energy-e)<tol: #Si el pico está más cerca de ese valor de energía que nuestra tolerancia, lo guardamos
      E_elem.append(energy)
      I_elem.append(peaks['peak_heights'][index])
      ind_elem.append(np.argwhere(ref[element]['Lines']==e)[0][0])
      
  return E_elem, I_elem, ind_elem


# ---------------------------------------
# Hallar la linea principal de un elemento

def abundant_line(ref_df, element, liminf, limsup):
    
    """
    ACÁ NO DEBERÍA ACOTAR LA ENERGÍA EN VEZ DE LA INTENSIDAD?? CHEQUEAR
    """
    
    j=0
    while liminf<=ref_df[element]['Intensity'][j]<=limsup: 
        j=j+1
    el = ref_df[element]['Lines'][j]
    return el


# ---------------------------------------
# Armado del DataFrame para hacer PCA


def df_pca(names,data,elem_list, liminf, limsup):
    
    """
    
    names: lista con nombres de los archivos con los espectros
    data: contenido de los espectros; DataFrame con data['name'] = {'Energy': energy, 'Intensity': intensity}
    elem_list: los elementos a seleccionar
    liminf, limsup: límites inferior y superior de la intensidad de la linea 
    
    """
    
    I_elem = {}
    for i in range(len(elem_list)):
        I = []
        for j in names:

            # Busco el rodio para normalizar
            X_Rh, Y_Rh = select_element2(data[j], abundant_line('Rh', liminf, limsup))

            # Selecciono las coordenadas del pico del elemento buscado
            X, Y = select_element2(data[j], abundant_line(elem_list[i], liminf, limsup))
            # Guardo la inensidad normalizada
            I.append(Y/Y_Rh)

        # Añado las intensidades al diccionario, con clave símbolo del elemento
        I_elem[elem_list[i]] = I

    return pd.DataFrame(I_elem, index=names)

# ---------------------------------------
# Hacer PCA

def PCA(df, elem_name):
  """
  df: dataframe con las proporciones de los elementos, cuyas filas son las mediciones y las columnas los elementos

  elem_name: lista de strings de los elementos
  """   
  for i in elem_name:
    #Normalizamos
    mean = np.mean(df[i])
    std = np.std(df[i])
    df[i] = (df[i] - mean)/std
    
    
  # Armamos la matriz de covarianzas
  cov_matrix = np.cov(np.transpose(df))
  
  # Buscamos autovectores y autovalores
  eigval1, eigvec1 = np.linalg.eigh(cov_matrix)
  
  # Ordenamos autovalores por módulo decreciente y los autovectores también
  eigval=sorted(eigval1, key=abs)
  eigval=eigval[::-1]
  eigvec=[]
  for i in range(len(eigvec1)):
      j=np.argwhere(eigval1==eigval[i])[0][0]
      v=eigvec1[:,j]
      eigvec.append(v)
      
  #La matriz de autovectores está traspuesta
  eigvec=np.array(eigvec)


  # Calculamos el principal component score para cada medición
  pc_score = np.transpose(np.matmul(eigvec , np.transpose(df)))
  
  # Calculamos la varianza acumulada
  exp_var = eigval/np.sum(eigval)
  cumulative_exp_var = []
  for i in range(len(eigval)):
    sum = 0
    for j in range (i+1):
      sum = sum + exp_var[j]
    cumulative_exp_var.append(sum)

  return eigval,eigvec, pc_score, cumulative_exp_var
    