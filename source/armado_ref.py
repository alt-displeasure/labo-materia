import pandas as pd

df = pd.read_csv('source/tabla lineas_v2.csv', sep=';' , decimal=',')

clr = pd.read_csv('source/colors.csv')
colors=clr['Color']

#---- armamos la referencia ----

ref = {}

for i in range(len(df)): #Itero sobre todos los elementos
  A = df[df['Atomic#']==i+1] #selecciono la fila del i-ésimo elemento
  symbol = df['Symbol'][i] #guardo su símbolo
 
  lines = []
  intensity = []
  color=colors[i]
  j = 4
  while j < 39:
    lines.append(A.iloc[:, j][i])
    intensity.append(A.iloc[:, j+1][i])
    j = j + 2

  d = pd.DataFrame({'Lines' : lines , 'Intensity' : intensity, 'Color' : color})

  ref[symbol] = d

  del A
  