import pickle
import warnings
import numpy as np 

warnings.filterwarnings("ignore")
#-122.910000,39.050000,27.000000,789.000000,208.000000,295.000000,108.000000,3.766700,95000.000000
datos = [-119.700000,36.300000,10.000000,956.000000,201.000000,693.000000,220.000000,2.289500]
regr = pickle.load(open('modelo.p', 'rb'))
#para un artículo de x palabras #nos devuelve una predicción de y “Shares” 
resultado = regr.predict([datos])
print(resultado)