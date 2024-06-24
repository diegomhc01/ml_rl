import pickle
import warnings
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings("ignore")

tipomodelo = 2
if tipomodelo==1:
    regr = pickle.load(open('modelo.p', 'rb'))
    #para un artículo de x palabras #nos devuelve una predicción de y “Shares” 
    resultado = regr.predict([[1000]])
    print(f"Para 1000 palabaras, obtengo {int(resultado)} \"compartidos\"")

    resultado = regr.predict([[2000]])
    print(f"Para 2000 palabaras, obtengo {int(resultado)} \"compartidos\"")

    resultado = regr.predict([[3000]])
    print(f"Para 3000 palabaras, obtengo {int(resultado)} \"compartidos\"")

    resultado = regr.predict([[4000]])
    print(f"Para 4000 palabaras, obtengo {int(resultado)} \"compartidos\"")

    resultado = regr.predict([[5000]])
    print(f"Para 5000 palabaras, obtengo {int(resultado)} \"compartidos\"")


    valores_x = np.array([1000,2000,3000,4000,5000])
    y = []
    for x in valores_x:    
        y.append(int(regr.predict([[x]])))
    str = "Para 1000 palabaras, obtengo 16897 \"compartidos\" \n"
    str += "Para 2000 palabaras, obtengo 22595 \"compartidos\" \n"
    str += "Para 3000 palabaras, obtengo 28293 \"compartidos\" \n"
    str += "Para 4000 palabaras, obtengo 33990 \"compartidos\" \n"
    str += "Para 5000 palabaras, obtengo 39688 \"compartidos\""
    fig, axs = plt.subplots()
    axs.plot(valores_x, y)
    plt.title(str)
    plt.show()
elif tipomodelo==2:
    XY_train = 2000
    z_train = 10+4+6
    regr = pickle.load(open('modelo2.p', 'rb'))
    resultado = regr.predict([[2000, 10+4+6]])

    fig = plt.figure()
    ax = Axes3D(fig)
    figura = plt.figure(figsize=(7,7))
    ax = plt.subplot(111, projection="3d")
    
    # Creamos una malla, sobre la cual graficaremos el plano
    xx, yy = np.meshgrid(np.linspace(0, 3500, num=10), np.linspace(0, 60, num=10))
    
    # calculamos los valores del plano para los puntos x e y
    nuevoX = (regr.coef_[0] * xx)
    nuevoY = (regr.coef_[1] * yy) 
        
    # calculamos los correspondientes valores para z. Debemos sumar el punto de intercepción
    z = (nuevoX + nuevoY + regr.intercept_)
    
    # Graficamos el plano
    ax.plot_surface(xx, yy, z, alpha=0.2, cmap='hot')
    
    # Graficamos en azul los puntos en 3D
    
    ax.scatter(nuevoX[:, 0], XY_train[:, 1], z, c='blue',s=30)
    
    # Graficamos en rojo, los puntos que 
    ax.scatter(nuevoX[:, 0], XY_train[:, 1], z, c='red',s=40)
    
    # con esto situamos la "camara" con la que visualizamos
    ax.view_init(elev=30., azim=65)
            
    ax.set_xlabel('Cantidad de Palabras')
    ax.set_ylabel('Cantidad de Enlaces,Comentarios e Imagenes')
    ax.set_zlabel('Compartido en Redes')
    ax.set_title('Regresión Lineal con Múltiples Variables')
    plt.show()
