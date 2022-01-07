import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('D:/TRABAJO ARIHUS IA/PROGRAMAS CURSO IA/PARTE 2/P14-Part2-Regression/Section 8 - Polynomial Regression/Python/Position_Salaries.csv')
print("DataSet".center(150, "-"))
print(dataset)


# X = dataset.iloc[:, 1].values    # Regresa la columna 1 del data set como arreglo
# y = dataset.iloc[:, 2].values    # Regresa la columna 2 del data set como arreglo
# print("\n", X)
# print("".center(150, "-"))
# print(y)
# print("".center(150, "-"))

# Para especificar que "X" y "y" no es una matris, si no un vector hacemos lo siguiente
# Cuando obtenemos el dataset, vemos que X tiene una dimension (10,), ahora  en lugar de indicar que queremos la columna numero 1 le indicamos    " .iloc[:, 1:2].values "
#    indicamos que queremos de la columna 1 a la 2, sabiendo que con esta sintaxis no se toma la ultima columna, con esto veremos que ahora no sera un vector, si no una matris y
#    vemos ahora que el conjunto de datos tiene dos dimenciones (10, 1)

X = dataset.iloc[:, 1:2].values   # Regresa la columna 1 del data set en forma de matris con dimension (10, 1)
y = dataset.iloc[:, 2].values     # Regresa la column
print("X".center(150, "-"))
print("\n", X)
print("y".center(150, "-"))
print(y)


# Dividir el data set en conjuntos de entrenamiento y conjunto de testing
# Para este algoritmo no sera necesario realizar la division del data set en conjunto de entrenamiento y conjunto de testing porque el numero de datos es muy pequeño
#     mo tendriamos suficiente informacion para poder entrenar el modelo con un conjunto de datos tan pequeño
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Escalado de variables
# Tampoco usaremos este metodo por que el Modelo de regresion lineal multiple busca entender las relaciones no lineales, y las librerias de regresion usadas pueden a traves de
#    un parametro escalar o normaliza los datos, pero en este caso no nos hara falta, no nos hara falta porque la gracia esta en visualizar como esos datos que no son lineales
#    (forma de funcion exponencial) se traducen automaticamente a un modelo (ecuacion no lineal)

#---------------------------------------------------------------------------Regresion lineal ----------------------------------------------------------------------------------------

# Ajustar la regresion lineal con el dataset
from sklearn.linear_model import LinearRegression
# Creamos un objeto de la clase
lin_reg = LinearRegression()
lin_reg.fit(X, y)      # Para poder entrenar el modelo, al no utlizar subconjuntos de datos le pasamos el conjunto de datos "X" y "y"
print('Prediccion --lin_reg.predict(X)-- '.center(150, "-"))
print(lin_reg.predict(X))
#--------------------------------------------------------------------------Regresion polinomica------------------------------------------------------------------------------------------

# Ajustar la regresion polinomica con el dataset
# Creamos un modelo y agregamos los terminos polinomiales (incluir hasta que grado queremos la regresion)
from sklearn.preprocessing import PolynomialFeatures  # Rasgos o caracteristicas polinomiales
poly_reg = PolynomialFeatures(degree=6)  # Objeto que utlizara la funcion polinomial PolynomialFeatures, transformamos nuestra matriz de caracteristicas "X" en una "X_polinomial",
             #    para esto necesitaremos la variable "X" y ademas de tener los terminos independientes o las potencias a la 1 vamos a poder añadir las potencias de X a la 2, a la 3 o
             #    lo que nos haga falta, hasta el orden que se necesite de nuestra regresion polinomica
             # Se trendra que transformar "X" para que no solo se aun matris de caracteristicas incluyendo las variables independientes si no tambien las nuevas potencias
             #    de las variables independientes que queremos que se tomen en cuenta como parte del modelo.
# El primer parametro de "PolynomialFeatures" es el grado del polinomio (hasta que grado queremos tener las caracteristicas polinomiales de nuestra matris original).
# Si nosotros no le decimos nada por defecto genera una regresion polinomica de grado "2" que es lo miso que colocar "degree = 2", basicamente generara las caracteristicas
#     de X y sus cuadrados
# Si colocamos el "degree = " a un grado de polinomia superior de 2, osea de un polinomia de grado 3, 4, o 5, podemos ver como varia el resultado, la prediccion se puede ajustar mas,
#    podemos ver que en un grado de polinomio 2 la funcion es convexa, pero en un grado polinomico mayor la funcion ya no es convexa, se ven cambios importantes

X_poly = poly_reg.fit_transform(X)# Transformamos la variable, le decimos que al objeto "poly_reg" se encargue de transformar nuestra matriz de datos X generando no solo las
#     columnas de la misma si no sus cuadrados, ".fit_transform()" aplica los cambios al propio objeto, en este caso se aplicar el fit_transform a X, y en X_poly tendremos las nuevas
#     columnas que hagan falta para la regresion
print("X_poly".center(150, "-"))
print(X_poly)  # Nueva matriz con 3 columnas, columna 0, columna 1 y columna 2
# columna 0 = termino independiente     columna 1 = variable X   columna 2 = cuadrado

# Hasta aqui solo hemos hecho la transformacion de la matriz de caracteristicas, ahora necesitamos un nuevo objeto de regresion, no nos confundamos con regresion lineal, haremos
#   ahora la regresion polinomial

lin_reg_2 = LinearRegression()   # Usamos la liena "LinearRegression" para reliazar regresion polinomica, es el mismo modelo pero en estre caso con los datos que le vamos
                                 #    a suministrar cuando hagamos el "fit", le vamos a pasar los "X_poly" pues la regreseion que llevaemos a cabo sera una "REGRESION LINEAL POLINOMICA"
lin_reg_2.fit(X_poly, y)         # Entrenamos a "lin_reg_2" y como termino independiente le damos a "X_poly"y como variable que queremos predecir le damos el vector "y"
# Hacemos la transformacion de las variables antes de crear el modelo, de modo que ahora lo que hacemos es seleccionar el nuevo modelo de regresion lineal y que ajuste con datos
#     polinomiales, la funcion se encarga directamente de utilizar las variables independientes que le hemos indicado ya no para hacer una regresion lineal si no en este caso utlizando
#     informacion de una regresion polinomica
#--------------------------------------------------------------------------------Grafica Regresion lineal------------------------------------------------------------------------------------
# Visaulizacion de los resultados del Modelo Lineal
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
plt.scatter(X, y, color='y', label="Datos")                      # Pinta una serie de puntos donde pintaremos los datos originales en la grafica a la hora de mostrarla
plt.plot(X, lin_reg.predict(X), color='c', label="Prediccion")   # "lin_reg.predict(X)" se encargara de hacer un predict para predecir que valores segun el modelo de regresion lineal
                                                                 #    deberian tener, cual seria el sueldo esperado para el conjunto de datos "X", para los niveles del 1, 2, 3, ..., al 10
plt.scatter(6.5, lin_reg.predict([[6.5]]), color='m', label="Prediccion Sueldo")   # Prediccion de un valor a corde al nivel o puesto de un profecionista
plt.legend(loc="best")
plt.title(" Modelo de Regresion Lineal")
plt.xlabel(" Posicion del empleado")
plt.ylabel("Sueldo (en $)")
plt.ticklabel_format(useOffset=False, style='plain')
#plt.show()

#-------------------------------------------------------------------------------Grafica Regresion polinomica----------------------------------------------------------------------------------
# Visualizacion de los resultados del modelo polinomico

# Para hacer mas precisa la prediccion, podemos poner valores intermedios entre digamos, el 2 y el 3, 5 y 6, etc, porque solo hemos indicado 10 puntos y matplotlib trasa segmentos de
#   lineas rectas entre esos 10 puntos que pusimos, para solucionarlo lo que hacemos es en lugar de pintar esta funcion continua a ese modelo de regresion lineal multiple como trosos de rectas
#   debemos de indicar mas puntos intermedios, crear una secuencia de valores, que empiece en 1 y acabe en 10, pero en lugar de tener valores discretos (1 en 2, 1 en 2, 1 en 3, etc),
#   puede tomar valores intermedios, total nuestro modelo de regresion multiple no tiene ningun efecto al poder calcula los valores intermedios
X_grid = np.arange(min(X), max(X), 0.1)  # Nos creara una secuencia de datos entre el valor minimo y maximo que le especifiquemos, y con el intervalo del salto de un numero a otro que se
                                         #      indique, desde el minimo de conjunto X (min(X)), hasta el maximo del conjunto X (max(X)), de intervalos de 0.1
# Dada esta cuadricula tenemos la matriz de los elementos, lo unico que tenemos que hacer es colocarla en el tamaño que toca, es decir ahora es un vector fila, para colocarlo en vector
#    columna lo que toca hacer es usar "reshape" para redimencionar. Le indicamos que el numero de filas para cambiar a vector columna
X_grid = X_grid.reshape(len(X_grid), 1)   #  Redimencionamos un vector en fila a un vector en columna, definimos la longitud de la columna como X_grid, y definimos como una sola columna
print("X_grid con valores intermedios".center(150, "-"))
print(X_grid)
plt.subplot(1,2,2)
plt.scatter(X, y, color='y', label="Datos")      # Pinta una serie de puntos donde pintaremos los datos originales en la grafica a la hora de mostrarla
#plt.plot(X_grid, lin_reg_2.predict([[X_poly]]), color='b', label="Prediccion")
# "lin_reg_2.predict(X)" se encargara de hacer un predict para predecir que valores segun el modelo de
#     regresion polinomica deberian tener, cual seria el sueldo esperado para el conjunto de datos "X",
#     para los niveles del 1, 2, 3, ..., al 10
# Al colocar a "X_grid" en lugar de "X" notaremos una mejor resolucion y
# plt.plot(X, lin_reg_2.predict([[poly_reg.fit_transform(X)]]), color='b', label="Prediccion")    # Es lo mismo que la de arriba
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='c', label="Prediccion")    # Es lo mismo que la de arriba pero ya optimizado
plt.scatter(6.5, lin_reg_2.predict(poly_reg.fit_transform([[6.5]])), color='m', label="Prediccion Sueldo")    # Prediccion de un valor a corde al nivel o puesto de un profecionista
plt.legend(loc="best")
plt.title(" Modelo de Regresion Polinomico")
plt.xlabel(" Posicion del empleado")
plt.ylabel("Sueldo (en $)")
plt.ticklabel_format(useOffset=False, style='plain')
plt.show()
#------------------------------------------------------------------- Prediccion en el modelo de regresion lineal --------------------------------------------------------------------------
# Prediccion de nuestro modelo
print("Prediccion en un modelo lineal".center(150, "-"))
print(lin_reg.predict([[6.5]]))   # Forma de poder predecir un valor a corde al nivel o puesto de un profecionista
#------------------------------------------------------------------- Prediccion en el modelo de regresion polinomica --------------------------------------------------------------------------
# Prediccion de nuestro modelo polinomico
print("Prediccion en un modelo polinomico".center(150, "-"))
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))   # Forma de poder predecir un valor a corde al nivel o puesto de un profecionista