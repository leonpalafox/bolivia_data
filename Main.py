import Libreria  #Importar la libreria creada
import rand
"""
La Tabla de Excel fue copiada en una hoja diferente de la que originalmente se
presento, aun dentro del mismo documento, debido a que existian problemas de 
formatoen la hoja de excel, externos a los configurados o soportados por la
funcion de lectura.
los datos en la nueva hoja no fueron alterados en ningun otro aspecto mas que
la ubicacion y el orden de las columnas.  la informacion numerica se mantiene
identica a la original.
"""


################	   Excel	     ################
#wb = openpyxl.load_workbook('/Test/Base.xlsx')  #Abrir el archivo Excel
#sheet = wb.get_sheet_by_name('Hoja1')           #Identificar la hoja de Datos
#####################################################

wb = openpyxl.load_workbook('/Test/F844C800.xlsx')  #Abrir el archivo Excel
sheet = wb.get_sheet_by_name('Hoja2') 


Min_Col     =   7
Max_Col     =   16
Min_Fil     =   2
Max_Fil     =   4500
Variable    =   3       #
Dimension   =   2       #   2   2D          -   3   3D
Cluster     =   2       #   0   DBSCAN      -   1   KMeans  -   2   MeanShift
Grafica     =   1       #   0   Variable    -   1   Cluster
Desdoblar   =   1       #   0   TSNE        -   1   DSM

Procesamiento( Min_Fil, Max_Fil,Min_Col, Max_Col, Variable, Dimension, Cluster, 
              Grafica, Desdoblar)         #Llamar la funcion principal con los             
                                          #parametros correctos