from threads import Thread_RNA, Thread_Sispi_Files
from os import path, listdir
import re
import threading
from files import*
import bz2

# Variables Globales

BASE_DIR = path.dirname(path.dirname(path.abspath(__file__)))

CMORPH = {
    "TYPE": "CMORPH",
    "DIR": path.join(BASE_DIR, 'data/cmorph/'),
    "REG_EXPRESS": "CMORPH_V1.0_ADJ_8km-30min_(\d{1,6}).tar",
    "FILES_QUEUE": [],
    "FILES": [],
    "FILE": "",
}

SISPI_DIR = path.join(BASE_DIR, 'data/sispi/')
CMORPH_DIR = path.join(BASE_DIR, 'data/cmorph/')
SISPI_FILE_REG_EXPRESS = "wrf_output_(\d{1,2})_(\d{1,2})"
CMORPH_REG_EXPRESS = "CMORPH_V1.0_ADJ_8km-30min_(\d{1,6})"
CMORPH_TAR_FILES_QUEUE = []
CMORPH_BZ2_FILES_QUEUE = []
SISPI_FILES_QUEUE = []
THREADS_LIST = []
RNA_LIST = []
# Variables de DATA_LIST. Tienen la forma [predictores, objetivo],
# o sea [sispi, cmorph]
data_1 = []  
data_2 = []
DATA_LIST = [data_1, data_2] # Arreglo de datos 


cmorph = ManagerCMORPHFiles(CMORPH)
cmorph.files_queue() # Cargar la cola de ficheros comprimidos]
cmorph.uncompres_cmorph_tar_file()

#cmorph.delete_uncompres_cmorph_tar_file_folder(dirpath)
#print(CMORPH["FILES_QUEUE"])
#cmorph_files_queue(CMORPH)
#files_queue("sispi", SISPI_FILES_QUEUE, SISPI_DIR, CMORPH_DIR)


#var = CMORPH_DIR + CMORPH_TAR_FILES_QUEUE[1]
#var2 = CMORPH_DIR + '/201701/'



"""
Seudocodigo del algoritmo

creada una variable global que tiene la forma [var1, var2] donde
var1 = [x1, y1] y var2 = [x2, y2] y:
x = valor de las variables predictoras,
y = valor objetivo  

creada una variable que es una lista de threads

dado:
- una lista de rna diferentes
- una lista de ficheros wrf ordenados
- una lista de ficheros cmorph ordenados

construir los siguientes threads:
t1: se encarga de leer el archivo wrf 
t2: se encarga de leer el archivo cmorph

si var1 = []
  asignar el dataset leido a var1 # Significa que las RNA estan trabajando con el var2
sino
  asignar el dataset leido a var2

crear diferentes threads (t3, t4, t5, t6, t7, tn) para 1 rna de de la lista de rna
e ir añadiendolos a la lista. Los parametros del constructor del thread seran:
- la rna
- var1 o var2, en dependencia de quien este lleno
- el id del thread. Es un numero entero

lanzar nuevamente t1 y t2 para que se vayan leyendo en paralelo en lo que corren las rna

correr los hilos
hacer join a los hilos.

Esto se va ha hacer para cada n rna de la lista de rna
"""
"""
for i in range(3):
    t = Thread_RNA(i, SISPI_FILES_QUEUE, val)
    t.start()
    t.join()

for i in val:
    print(i)
"""