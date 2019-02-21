import gzip
import os
import re
import threading

# Variables Globales
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SISPI_DIR = os.path.join(BASE_DIR, 'data\sispi')
SISPI_FILE_REG_EXPRESS = "wrf_output_(\d{1,2})_(\d{1,2})"
CMORPH_REG_EXPRESS = "cmorph_(\d{1,2})_(\d{1,2})"
SISPI_FILES_QUEUE = []

class Files():
        def WriteGZ(self):
                text = 'A nine mile walk is no joke'
                text2 =  'A nine mile walk is no joke 2'
                with gzip.open('out.log.gz', mode='wb') as zfile:
                        zfile.write(text.encode('utf-8'))
                        
                with gzip.open('out.log.gz', mode='wb') as z:
                        z.write(text2.encode('utf-8'))
                   
def sispi_files_queue():        
        # Esta funcion añade a la variable global SISPI_FILES_QUEUE
        # los ficheros contenidos en el directorio SISPI_DIR
        for ruta, directorios, archivos in os.walk(SISPI_DIR, topdown=True):                 
                for archivo in archivos:
                        SISPI_FILES_QUEUE.append(archivo)                        

def eval_regular_expresion():
        """
        if re.match(SISPI_REG_EXPRESS, "wrf_output_25_36"):
                print("cierto")
        else:
                print("falso")
        """
        mo = re.match(SISPI_FILE_REG_EXPRESS, "wrf_output_25_3")
        # Esta funcion devuelve una tupla con los valores que
        # cambian en la expresion regular en formato int
        print(mo.groups())
       


print("Soy el hilo principal")

for i in range(0, 10):
        t = RNA1(i)
        t.start()
        t.join()

#rna1_thread = RNA1()


#sispi_files_queue()

#eval_regular_expresion()

