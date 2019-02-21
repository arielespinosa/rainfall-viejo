import threading

class Thread_RNA(threading.Thread):

        def __init__(self, num, dat, val):
                threading.Thread.__init__(self)
                self.num = num
                self.dat = dat
                self.val = val

        def run(self):
                """ 
                Aqui se le pasa la matriz de datos se 
                almacena en DATA_LIST. Si el data1 es el que esta en None significa
                que las RNA estan entrenandose con el data2 por lo que se almacena
                el valor en data1
                """
                return 0


class Thread_Sispi_Files(threading.Thread):

        def __init__(self, num, DATA_LIST, sispi_file):
                threading.Thread.__init__(self)
                self.num = num
                self.DATA_LIST = DATA_LIST
                self.sispi_file = sispi_file                

        def run(self):
                """ 
                Aqui se lee el fichero sispi_file y la matriz de datos se 
                almacena en DATA_LIST. Si el data1 es el que esta en None significa
                que las RNA estan entrenandose con el data2 por lo que se almacena
                el valor en data1
                """
                return 0

class Thread_Cmorph_Files(threading.Thread):

        def __init__(self, num, DATA_LIST, cmorph_file):
                threading.Thread.__init__(self)
                self.num = num
                self.DATA_LIST = DATA_LIST
                self.sispi_file = sispi_file
                
        def run(self):
                """ 
                Aqui se lee el fichero cmorph_file y la matriz de datos se 
                almacena en DATA_LIST. Si el data1 es el que esta en None significa
                que las RNA estan entrenandose con el data2 por lo que se almacena
                el valor en data1
                """