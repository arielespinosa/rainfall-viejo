
""" uncompress.py
    Clase utilitaria para descompactar tanto tar.gz
  
"""

import sys
import zipfile
import tarfile
import os
import os.path

class UnCompress():
    # file - str. Path del archivo
    # dir - str. Path donde descompactar el file
    def uncompress(self, file, dir):
        if os.path.exists(dir):
            # si existe lo borramos
            for root, dirs, files in os.walk(dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            if os.path.exists(dir): 
                os.rmdir(dir)
            # creamos el dir target
            os.mkdir(dir)
            # si es un tar.gz
            if file.endswith(".tar.gz"):
                # extremadamente trivial
                tar = tarfile.open(file, mode='r:gz')
                tar.extractall(dir)
            # sino, si es un zip (mas complejo)
            elif file.endswith(".zip"):
                # creamos la estructura de directorio
                # donde copiar los archivos
                self._createstructure(file, dir)
                zf = zipfile.ZipFile(file)
                # para cada archivo que lista el zip
                for i, name in enumerate(zf.namelist()):
                    # si no termina con '/' asumimos que es un archivo
                    if not name.endswith('/'):
                        # creamos el archivo
                        outfile = open(os.path.join(dir, name), 'wb')
                        # copiamos el contenido
                        outfile.write(zf.read(name))
                        outfile.flush()
                        outfile.close()

    def _createstructure(self, file, dir):
        self._makedirs(self._listdirs(file), dir)

    def _makedirs(self, directories, basedir):
        """ crea los directorios """
        for dir in directories:
            curdir = os.path.join(basedir, dir)
            if not os.path.exists(curdir):
                os.mkdir(curdir)

    def _listdirs(self, file):
        """ crea una lista de los directorios
            que hay que crear
        """
        zf = zipfile.ZipFile(file)
        dirs = []
        for name in zf.namelist():
            if name.endswith('/'):
                dirs.append(name)
        # la ordenamos de menor a mayor
        dirs.sort()
        return dirs

f = UnCompress()
f..uncompress('descomprimir.zip', 'descomprimir')