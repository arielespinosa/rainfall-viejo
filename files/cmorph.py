""" Leer archivos binarios de la base CMORPH, datos de precipitacion con resolucion de aproximadamente 8 km
    y exportarlos en formato Long, Lat, Precip (mm/h).
    Los datos originales estan compactados a byte por punto, con codigo 255 para datos missing. Los archivos
    binarios tienen 6 bloques, el primero y el cuarto traen los valores de precipitacion de la primera y la
    segunda media hora del plazo. El script se invoca con el nombre del archivo binario CMORPH como argumento
    y la salida es por la consola.

    python CmorphXYZ.py nombre_fichero_cmorph """

import numpy as np
import sys

# valores iniciales e incrementos por latitud y longitud
# de los datos de CMORPH (-60 - 60 Lat y 360 Long)
Lon0 = 0.036378335                                              
Dlon = 0.072756669                                              
Lat0 = -59.963614
Dlat = 0.072771377

# Dimensiones de la matriz de datos de CMORPH
Ny = 1649                                                       
Nx = 4948

MISSING = -999.0

# Coordenadas de la ventana a extraer
TOP     = 40                                                    
BOTTOM  = 5
LEFT    = -105
RIGHT   = -50

class CMORPHF():
        
        def Read(self, filename):                
                dataset = []
                # Abrir el fichero CMORPH
                try:                
                        file = open(filename, 'rb')
                        # Leer bloque de la primera media hora
                        data1 = file.read(Ny*Nx)                                        
                        dumm  = file.read(Ny*Nx)
                        # Leer bloque de la segunda media hora
                        dumm  = file.read(Ny*Nx)
                        data2 = file.read(Ny*Nx)                                        

                        file.close()
                except IOError:
                        print(IOError)             
                
                # Calcular latitud del extremo norte (Los datos estan de norte a sur)
                TopLat = Ny * Dlat + Lat0                                       

                k=0
                for y in range(Ny):
                        lat = TopLat - y * Dlat
                        for x in range(Nx):

                                lon = Lon0 + x * Dlon
                                if lon > 180.0:
                                        # Longitudes de -180 a 180
                                        lon -= 360.0                            
                                # convertir de string a entero
                                af1 = int(data1[k])                             
                                af2 = int(data2[k])
                                k += 1
                                # Missing??
                                if af1 != 255 and af2 != 255:
                                        # Sumar las dos medias horas y llevar a mm/h                   
                                        A = (af1 + af2) * 0.2                   
                                else:
                                        A = MISSING
                                # Seleccionar ventana de coordenadas
                                if TOP > lat >= BOTTOM and RIGHT > lon >= LEFT: 
                                       # Formar el dataset redondeando los valores a 4 lugares despues de la coma
                                       point_values = [round(lon,4), round(lat,4), round(A,4)]                                       
                                       dataset.append(point_values)
                # Devolver el dataset                     
                return dataset
     
        
        def Imprimir(self):
                print("%7.2f, %6.2f, %5.1f"%(lon, lat, A))


