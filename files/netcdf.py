import numpy as np
import netCDF4 as nc 
from os import walk       
from datetime import datetime,date                                                 

class NetCDF():
        # Constructor de la clase
        def __init__(self, filename=None, plots_path=None, coord=None, dataset=None):

                self.filename = filename
                self.coord = { "long":183, "lat":411 }
                self.dataset = nc.Dataset(self.filename, 'r')
        
        def Metadata(self):
                """
                Si quiero formatear la fecha de str a date
                date_str = '30-01-12'
                formatter_string = "%d-%m-%y" 
                datetime_object = datetime.strptime(date_str, formatter_string)
                date_object = datetime_object.date()
                """
                return self.dataset.START_DATE


        # Constructor de la clase
        def Dataset(self):                          
                return self.dataset
        
        # Dado el indice de un punto, devuelve los valores de latitud y longitud
        def Localization(self, i_lat, i_long):
                longitud = self.dataset.variables['XLONG'][0][i_long][i_lat]
                latitud = self.dataset.variables['XLAT'][0][i_long][i_lat]
                return [longitud, latitud]

        
        # Devuelve un dataset con las variables solicitadas en el parametro var_list
        # Cada elemento de la fila devuelta corresponde a los puntos de la 1ra fila, 
        # luego a la 2da, 3ra y asi sucesivamente
        def Variables(self, var_list):                
                values = []
                data = []

                for x in range(self.coord['long']):
                        for y in range(self.coord['lat']):
                                for var in var_list: 
                                        val = self.dataset.variables[var][0][x][y]
                                        values.append(val)
                                data.append(values)                                
                                values = []
                return data    


class ManagerNetCDF():

        def __init__(self, path=None, files_format=None, compress=None):

                self.path = path
                self.files_format = "wrfout_d03_2019-01-30_00:00:00"
                self.compress = True


        def Files(self):
               ruta = 'd03'
               listaarchivos = []

               for (_, _, archivos) in walk(ruta):
                     listaarchivos.extend(archivos)
               return listaarchivos

