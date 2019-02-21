from preprocess.netcdf import*
import numpy as np

"""
Variables:

QRAIN = LLuvia acumulada en kg(-l)
RAINC = LLuvia acumulada en kg(-l)
XLAT U
XLONG V
XLAT 
XLONG
Que significan estas variables?
RAINC
RAINSH
RAINNC
QNRAIN

XTIME = Fecha y Hora (2019-01-30 00:00:00)


183=Cantidad de puntos horizontales (long)
411=Cantidad de puntos verticales (lat)
    START_DATE: 2019-01-30_00:00:00
    SIMULATION_START_DATE: 2019-01-30_00:00:00

LA estructura es [TIME=0][X=Fila][Y=Coloumna]  


-85.71704-LON
19.340546-LAT

a = np.array([
      [[1,2,3]], 
      [[4,5,6]], 
    ])
b = np.array([
      [[7,8,9]], 
      [[10,11,12]], 
    ])
m = [[1, 2, 3], [4, 5, 6]]

for v in range(1, 3):
    for x in range(1, 5):       
        print(x)
 
"""

a = ManagerNetCDF()
p = NetCDF("wrfout_d03")
print(type(p.Metadata()))
#print(a.Files())

#return np.concatenate((lis), axis=None)
#print(a.Variables(var))
#print(a.Variables(var)[0])
#print(a.Variables(var)[1])
#print(a.Localization(0, 0))
#temp = [] 
#lis = np.empty_like(f.variables['XLONG'][0][0][0])


#for var in var_list:
       # temp = self.dataset.variables[var][0][0][0]
        #lis = np.concatenate((lis, temp), axis=None)
        
#return lis

#print(t)

#print("Variables separadas: ")
#print(f.variables['XLONG'][0][0][2])
#print(f.variables['RAINC'][0][2][0])
#print(f.variables['RAINC'][0][0][2])

#indice = np.where (f.variables['XLONG'][0] == -85.71704)
#print(indice)
#print(f.START_DATE)
