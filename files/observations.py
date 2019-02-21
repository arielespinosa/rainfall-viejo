#header=None

import pandas as pd

class Observations():

    def __init__(self, filename=None, dataset=None):
        self.filename = filename
        self.dataset = dataset

    def Read(self):
        val = pd.read_csv(self.filename)
        return val

    def ReadColums(self):        
        d = pd.read_csv(self.filename)
        dataset = d.iloc[:,[1,2,3,4,5,44]]
        return dataset