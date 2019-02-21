import json

class Configuration():

    def __init__(self, filename=None, config=None):
        self.filename = filename 
        self.config   = config

    def load(filename):
        a = STATICFILES_DIRS + "/" + jsonfile
        try:
            configfile = json.loads(a)
        except: configfile = None

        with open(self.filename, mode='r', encoding='utf-8') as f :
            json.dump(self.filename, f, indent=2)

    def save(self):        
        fichero = {}
        fichero['id'] = 256
        fichero['title'] = 'Diveint o history'
        fichero['tags'] = ('dive intopython', 'docbook', 'html')
        fichero['published'] = True
        fichero['commentslink'] = None
        
        with open(self.filename, mode='w', encoding='utf-8') as f :
            json.dump(fichero, f, indent=2)
    

        
#con = Configuration('config.json')
#con.save()
#print(con['id'])

import gzip

class Files():

	def WriteGZ(self):
		text = 'A nine mile walk is no joke'
		with gzip.open('out.log.gz', mode='wb') as zfile:
			zfile.write(text.encode('utf-8'))
    
    def ReadGZ():
        with gzip.open('out.log.gz', mode='r') as zfile:
            zfile.read(text.encode('utf-8'))
f = Files()
f.ReadGZ()
print(f)



