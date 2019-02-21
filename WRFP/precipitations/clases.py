import WRFP.settings
import json

class Configuration():
    def load_configuration(jsonfile):
        a = STATICFILES_DIRS + "/" + jsonfile
        try:
            configfile = json.loads(a)
        except: configfile = None

        return configfile