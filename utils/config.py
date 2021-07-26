import os
import sys

from configparser import ConfigParser, ExtendedInterpolation

prj_root:str = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
#sys.path.append(prj_root)
config_path:str = os.path.join(prj_root, 'setting.cfg')

class _ConfigManager():
    
    def __init__(self, cfg_path: str):
        self.cfg_path = cfg_path
        self.config = ConfigParser(interpolation=ExtendedInterpolation())
        self.config.read(self.cfg_path)
        self.get_sections = self.config.sections()

    def load(self, cfg_path=None):
        if cfg_path:
            self.cfg_path = cfg_path
        self.config.read(self.cfg_path)

    def __getitem__(self, item):
        return self.config[item]


cm = _ConfigManager(config_path)



