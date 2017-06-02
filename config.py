'''
config
'''
import ConfigParser as cp


config = cp.RawConfigParser()
config.read('./config.cfg')


eg_shoes7k_pos_path = config.get('examples', 'shoes7k_pos_path')
eg_shoes7k_neg_path = config.get('examples', 'shoes7k_neg_path')
