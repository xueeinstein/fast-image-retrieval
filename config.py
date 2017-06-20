'''
config
'''
import ConfigParser as cp


config = cp.RawConfigParser()
config.read('./config.cfg')


# config for example shoes7k
eg_shoes7k_pos_path = config.get('shoes7k', 'pos_path')
eg_shoes7k_neg_path = config.get('shoes7k', 'neg_path')
eg_shoes7k_latent_num = config.getint('shoes7k', 'latent_num')
eg_shoes7k_class_num = config.getint('shoes7k', 'class_num')

# config for example facescrub
eg_facescrub_folder = config.get('facescrub', 'root')
