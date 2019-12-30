try:
    import keras
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten
    from keras.layers import Conv2D, MaxPooling2D
    from keras import backend as K
except:
    raise ImportError("For this example you need to install keras.")

try:
    import torchvision
    import torchvision.transforms as transforms
except:
    raise ImportError("For this example you need to install pytorch-vision.")



import numpy as np
import re


from imas_multi_single import main as trainmain
from imas_multi_single import smooth_data


import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker

import logging
logging.basicConfig(level=logging.DEBUG)


class ImasWorker(Worker):
    def __init__(self, N_train=8192, N_valid=1024, **kwargs):
            super().__init__(**kwargs)

            #self.batch_size = 64


    def compute(self, config, budget, working_directory, *args, **kwargs):
            """
            Simple example for a compute function using a feed forward network.
            It is trained on the MNIST dataset.
            The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
            """
            epochs = int(budget * 5)

#'epochs' : 5, \
#'rnn_type' : 'LSTM', \
#'rnn_size' : 8, \
#'learning_rate' : 1e-3, \
#'batch_size' : 64, \
#'sequence_length' : 24 * 7 * 1, \
#'steps_per_epoch' : 2, \
#'layers' : 1, \
#'dropout' : 0.5, \
#'warmup_steps' : 50, \
#'optimizer_type' : 'RMSprop', \
#'weight_initialization' : False}  

            #'weight_initialization' : False}
            #steps_per_epoch
            #'warmup_steps' : 50, \
                   
            #adds the train_steps hparam, as the variable budget, to the hparams_list                        
            
            #default values
            config['epochs'] = epochs
            config['weight_initialization'] = False
            config['steps_per_epoch'] = 10
            config['warmup_steps'] = 50
            config['learning_rate'] = 1e-3
            
            #values on different scale
            config['sequence_length'] = 12*7*config['sequence_length']
            config['batch_size'] = 2**config['batch_size']
            config['rnn_size'] = 2**config['rnn_size']

            print('configs: ', config)

            metrics_history = trainmain(config)
            train_loss = metrics_history['train_loss'] 
            val_loss = metrics_history['val_loss']
			      
             
            train_loss = min(smooth_data(train_loss ,2 , extension  = True))
            val_loss = min(smooth_data(val_loss ,2 , extension  = True))
            #import IPython; IPython.embed()
            return ({
                    'loss': val_loss,
                    'info': {'train loss': train_loss,
                             'number of parameters': 0,
                             }                  

            })


    @staticmethod
    def get_configspace():

            cs = CS.ConfigurationSpace()


            # Hyperparameters
            
            #hparams_dict ={ \
            #'rnn_type' : 'LSTM', \
            #'rnn_size' : 8, \
            #'batch_size' : 64, \
            #'sequence_length' : 12 * 7 * 1, \
            #'layers' : 1, \
            #'dropout_rate' : 0.5, \

            #'optimizer_type' : 'RMSprop', \
                                                
            #'weight_initialization' : False}
            #steps_per_epoch
            #'warmup_steps' : 50, \
            
            #Integer hyperparameters
            
            layers  = CSH.UniformIntegerHyperparameter('layers', lower=1, upper=2, default_value=1, log=False)
            sequence_length  = CSH.UniformIntegerHyperparameter('sequence_length', lower=1, upper=6, default_value=2, log=False)
                   
            rnn_size  = CSH.UniformIntegerHyperparameter('rnn_size', lower=4, upper=10, default_value=7, log=False)
            batch_size  = CSH.UniformIntegerHyperparameter('batch_size', lower=3, upper=8, default_value=6, log=False)
            
            #Float hyperparameters
            dropout = CSH.UniformFloatHyperparameter('dropout', lower=0.0, upper=0.9, default_value=0.2, log=False)

            
            #Catagorical hyperparameters
            rnn_type = CSH.CategoricalHyperparameter('rnn_type', ['LSTM', 'GRU'])            
            optimizer_type = CSH.CategoricalHyperparameter('optimizer_type', ['SGD', 'RMSprop', 'Adagrad'])


            #cs.add_hyperparameters([layers, transformer_ff, heads, rnn_size, batch_size, \
                                    #warmup_steps, dropout, label_smoothing, param_init, optim])
            cs.add_hyperparameters([layers, sequence_length, rnn_size, batch_size, dropout, \
                                    rnn_type, optimizer_type])
            return cs



if __name__ == "__main__":
    worker = TransformerWorker(run_id='0')
    cs = worker.get_configspace()

    config = cs.sample_configuration().get_dictionary()
    print('config: ', config)
    res = worker.compute(config=config, budget=1, working_directory='.')
    print(res)