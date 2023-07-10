
from tensorflow.keras.callbacks import Callback, EarlyStopping,History,ModelCheckpoint #, ReduceLROnPlateau # , TensorBoard
# loss per epoch
from time import time
from pdb import set_trace
from keras import backend as K
import matplotlib
matplotlib.use('Agg') 
import wandb


class wandbCallback(Callback):
    # Just log everything to wandb
    def __init__(self):
        self.curr_step=0

    def _record_data(self,logs):
        #step_number = self.params['steps'] * self.epoch + self.params['batch']
        wandb.log(logs)
        #print("step_number", step_number, "logs", logs)

    def on_batch_end(self,batch,logs={}):
        self._record_data(logs)
        self.curr_step += 1

    def on_epoch_end(self,epoch,logs={}):
        self._record_data(logs)
        self.curr_step += 1

