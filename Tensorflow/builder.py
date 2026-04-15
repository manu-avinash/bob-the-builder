from config.layers import keras_Layers as kl
from keras.models import Sequential
import keras.backend as backend
backend.clear_session()
layers=[kl.input_()]
model=Sequential()


