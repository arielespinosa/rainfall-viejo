'''
  Implementation of custom keras regression models
'''

import os
import pickle
from utils.custom_losses import *
from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout
from keras.layers import LSTM, GRU
from keras.layers.noise import AlphaDropout
from keras.layers import BatchNormalization
from keras.layers import Concatenate, Reshape
from utils.custom_layers import Antirectifier, Capsule
from utils.custom_layers import RBFlayer, margin_loss, Lambda
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau

# Definition of some tensorflow callbacks:
tensorboard = TensorBoard(
    log_dir='./logs/',
    write_graph=True,
    write_images=False,
    histogram_freq=3)

stop_train = EarlyStopping(
    monitor='val_loss',
    min_delta=1e-4,
    patience=100,
    verbose='auto',
    mode='auto')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

# Capsules neural network.
# Returns a compiled Keras model instance.
class Capsule():

	def __init__(self, parameters):
		self.num_capsules=parameters["num_capsules"]
	    self.dim_capsules=parameters["dim_capsules"]
	    self.routings=parameters["routings"]
	    self.activation=parameters["activation"]
	    self.kernel_initializer=parameters["kernel_initializer"]
	    self.optimizer=parameters["optimizer"]
	    self.loss=parameters["loss"]
	    self.share_weights=parameters["share_weights"]
	    self.num_inputs=parameters["num_inputs"]
	    self.num_features=parameters["num_features"]
	    self.num_classes=parameters["num_classes"]
	    self.for_regression=parameters["for_regression"]
		self.name = "capsule_model"
		
	def create_model():
		inputs = Input(shape=(self.num_inputs, self.num_features))

		capsule = Capsule(
			self.num_classes,
			self.dim_capsules,
			routings=self.routings,
			share_weights=self.share_weights,
			activation=self.activation)(inputs)
		#
		x = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule)

		if for_regression:
			x = Dense(num_classes)(x)

		model = Model(inputs=inputs, outputs=x, name=)
		model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
		model.summary()

		return model

# Fully-connected MLP neural network.
# Returns a compiled Keras model instance.
class MultiLayerPerceptron():
	
	def __init__(self, parameters):
		self.dense_units=parameters["dense_units"]
	    self.h_activation=parameters["h_activation"]
	    self.o_activation=parameters["o_activation"]
	    self.antirectifier=parameters["antirectifier"]
	    self.batch_norm=parameters["batch_norm"]
	    self.dropout=parameters["dropout"]
	    self.dropout_rate=parameters["dropout_rate"]
	    self.kernel_initializer=parameters["kernel_initializer"]
	    self.optimizer=parameters["optimizer"]
	    self.loss=parameters["loss"]
	    self.num_classes=parameters["num_classes"]
	    self.num_inputs=parameters["num_inputs"]
		self.name = "mlp_model"
	
	def create_model():
		if np.isscalar(dense_units):
			dense_units = (dense_units, )
		else:
			if len(dense_units) == 0:
				raise ValueError('dense_units must be a scalar or a tuple')

		inputs = Input(shape=(num_inputs, ), name='inputs')

		# Hidden Layers
		x = inputs
		for units in dense_units:
			x = Dense(
				self.units,
				activation='linear',
				kernel_initializer=self.kernel_initializer)(x)
			if antirectifier:
				x = Antirectifier()(x)
			else:
				if self.batch_norm:
					x = BatchNormalization()(x)
				x = Activation(self.h_activation)(x)

			if self.dropout is not None: 
				x = dropout(self.dropout_rate)(x)

		# Output Layer
		x = Dense(self.num_classes)(x)
		outputs = Activation(self.o_activation)(x)

		model = Model(inputs=self.inputs, outputs=self.outputs, name=self.name)
		model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
		model.summary()
		return model

# Radial-Basis-Function neural network.
# Returns a compiled Keras model instance.    
class RBF_Regressor():

	def __init__(self, parameters):	
		self.units=parameters["units"]
		self.output_activation=parameters["output_activation"]
	    self.kernel_initializer=parameters["kernel_initializer"]
	    self.kernel_activation=parameters["kernel_activation"]
		self.kernel_constraint=parameters["kernel_constraint"]
	    self.loss=parameters["loss"]
	    self.optimizer=parameters["optimizer"]
	    self.num_inputs=parameters["num_inputs"]	    
	    self.num_classes=parameters["num_classes"]
	    self.for_regression=parameters["for_regression"]
		self.name = "rbf_model"
		
	def create_model():
		inputs = Input(shape=(self.num_inputs,), name='inputs')
		# Need to fix some issues for multiple inputs here!!!
		# for now we use a dense layer to perform a linear
		# combination of the inputs ...
		x = Dense(1, activation='linear', use_bias=True)(inputs)

		# a single RBF layer would do, dont dare to use more (better wide than deep)
		x = RBFlayer(self.units, 
			kernel_activation=self.kernel_activation,
			kernel_initializer=self.kernel_initializer)(x)

		# perform a linear combination of RBF outputs, no need for bias here
		# (see later for convexity constraint !!)
		x = Dense(self.num_classes, use_bias=False)(x)
		output = Activation(self.output_activation)(x)

		model = Model(inputs=self.inputs, outputs=self.output, name=self.name)
		model.compile(loss=self.loss, optimizer=self.optimizer)
		model.summary()

		return model

# Long-Short-Tensor-Memory (LSTM) neural network.
# Returns a compiled Keras model instance. 
class LSTM():

	def __init__(self, parameters):	
		self.units=parameters["units"]
		self.activation=parameters["activation"]
		self.output_activation=parameters["output_activation"]
	    self.kernel_initializer=parameters["kernel_initializer"]	    
		self.kernel_constraint=parameters["kernel_constraint"]
	    self.loss=parameters["loss"]
	    self.optimizer=parameters["optimizer"]
	    self.num_inputs=parameters["num_inputs"]	    
	    self.num_classes=parameters["num_classes"]	    
		self.name = "lstm_model"
		
	def create_model():
		inputs = Input(shape=hparams['shape'], name='inputs')

		# define model
		with K.name_scope('input_layer'):
			x = TimeDistributed(BatchNormalization(epsilon=1e-5))(inputs)

		# define lstm layers (many to one):
		with K.name_scope('lstm_layers'):
			x = LSTM(self.units, return_sequences=True, activation=self.activation)(x)
			x = LSTM(self.units, return_sequences=False, activation=self.activation)(x)

		with K.name_scope('output_layer'):
			outputs = Dense(self.num_classes, activation=self.output_activation)(x)

		# model
		model = Model(inputs, outputs=outputs, name=name)
		model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
		model.summary()

		return model

# experiment
def run_experiment(model, x, y, parameters):

    # Training the model ...
    history = model.fit(
        x, y,
        batch_size=parameters['batch_size'],
        epochs=parameters['epochs'],
        validation_split=parameters['val_split'],
        shuffle=parameters['train_shuffle'],
        # callbacks=[tensorboard, stop_train, reduce_lr],
        callbacks=[stop_train, reduce_lr],
        verbose=1)

    return history


def save_keras_model(model, path=None):
    if path is None:
        save_path = model.name
    else:
        save_path = os.path.join(path, model.name)
    model.save(save_path)
    

