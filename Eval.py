import keras
import numpy as np
class Evaluate_on_epoch(keras.callbacks.Callback,x,y):
	def on_train_begin(self, logs={}):
		return

	def on_train_end(self, logs={}):
		return

	def on_epoch_begin(self, epoch, logs={}):
		return

	def on_epoch_end(self, epoch, logs={}):
		y_hat = self.model.predict(x)
		print('on epoch: ',epoch,' ,test_acc',sum(np.where( np.argmax(y_hat) == y) ) )
		return
