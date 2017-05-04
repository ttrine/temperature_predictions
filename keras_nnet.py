from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers

from helpers import *

def construct():
	model = Sequential()
	model.add(Dense(20, activation='relu', input_dim=2))
	model.add(Dense(30, activation='relu', kernel_regularizer=regularizers.l1_l2(0.1)))
	model.add(Dense(40, activation='relu', kernel_regularizer=regularizers.l1_l2(0.1)))
	model.add(Dense(30, activation='relu', kernel_regularizer=regularizers.l1_l2(0.2)))
	model.add(Dense(20, activation='relu', kernel_regularizer=regularizers.l1_l2(0.2)))
	model.add(Dense(20, activation='relu', kernel_regularizer=regularizers.l1_l2(0.2)))
	model.add(Dense(20, activation='relu', kernel_regularizer=regularizers.l1_l2(0.1)))
	model.add(Dense(20, activation='relu', kernel_regularizer=regularizers.l1_l2(0.1)))
	model.add(Dense(1))

	model.compile(optimizer='adam', loss='mean_squared_error')

	return model

def train(matrix, y_nans, z_nans):
	''' Predict values of z_nans.
		Assumes matrix has columns time, y, z.'''

	train = matrix[~np.logical_or(y_nans,z_nans)]
	test = matrix[z_nans]

	model = construct()
	model.fit(train[:,0:2], train[:,2], batch_size=len(train), epochs=1000, verbose=0)
	
	predictions = [e[0] for e in model.predict(test[:,0:2])]
	return predictions

def run(dataset):
	reader = get_data(dataset)
	labels = get_labels(dataset)

	matrix = parse(reader)
	normalize(matrix)

	tmax_nans, tmin_nans = handle_nas(matrix)

	print "Running training..."
	min_interp = train(matrix, tmax_nans, tmin_nans)

	# Switch y and z before predicting missing tmax values
	max_interp = train(matrix[:,[0,2,1]], tmin_nans, tmax_nans)

	predictions = merge_sort(tmax_nans,tmin_nans,min_interp,max_interp)

	score(predictions, labels)

if __name__ == '__main__':
	if len(sys.argv) != 2:
		print "Usage: python -m keras_nnet dataset"
		sys.exit()

	dataset = sys.argv[1]

	# Run against specified dataset
	run(dataset)
