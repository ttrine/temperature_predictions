from sklearn.neural_network import MLPRegressor
from helpers import *

def train(matrix, y_nans, z_nans):
	''' Predict values of z_nans.
		Assumes matrix has columns time, y, z.'''

	train = matrix[~np.logical_or(y_nans,z_nans)]
	test = matrix[z_nans]

	model = MLPRegressor(hidden_layer_sizes=(100,50,10),activation='logistic',solver='lbfgs',alpha=.001,random_state=113512)
	model.fit(train[:,0:2],train[:,2])

	return model.predict(test[:,0:2])

def run(dataset):
	reader = get_data(dataset)
	labels = get_labels(dataset)

	matrix = parse(reader)
	normalize(matrix)

	tmax_nans, tmin_nans = handle_nas(matrix)

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
