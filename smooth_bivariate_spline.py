from scipy.interpolate import SmoothBivariateSpline
from helpers import *

def interpolate(matrix, y_nans, z_nans):
	''' Interpolate values of z_nans.
		Assumes matrix has columns time, y, z.'''

	train = matrix[~np.logical_or(y_nans,z_nans)]
	test = matrix[z_nans]

	f = SmoothBivariateSpline(train[:,0], train[:,1], train[:,2],kx=2,ky=2,s=500)

	return f(test[:,0],test[:,1],grid=False)

def run(dataset):
	reader = get_data(dataset)
	labels = get_labels(dataset)

	matrix = parse(reader)
	normalize(matrix)

	tmax_nans, tmin_nans = handle_nas(matrix)

	min_interp = interpolate(matrix, tmax_nans, tmin_nans)

	# Switch y and z before predicting missing tmax values
	max_interp = interpolate(matrix[:,[0,2,1]], tmin_nans, tmax_nans)

	predictions = merge_sort(tmax_nans,tmin_nans,min_interp,max_interp)

	score(predictions, labels)

if __name__ == '__main__':
	if len(sys.argv) != 2:
		print "Usage: python -m keras_nnet dataset"
		sys.exit()

	dataset = sys.argv[1]

	# Run against specified dataset
	run(dataset)
