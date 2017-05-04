from scipy.interpolate import LSQBivariateSpline
from helpers import *

def interpolate(matrix, y_nans, z_nans):
	''' Interpolate values of z_nans.
		Assumes matrix has columns time, y, z.'''

	train = matrix[~np.logical_or(y_nans,z_nans)]
	test = matrix[z_nans]

	tx = np.linspace(np.min(train[:,0]), np.max(train[:,0]), 10)
	ty = np.linspace(np.min(train[:,1]), np.max(train[:,1]), 5)

	f = LSQBivariateSpline(train[:,0], train[:,1], train[:,2],tx,ty,kx=3,ky=3)

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
		print "Usage: python -m lsq_bivariate_spline dataset"
		sys.exit()

	dataset = sys.argv[1]

	# Run against specified dataset
	run(dataset)
