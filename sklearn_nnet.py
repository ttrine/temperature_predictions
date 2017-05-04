import sys, csv, datetime
import numpy as np
from sklearn.neural_network import MLPRegressor

def get_data(inp):
	r = csv.reader(inp, delimiter='	')

	# Skip header
	r.next()
	r.next()

	return r

def parse(reader):
	'''Preprocess into a 3-column numpy matrix.'''
	time = []
	tmax = []
	tmin = []

	for row in reader:
		dt = datetime.date(int(row[0]),datetime.datetime.strptime(row[1],'%B').month,1)
		time.append(dt.toordinal())
		if 'Missing' in row[2]: tmax.append(np.nan)
		else: tmax.append(float(row[2]))
		if 'Missing' in row[3]: tmin.append(np.nan)
		else: tmin.append(float(row[3]))

	return np.array([time,tmax,tmin]).transpose()

def normalize(matrix):
	''' Normalizes the date column only. '''
	matrix[:,0] -= np.mean(matrix[:,0])
	matrix[:,0] /= np.std(matrix[:,0])

def handle_nas(matrix):
	''' Find the missing values and return 
		logical vectors denoting their locations.'''

	tmax_nans = np.isnan(matrix[:,1])
	tmin_nans = np.isnan(matrix[:,2])

	return tmax_nans, tmin_nans

def train(matrix, y_nans, z_nans):
	''' Predict values of z_nans.
		Assumes matrix has columns time, y, z.'''

	train = matrix[~np.logical_or(y_nans,z_nans)]
	test = matrix[z_nans]

	model = MLPRegressor(hidden_layer_sizes=(100,50,10),activation='logistic',solver='lbfgs',alpha=.001)
	model.fit(train[:,0:2],train[:,2])

	return model.predict(test[:,0:2])

def merge_sort_print(tmax_nans,tmin_nans,min_interp,max_interp):
	''' Zip inferred values with their original indices, merge the lists,
		sort on the indices, and print each element on a new line.'''
	tmax_nan_inds = np.where(tmax_nans)[0]
	tmin_nan_inds = np.where(tmin_nans)[0]
	sorted_vals = sorted(zip(tmin_nan_inds,min_interp) + zip(tmax_nan_inds,max_interp),key=lambda x: x[0])
	for value in sorted_vals:
		print value[1]

if __name__ == '__main__':
	reader = get_data(sys.stdin)

	matrix = parse(reader)
	normalize(matrix)

	tmax_nans, tmin_nans = handle_nas(matrix)

	min_interp = train(matrix, tmax_nans, tmin_nans)

	# Switch y and z before interpolating missing tmax values
	max_interp = train(matrix[:,[0,2,1]], tmin_nans, tmax_nans)

	merge_sort_print(tmax_nans,tmin_nans,min_interp,max_interp)
