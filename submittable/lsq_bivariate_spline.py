import sys, csv, datetime
import numpy as np
from scipy.interpolate import LSQBivariateSpline

def get_data(inp):
	r = csv.reader(inp, delimiter='	')

	# Skip header
	r.next()
	r.next()

	return r

def parse(reader):
	'''Preprocess into a 3-column numy matrix.'''
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

def handle_nas(matrix):
	''' Find the missing values, replace them by an arbitrary
		value outside the data range, and return logical vectors
		denoting their locations.'''

	tmax_nans = np.isnan(matrix[:,1])
	tmin_nans = np.isnan(matrix[:,2])

	matrix[tmax_nans,1] = -1000.
	matrix[tmin_nans,2] = -1000.

	return tmax_nans, tmin_nans

def interpolate(matrix, y_nans, z_nans):
	''' Interpolate values of z_nans.
		Assumes matrix has columns time, y, z.'''

	train = matrix[~np.logical_or(y_nans,z_nans)]
	test = matrix[z_nans]

	tx = np.linspace(np.min(train[:,0]), np.max(train[:,0]), 5)
	ty = np.linspace(np.min(train[:,1]), np.max(train[:,1]), 5)

	f = LSQBivariateSpline(train[:,0], train[:,1], train[:,2],tx,ty,kx=3,ky=3)

	return f(test[:,0],test[:,1],grid=False)

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

	tmax_nans, tmin_nans = handle_nas(matrix)

	min_interp = interpolate(matrix, tmax_nans, tmin_nans)

	# Switch y and z before interpolating missing tmax values
	max_interp = interpolate(matrix[:,[0,2,1]], tmin_nans, tmax_nans)

	merge_sort_print(tmax_nans,tmin_nans,min_interp,max_interp)
