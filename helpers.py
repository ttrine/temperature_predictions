import sys, csv, datetime
import numpy as np

def get_data(dataset):
	f = file('test' + dataset + '.tsv')
	r = csv.reader(f, delimiter='	')

	# Skip header
	r.next()
	r.next()

	return r

def get_labels(dataset):
	f = file('test'+ dataset +'_expected.tsv')
	r = csv.reader(f, delimiter='	')

	return [float(i[0]) for i in r]

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

def merge_sort(tmax_nans,tmin_nans,min_interp,max_interp):
	''' Zip inferred values with their original indices, merge the lists,
		and sort on the indices.'''
	tmax_nan_inds = np.where(tmax_nans)[0]
	tmin_nan_inds = np.where(tmin_nans)[0]
	sorted_vals = sorted(zip(tmin_nan_inds,min_interp) + zip(tmax_nan_inds,max_interp),key=lambda x: x[0])

	return [val[1] for val in sorted_vals]

def score(predictions, labels):
	'''Measure the error against the true values.'''
	print max(0,1 - (np.mean([abs(val[0] - val[1]) for val in zip(predictions, labels)]) / 5.))
