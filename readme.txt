Project structure
	- helpers.py contains all the data manipulation logic not strictly related to a particular model. Look here for pre and post processing routines.
	- The runnable models are contained in smooth_bivariate_spline.py, lsq_bivariate_spline.py, sklearn_nnet.py, and keras_nnet.py.
	- While adding structure to the project, I had to break compatibility with the hackerrank interface. If, however, you wish to submit the code to the web interface, use the files in the 'submittable' subfolder. These are similar to the files in the root folder, but they accept input from sdtin instead of loading it from files.

Running the code locally
	1. Ensure you have Python 2, pip, and (optionally) virtualenv installed.
	2. Navigate to the project folder.
	3. Run 'pip install requirements.txt' from this folder in your local environment. This will ensure your system has the correct dependencies. 
		Note: If you wish to install these dependencies in an isolated environment, create a virtual environment first. To do this, run 'virtualenv venv'. This will create a separate Python 2 installation as a subdirectory of the project folder. You'll then need to switch it on with the command 'source venv/bin/activate'. To switch it off again, just type 'deactivate.'
	4. Run a model! The template for running a model file looks like this:
		python -m script_name dataset_number

	This means we are running the script_name python script as a module (a package concept) and passing in dataset_number as an argument. dataset_number should be either 0 or 1 and corresponds to the first and second test cases on the website respectively. To run the model against the smaller test case, use 0. To run it against the full dataset, use 1. Here are some examples:

		python -m smooth_bivariate_spline 0
		python -m lsq_bivariate_spline 1
		python -m sklearn_nnet 0
		python -m keras_nnet 1

	These commands will train the corresponding model and print out its score. Score is calculated by the same formula as the website:
		1 - (average error / 5.0)
	Where the average error is defined as the mean absolute difference between the function values and the data points.
