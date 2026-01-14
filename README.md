# TPP data curve fitting CLI Tool

This project is a Command-Line Interface (__CLI__) in Python to fit sigmoid curves to Thermal Proteome Profiling (__TPP__) data.
The main goal of this project is compute parameters of the sigmoid curve given protein melting data from TPP. The resulting curve can be used as a model for the "melting profile" of a protein from which the melting point (__T<sub>m</sub>__) can be estimated. 

This work is based on the following [paper](https://www.science.org/doi/10.1126/science.1255784).

### Description

- `main.py`: contains all classes, functions and functional code. 

- `test.ipynb`: test bench for the code. Long term scope is to replace it with a proper set of examples and/or unit tests.

- `meltome_example.py` : example script for FLIP/Meltome data format. Check for correct processing over a subset.

- `example/`: contains example of inputs format and expected output to check for proper running.


### Running example

For Meltome format:
```
$ python meltome_example.py
```
It processes the example file `meltome_example.json`. It will create a log file, an output directory and compare the results with expected output (`meltome_curve_fit.csv`)

For LongF format:
```
$ python longf_example.py
```
It processes the example file `longf_example.csv`. It will create a log file, an output directory and compare the results with expected output (`longf_curve_fit.csv`)