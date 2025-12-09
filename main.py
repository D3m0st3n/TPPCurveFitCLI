#!/usr/bin/env python3
"""
Curve Fitting CLI Tool with External Function for Protein Melting Curves
=============================================
A command-line tool for fitting curves using functions defined in external Python files.
Curves are fitted to protein melting data coming from TPP

"""
### Imports 
import os
import sys
import argparse
import logging
import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import curve_fit

### Classes
class SigmoidFitter:
    """
    Sigmoid curve fitting class to use on TPP data. Temperature unit is °C.
    """

    def __init__(self, log_level : int = logging.INFO):
        """
        Initialize the CurveFitting class.
        
        Args:
            log_level: Logging level (default: logging.INFO)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

        # Optimization parameters
        self.method = 'trf'
        self.maxfev = 10000
        # Intial parameters
        self.p0 = [0.1, 1, 0.1]
        # Curve parameters
        self.pl = np.nan
        self.a = np.nan
        self.b = np.nan
        self.pcov = np.nan
        # Statistics
        self.rmse = np.nan
        self.r_squared = np.nan

    def __repr__(self):
        print(f"""Sigmoid Curve Fitter \n 
              With parameters : pl = {self.pl:.4f}, a = {self.a:.4f}, b = {self.b:.4f} \n 
              And statistics : rmse = {self.rmse:.4f}, r_squared = {self.r_squared:.4f}""")

    # TPP formula
    @staticmethod
    def tpp_sigmoid(x, pl, a, b):
        y = (1 - pl) / (1 + np.exp(b - (a / x))) + pl
        return y

    @staticmethod
    def get_parameter_names():
        return ['pl', 'a', 'b']

    def fit_curve(self, temperature : np.ndarray, fold_change : np.ndarray, p0 : np.ndarray = None):

        # Initial parameters
        if p0 is not None:
            self.p0 = p0
        self.logger.info(f"Initial parameters: {self.p0}")
        
        try:
            self.method = 'trf'
            self.maxfev = 10000

            # Curve fitting
            popt, pcov = curve_fit(self.tpp_sigmoid, temperature, fold_change, self.p0, method=self.method, maxfev=self.maxfev)

            self.pl = popt[0]
            self.a = popt[1]
            self.b = popt[2]
            self.pcov = pcov
            
            # Calculate fitted values
            y_fit = self.tpp_sigmoid(temperature, *popt)
            
            # Calculate statistics
            residuals = fold_change - y_fit
            rmse = np.sqrt(np.mean(residuals**2))
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((fold_change - np.mean(fold_change))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            # Std error of parameters
            perr = np.sqrt(np.diag(pcov))
            
            self.r_squared = r_squared
            self.rmse = rmse
            # self.param_error = perr

            results = {
                'fitted_parameters': popt.tolist(),
                'parameter_errors': perr.tolist(),
                'covariance_matrix': pcov.tolist(),
                'rmse': float(rmse),
                'r_squared': float(r_squared),
                'residuals': residuals.tolist(),
                'y_fitted': y_fit.tolist(),
                'initial_parameters': self.p0
            }

            self.logger.info(f"Fitting complete. R² = {self.r_squared:.4f}, RMSE = {self.rmse:.4f}")
            self.logger.info(f"Fitted parameters: pl = {self.pl:.6f}, a = {self.a:.6f}, b = {self.b:.6f}")

            return results
        except RuntimeError as e:
            self.logger.error(f"Curve fitting failed: {str(e)}")
            raise ValueError(f"Failed to fit curve: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error during fitting: {str(e)}")
            raise
        
    def eval(self, temperature):
        return self.tpp_sigmoid(temperature, self.pl, self.a, self.b)
    
    def get_melting_temp(self) -> float:
        """Compute estimated melting temperature of protein.
        Estimation done on an 0°C to 100°C with a step of 0.01.

        Returns:
            float : melting temperature in °C. 2 decimals precision.
        """
        x = np.arange(0, 100.01, 0.01)
        melting_temp = x[(np.abs(self.tpp_sigmoid(x, self.pl, self.a, self.b) - 0.5)).argmin()] 

        return melting_temp

    def get_parameters(self) -> dict:
        """
        Get parameters of sigmoid function as a dictionary.
        Parameter names are pl (plateau), a & b.

        Returns:
            dict: parameters of sigmoid function
        """
        return {'pl' : self.pl, 'a' : self.a, 'b' : self.b}
    
    def get_parameters_error(self):
        return np.sqrt(np.diag(self.pcov))
    
    def get_intial_parameters(self):
        return self.p0
    
    def get_optimization_parameters(self):
        return self.method, self.maxfev

    def get_statistics(self):
        return {'rmse': self.rmse, 'r_squared' : self.r_squared}
    
class DataHandler:
    """
    Generic Data handling class for loading TPP data, fit a sigmoid, and save results.

    """
    def __init__(self, log_level : int = logging.INFO):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        
        # self.ifilename = None
        # self.ofilename = None
        self.data = None
        self.fit_results = None
    
    
    def load_data_from_path(self, input_path : str):
        """
        Load data in Dataframe.

        Args:
            input_path (str): path to input file.

        Raises:
            FileNotFoundError: _description_
            ValueError: _description_
        """
        self.logger.info(f"Loading data from: {input_path}")
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        try:
            # Try to read as CSV/JSON first
            if input_path.endswith('.csv'):
                df = pd.read_csv(input_path)
            if input_path.endswith('.json'):
                df = pd.read_json(input_path)
            else:
                # For other files, try to read as space/tab separated
                df = pd.read_csv(input_path, delim_whitespace=True, header=None)
            
            self.ifilename = Path(input_path).stem
            self.data = df
            
        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")

    def save_results(self, output_path):
        pass

    def process(self):
        pass
    

class MeltomeAtlasHandler(DataHandler):
    """
    Meltome Altas data handler class.
    It loads and splits data in different ways to use in other scripts.
    It also manage output directories for various tasks.
    """
    def __init__(self, flip_meltome_path : str, output_path : str, log_level : int = logging.INFO):
        super().__init__()
        self.logger.info(f"Meltome Handler initialization")

        # Meltome loading
        if not os.path.exists(flip_meltome_path):
            raise FileNotFoundError(f"Input file not found: {flip_meltome_path}")
        
        try:
            self.data = pd.read_json(flip_meltome_path)
            self.logger.info(f"Meltome data loaded from {flip_meltome_path}")
            self.logger.info(f"Meltome header {self.data.columns.to_list()}")
    
        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")
        
        # Ouptut creation
        self.output_dir = Path(output_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Output directory: {self.output_dir.absolute()}")
    
    def select_subset(self, n : int, group_key : str = 'runName'):
        try:
            subset = self.data.groupby(by=group_key).sample(n)
            return subset
        except Exception as e:
            self.logger.error(f"Error sampling data: {str(e)}")
            raise 

    def filter_data(self):
        pass
    
    def process_chunk(self, chunk : pd.DataFrame):

        results = {'pid' : [], 'runName' : [], 'pl' : [], 'a' : [], 'b' : [],
                   'rmse' : [], 'r_squared' : [], 'tm_pred' : [], 'tm_flip' : []}
        
        try :
            for i, row in chunk.iterrows():

                self.logger.debug(f"Chunk progress : {i} / {len(chunk)}, pid : {row.uniprotAccession}, specie : {row.runName}")

                # Intialize curve fitter and loading melting behaviour
                melting_curve = SigmoidFitter(self.logger.level)
                melting_data = pd.DataFrame(row.meltingBehaviour)
                
                # Fit melting curve
                melting_curve.fit_curve(melting_data.temperature.to_numpy(), melting_data.fold_change.to_numpy())

                # Fill results 
                results['pid'].append(row.uniprotAccession)
                results['runName'].append(row.runName)
                results['pl'].append(round(melting_curve.pl, 6))
                results['a'].append(round(melting_curve.a, 6))
                results['b'].append(round(melting_curve.b, 6))
                results['rmse'].append(round(melting_curve.rmse, 4))
                results['r_squared'].append(round(melting_curve.r_squared, 4))
                results['tm_pred'].append(melting_curve.get_melting_temp())
                results['tm_flip'].append(row.meltingPoint)
            return results
        
        except Exception as e:
            self.logger.error(f" Error in processing chunk : {e}")
            raise

    def process(self, num_chunks : int = 100):
        
        self.logger.info(f"Starting curve fitting process")

        # Split data into chuncks
        num_chunks = num_chunks
        if len(self.data) > num_chunks:
            chunk_indices = np.array_split(np.arange(len(self.data)), num_chunks)
        else:
            chunk_indices = np.arange(len(self.data))
        
        self.logger.debug(f"Data split into {len(chunk_indices)} chunks")

        for chunk_i in chunk_indices:
            
            self.logger.debug(f"Processing chunk {chunk_i} / {len(chunk_indices)} (size : {len(chunk_i)})")
            
            chunk = self.data.loc[chunk_i]
            chunk_results = self.process_chunk(chunk)

        pass


    def save_results(self, output_path):
        pass

def setup_logging(log_file: Optional[str] = None, log_level: int = logging.INFO) -> None:
    """
    Set up logging configuration.
    """
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)
    
    # File handler (if log_file provided)
    if log_file:
        # Create directory for log file if it doesn't exist
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
        
        logging.info(f"Logging to file: {log_file}")


def main():
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M")
    setup_logging(f'C:/Users/alexa/Documents/PROHITS/Code/MeltingBehaviourCLI/{timestamp_str}_main.log', logging.INFO)
    
    return 0

if __name__=="__main__":
    main()