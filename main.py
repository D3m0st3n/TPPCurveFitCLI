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
import textwrap
from pathlib import Path
from typing import Optional
from enum import Enum

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import seaborn as sns

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
        return f"""Sigmoid Curve Fitter \n 
        With parameters : pl = {self.pl:.4f}, a = {self.a:.4f}, b = {self.b:.4f} \n 
        And statistics : rmse = {self.rmse:.4f}, r_squared = {self.r_squared:.4f}"""

    # TPP formula
    @staticmethod
    def tpp_sigmoid(x, pl, a, b):
        y = (1 - pl) / (1 + np.exp(b - (a / x))) + pl
        return y

    def eval(self, temperature):
        return self.tpp_sigmoid(temperature, self.pl, self.a, self.b)
    
    @staticmethod
    def get_parameter_names():
        return ['pl', 'a', 'b']
    
    def fit_curve(self, temperature : np.ndarray, fold_change : np.ndarray, p0 : np.ndarray = np.array([0.1, 1, 0.1])): 

        # Initial parameters
        if (p0 != self.p0).any():
            self.p0 = p0
        self.logger.debug(f"Initial parameters: {self.p0}")
        
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

            self.logger.debug(f"Fitting complete. R² = {self.r_squared:.4f}, RMSE = {self.rmse:.4f}")
            self.logger.debug(f"Fitted parameters: pl = {self.pl:.6f}, a = {self.a:.6f}, b = {self.b:.6f}")

            return results
        except RuntimeError as e:
            raise ValueError("Curve fitting failed to converge.") from e
        
        except Exception as e:
            self.logger.error(f"Error during fitting: {str(e)}")
            raise
   
    def get_melting_temp(self) -> float:
        """Compute estimated melting temperature of protein.
        Estimation done on an 0°C to 100°C with a step of 0.01.

        Returns:
            float : melting temperature in °C. 2 decimals precision.
        """
        x = np.arange(0, 100.01, 0.01)
        melting_temp = x[(np.abs(self.tpp_sigmoid(x, self.pl, self.a, self.b) - 0.5)).argmin()] 

        return float(melting_temp)

    def get_parameters(self) -> dict:
        """
        Get parameters of sigmoid function as a dictionary.
        Parameter names are pl (plateau), a & b.

        Returns:
            dict: parameters of sigmoid function
        """
        return {'pl' : self.pl, 'a' : self.a, 'b' : self.b}
    
    def set_parameters(self, params : dict):
        self.a, self.b, self.pl = params['a'], params['b'], params['pl']
    
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
    def __init__(self, output_path : str, log_level : int = logging.INFO):
        # Logger set up
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        
        
        # Ouptut creation
        self.output_dir = Path(output_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Output directory: {self.output_dir.absolute()}")

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

    def save_curve_fit_results(self, results : pd.DataFrame, intialize : bool = False, output_file = "curve_fit.csv"):
        """
        Save results to a CSV file in output path of instance.
        Handle file creation or append with the initialize argument.

        Args:
            results (pd.DataFrame): Dataframe containing curve fit results
            intialize (bool, optional): Define if results should be appeneded or if a new file has to be created. Defaults to False.
            output_file (str): Name of output file. defautls to curve_fit.csv.
        """
        if results.empty:
            return

        output_file = Path(self.output_dir / output_file)
        file_mode = 'a'
        header = False
        
        if intialize:
            file_mode = 'x'
            header = True 

        self.logger.debug(f"Output dir {output_file}")
        self.logger.debug(f"Writing results to {output_file.name} in mode {file_mode}")

        try:
            results.to_csv(output_file, mode=file_mode, index=False, header=header)

        except IOError as e:
            self.logger.error(f"Failed to write results to CSV file {output_file}: {e}")
            raise

    def process(self) -> pd.DataFrame:
        return pd.DataFrame()

class LongFormatHandler(DataHandler):
    
    def __init__(self, file_path : str, output_path : str, log_level : int = logging.INFO) -> None:
        super().__init__(output_path, log_level)
        self.logger.info("LongF Handler initialization")
        
        self.n_jobs = 6
        
        # Check if input path/file is viable
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        if not str(file_path).endswith('.csv'):
            raise ValueError(f"Input file is not in CSV format: {file_path}")

        # Result header 
        self.header = ['pid', 'replicate', 'pl', 'a', 'b', 'rmse', 'r_squared', 'tm_pred']
        
        # Load data
        try:
            self.data = pd.read_csv(file_path)
            self.logger.info(f"LongF data loaded from {file_path}")
            self.logger.info(f"LongF header {self.data.columns.to_list()}")
    
        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")
        
    def normalize_chunk(self, data_serie : pd.DataFrame) -> pd.DataFrame:
        data_serie.Abundance = data_serie.Abundance / data_serie.sort_values(by='Temperature').Abundance.iloc[0]
        return data_serie

    def normalize_dataframe_parallel(self, data : pd.DataFrame, n_jobs : int = 6):
        
        chunks = data.groupby(by=['Accession', 'Replicate']).groups
        normalized_data = pd.concat(Parallel(n_jobs=6)(delayed(self.normalize_chunk)(data.loc[ids]) for _, ids in chunks.items()))
        
        normalized_data
        
        return normalized_data
    
    def normalize_data(self, n_jobs : int = 6):
        self.data = self.normalize_dataframe_parallel(self.data, n_jobs)
    
    def process_serie(self, data_serie : pd.DataFrame):
        pid = data_serie.Accession.iloc[0]
        replicate = data_serie.Replicate.iloc[0]
        
        if data_serie.sort_values(by='Temperature').Abundance.iloc[0] != 1.0:
            self.logger.warning(f"Uh oh, looks like your data has not been normalized!")
        
        try:
            # Intialize curve fitter and loading melting behaviour
            melting_curve = SigmoidFitter(self.logger.level)
            
            # Fit melting curve
            melting_curve.fit_curve(data_serie.Temperature.to_numpy(), data_serie.Abundance.to_numpy())

                
            return {
            'pid': pid,
            'replicate' : replicate,
            'pl': round(melting_curve.pl, 6),
            'a': round(melting_curve.a, 6),
            'b': round(melting_curve.b, 6),
            'rmse': round(melting_curve.rmse, 4),
            'r_squared': round(melting_curve.r_squared, 4),
            'tm_pred': round(melting_curve.get_melting_temp(), 2),
            'status': 'SUCCESS' # Add a status flag
            }   
            

        except ValueError as e:
            # Log the failure with full context and the index
            self.logger.error(
                f"FAILURE (Fit): {pid} - {replicate} failed to converge at index."
                f"Reason: {e}"
            )
            
            return {
            'pid': pid, 
            'replicate': replicate, 
            'status': 'FAILURE', 
            'error_message': str(e),
            'pl': np.nan, 'a': np.nan, 'b': np.nan,  'rmse': np.nan, 'r_squared': np.nan, 
            'tm_pred': np.nan
            }
        
        except Exception as e:
            self.logger.critical(
                f"FATAL ERROR in processing serie {pid} - {replicate}. "
                f"Unexpected Exception: {e}"
            )
            raise
    
    def process(self, n_jobs : int = 10) -> pd.DataFrame:
        
        self.logger.info(f"Normalize data")
        
        self.normalize_data(n_jobs)
        
        self.logger.info(f"START - curve fitting process parallel")
        
        chunks = self.data.groupby(by=['Accession', 'Replicate']).groups
        try:
            results = pd.DataFrame(Parallel(n_jobs)(delayed(self.process_serie)(self.data.loc[ids]) for _, ids in chunks.items()))
    
            # Save chunk results              
            self.save_curve_fit_results(results[self.header], intialize=True)
        
        except Exception as e:
            self.logger.error(f"Error in processing: {e}")
            raise
        
        self.logger.info(f"END - Curve fitting process parallel")

        return results  

class MeltomeAtlasHandler(DataHandler):
    """
    Meltome Altas data handler class.
    It loads and splits data in different ways to use in other scripts.
    It also manage output directories for various tasks.
    """
    def __init__(self, flip_meltome_path : str, output_path : str, log_level : int = logging.INFO):
        super().__init__(output_path, log_level)
        self.logger.info(f"Meltome Handler initialization")

        # Meltome loading
        if not os.path.exists(flip_meltome_path):
            raise FileNotFoundError(f"Input file not found: {flip_meltome_path}")
        
        if not str(flip_meltome_path).endswith('.json'):
            raise ValueError(f"Input file is not in JSON format: {flip_meltome_path}")
        
        # Result header 
        self.header = ['pid', 'runName', 'pl', 'a', 'b', 'rmse', 'r_squared', 'tm_pred', 'tm_flip']
        
        # Load data
        try:
            self.data = pd.read_json(flip_meltome_path)
            self.logger.info(f"Meltome data loaded from {flip_meltome_path}")
            self.logger.info(f"Meltome header {self.data.columns.to_list()}")
    
        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")

    def select_subset(self, n : int, group_key : str = 'runName'):
        try:
            subset = self.data.groupby(by=group_key).sample(n)
            return subset
        except Exception as e:
            raise ValueError(f"Error sampling data: {str(e)}")

    def filter_data(self):
        pass

    def process_row(self, row : pd.DataFrame):
        pid = row.uniprotAccession
        run_name = row.runName

        try:
            # Intialize curve fitter and loading melting behaviour
            melting_curve = SigmoidFitter(self.logger.level)
            melting_data = pd.DataFrame(row.meltingBehaviour)
            
            # Fit melting curve
            melting_curve.fit_curve(melting_data.temperature.to_numpy(), melting_data.fold_change.to_numpy())

            return {
            'pid': pid,
            'runName': run_name,
            'pl': round(melting_curve.pl, 6),
            'a': round(melting_curve.a, 6),
            'b': round(melting_curve.b, 6),
            'rmse': round(melting_curve.rmse, 4),
            'r_squared': round(melting_curve.r_squared, 4),
            'tm_pred': melting_curve.get_melting_temp(),
            'tm_flip': row.meltingPoint,
            'status': 'SUCCESS' # Add a status flag
        }
        

        except ValueError as e:
            # Log the failure with full context and the index
            self.logger.error(
                f"FAILURE (Fit): {pid} - {run_name} failed to converge at index {row.name}."
                f"Reason: {e}"
            )
            
            return {
            'pid': pid, 
            'runName': run_name, 
            'status': 'FAILURE', 
            'error_message': str(e),
            'pl': np.nan, 'a': np.nan, 'b': np.nan,  'rmse': np.nan, 'r_squared': np.nan, 
            'tm_pred': np.nan, 'tm_flip': row.meltingPoint
        }
    
        except Exception as e:
            self.logger.critical(
                f"FATAL ERROR in processing row {pid} - {run_name} at index {row.name}. "
                f"Unexpected Exception: {e}"
            )
            raise
    
    def process_chunk(self, chunk : pd.DataFrame):

        results = {'pid' : [], 'runName' : [], 'pl' : [], 'a' : [], 'b' : [],
                   'rmse' : [], 'r_squared' : [], 'tm_pred' : [], 'tm_flip' : []}
        
        for i, (index, row) in enumerate(chunk.iterrows()):
            pid = row.uniprotAccession
            run_name = row.runName

            try :
                self.logger.info(f"Chunk progress : {i+1} / {len(chunk)}, pid : {row.uniprotAccession}, specie : {row.runName}")

                # Intialize curve fitter and loading melting behaviour
                melting_curve = SigmoidFitter(self.logger.level)
                melting_data = pd.DataFrame(row.meltingBehaviour)
                
                # Fit melting curve
                melting_curve.fit_curve(melting_data.temperature.to_numpy(), melting_data.fold_change.to_numpy())

                # Fill results 
                results['pid'].append(pid)
                results['runName'].append(run_name)
                results['pl'].append(round(melting_curve.pl, 6))
                results['a'].append(round(melting_curve.a, 6))
                results['b'].append(round(melting_curve.b, 6))
                results['rmse'].append(round(melting_curve.rmse, 4))
                results['r_squared'].append(round(melting_curve.r_squared, 4))
                results['tm_pred'].append(melting_curve.get_melting_temp())
                results['tm_flip'].append(row.meltingPoint)

            except ValueError as e:
                self.logger.error(
                    f"FAILURE (Fit): {pid} - {run_name} at index {index} failed to converge. "
                    f"Reason: {e}"
                )
                # Also curve fitting fails, results will be filled with NaN.
                results['pid'].append(pid)
                results['runName'].append(run_name)
                results['pl'].append(np.nan)
                results['a'].append(np.nan)
                results['b'].append(np.nan)
                results['rmse'].append(np.nan)
                results['r_squared'].append(np.nan)
                results['tm_pred'].append(np.nan)
                results['tm_flip'].append(row.meltingPoint)
                
                continue # Go to the next iteration (next row)

            except Exception as e:
                self.logger.critical(
                    f"FATAL ERROR in processing row {pid} - {run_name} at index {index}. "
                    f"Unexpected Exception: {e}"
                )
                raise

        return results

    def process(self, num_chunks : int = 100) -> pd.DataFrame:
        self.logger.info(f"START - curve fitting process")

        # Split data into chuncks
        if len(self.data) > num_chunks:
            chunk_indices = np.array_split(np.arange(len(self.data)), num_chunks)
        else:
            chunk_indices = np.arange(len(self.data))
        
        self.logger.debug(f"Data split into {len(chunk_indices)} chunks")

        # Main processing loop
        results = pd.DataFrame()
        intialize = True
        for i, chunk_i in enumerate(chunk_indices):
            
            self.logger.info(f"Processing chunk {i+1} / {len(chunk_indices)} (size : {len(chunk_i)})")
            
            try:
                # Process chunk
                chunk = self.data.iloc[chunk_i]
                chunk_results = pd.DataFrame(self.process_chunk(chunk))
                # Save chunk results              
                self.save_curve_fit_results(chunk_results[self.header], intialize)
                intialize = False
                # 
                results = pd.concat([results, chunk_results], ignore_index=True)

            except Exception as e:
                self.logger.error(f"Error in processing chunk : {e}")
                raise
        
        self.logger.info(f"END - Curve fitting process")

        return results
    
    def process_parallel(self, num_chunks : int = 100, n_jobs : int = 10) -> pd.DataFrame:
        
        self.logger.info(f"START - curve fitting process parallel")

        # Split data into chuncks
        if len(self.data) > num_chunks:
            chunk_indices = np.array_split(np.arange(len(self.data)), num_chunks)
        else:
            chunk_indices = np.arange(len(self.data))
        
        self.logger.debug(f"Data split into {len(chunk_indices)} chunks")
        self.logger.info(f"NOTE : Parallel processing of chunks does not log row progress")

        
        # Main processing loop
        results = pd.DataFrame()
        intialize = True
        for i, chunk_i in enumerate(chunk_indices):
            
            self.logger.info(f"Processing chunk {i+1} / {len(chunk_indices)} (size : {len(chunk_i)})")
            try:
                # Process chunk
                chunk = self.data.iloc[chunk_i]
                chunk_results = pd.DataFrame(Parallel(n_jobs=n_jobs)(delayed(self.process_row)(row) for _, row in chunk.iterrows())) # type: ignore
                
                # Save chunk results              
                self.save_curve_fit_results(chunk_results[self.header], intialize)
                intialize = False
                #
                results = pd.concat([results, chunk_results],ignore_index=True)

            except Exception as e:
                self.logger.error(f"Error in processing chunk {i}: {e}")
                raise
        
        self.logger.info(f"END - Curve fitting process parallel")

        return results
      
class DataType(Enum):
    """
    Class for easier type management of TppSigmoidPlotter. 
    """
    GENERIC = 0
    FLIP = 1
    LONGF = 2
    MSPEC = 3 # Keep it? Although it will be functionally identical to LONGF processing - it just needs preporcessing

class TppSigmoidPlotter:
    
    KEYS = ['Temperature', 'Abundance', 'Replicate']
    
    def __init__(self, curve_params, melting_data = pd.DataFrame(), data_type : DataType = DataType.GENERIC, log_level = logging.INFO) -> None:
        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        
        # Curve parameters
        self.params = pd.Series(curve_params)
        # Initilaze curve from params for evaluation
        self.fitter = SigmoidFitter(log_level)
        self.fitter.set_parameters(curve_params)
        
        # Intialize melting for scatterplot
        self.melting_data = melting_data

        # Data type - for handling headers and unique fields
        self.data_type = data_type
        
        match(self.data_type):
            case DataType.GENERIC:
                pass
            case DataType.FLIP:
                # Rename columns
                self.melting_data.rename({'fold_change' : 'Abundance', 'temperature' : 'Temperature'})
                # Add Replicate information
                self.melting_data['Replicate'] = 'REP0'
                if len(self.melting_data) == 20: # Data contains 2 rpelicates
                    self.melting_data['Replicate'][1::2] = 'REP1'
            case DataType.LONGF:
                pass
            
    
    @classmethod
    def from_flip(cls, curve_params, melting_data, log_level = logging.INFO):
        return cls(curve_params, melting_data, DataType.FLIP, log_level)
    
    @classmethod
    def from_longf(cls, curve_params, melting_data, log_level = logging.INFO):
        return cls(curve_params, melting_data, DataType.LONGF, log_level)
    
    @classmethod
    def from_mspec(cls, curve_params, melting_data, log_level = logging.INFO):
        # format mspec to longf
        # return cls(curve_params, melting_data, DataType.LONGF, log_level)
        pass
    
    @staticmethod
    def plot_curve(curve_params : pd.Series, figsize=(8, 5)):
        # Draw melting curve alone with melting point
        pass
    
    @staticmethod
    def plot_from_longf(curve_params : pd.Series, data_exp : pd.DataFrame = pd.DataFrame(), figsize=(8, 5)):
    
        fitter = SigmoidFitter()
        fitter.set_parameters(curve_params.to_dict())
        
        sns.set_theme()
        fig, ax = plt.subplots(figsize=figsize)
        palette = sns.color_palette('colorblind')
        legend_dots = []
        number_repl = 0
        try :
            if not data_exp.empty:
                number_repl = len(data_exp.Replicate.unique())

                # Plot Experimental Data
                sns.scatterplot(data=data_exp, x='Temperature', y='Abundance', hue='Replicate', palette=palette, 
                                alpha=0.7, s=50, edgecolor='k',
                                ax=ax, legend=False)

                # Extract the unique colors used by the scatterplot
                unique_colors = sns.color_palette(palette, number_repl)

                # Create "Proxy" dots for the legend
                for col in unique_colors:
                    d = plt.Line2D([0], [0], alpha=0.7, marker='o',  # type: ignore
                                markerfacecolor=col, markeredgecolor='k', markeredgewidth=1, 
                                markersize=8, linestyle='')
                    legend_dots.append(d)
                    
            # Plot Interpolated Curve
            sns.lineplot(x=np.arange(0, 100.1, 0.1), y=fitter.eval(np.arange(0, 100.1, 0.1)), color=palette[number_repl], label="Interpolated Fit", ax=ax)

            # Plot Melting Point Marker
            ax.scatter(fitter.get_melting_temp(), 0.5, color="red", marker='x', s=50, zorder=5, label='Melting Point')

            # Add the Coordinate Label
            ax.annotate(f'Tm = {fitter.get_melting_temp():.2f}', 
                        xy=(fitter.get_melting_temp(), 0.5), 
                        xytext=(12, -5), 
                        textcoords='offset points',
                        fontsize=8, 
                        color='red', 
                        fontweight='bold')

            # Get the existing handles (the Fit line)
            handles, labels = ax.get_legend_handles_labels()

            ax.set_ylim(0)
            ax.set_xlabel('Temperature (°C)')
            ax.set_title(f"Melting Behaviour of {curve_params.pid} with R2 = {curve_params.r_squared:.3f}, RMSE = {curve_params.rmse:.3f}")
            
            if not data_exp.empty:
                # Add our grouped dots tuple to the handles
                ax.legend(
                    handles=[handles[0], tuple(legend_dots), handles[1]], 
                    labels=[labels[0], "Experimental Replicates", labels[1]],
                    handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)},
                    loc='upper right'
                )
                # Resize plot for experimental data
                ax.set_xlim(data_exp.Temperature.min() - 5, 100)
                
            else:
                ax.legend(loc='upper right')
                
            
            plt.show()
        except Exception as e:
            raise
    
    def plot(self, plot_exp = True, figsize = (8, 5)):
        # Figure parameters
        sns.set_theme()
        fig, ax = plt.subplots(figsize=figsize)
        palette = sns.color_palette('colorblind')
        legend_dots = []
        number_repl = 0
        try :
            if plot_exp and not self.melting_data.empty:
                number_repl = len(self.melting_data.Replicate.unique())

                # Plot Experimental Data
                # sns.scatterplot(data=data, x='Temperature', y='Abundance', color="gray", label="Experimental", ax=ax)
                sns.scatterplot(data=self.melting_data, x='Temperature', y='Abundance', hue='Replicate', palette=palette, 
                                alpha=0.7, s=50, edgecolor='k',
                                ax=ax, legend=False)

                # Extract the unique colors used by the scatterplot
                unique_colors = sns.color_palette(palette, number_repl)

                # Create "Proxy" dots for the legend
                for col in unique_colors:
                    d = plt.Line2D([0], [0], alpha=0.7, marker='o',  # type: ignore
                                markerfacecolor=col, markeredgecolor='k', markeredgewidth=1, 
                                markersize=8, linestyle='')
                    legend_dots.append(d)
                    
            # Plot Interpolated Curve
            sns.lineplot(x=np.arange(0, 100.1, 0.1), y=self.fitter.eval(np.arange(0, 100.1, 0.1)), color=palette[number_repl], label="Interpolated Fit", ax=ax)

            # Plot Melting Point Marker
            ax.scatter(self.params.tm_pred, 0.5, color="red", marker='x', s=50, zorder=5, label='Melting Point')

            # Add the Coordinate Label
            ax.annotate(f'Tm = {self.params.tm_pred:.2f}', 
                        xy=(self.params.tm_pred, 0.5), 
                        xytext=(12, -5), 
                        textcoords='offset points',
                        fontsize=8, 
                        color='red', 
                        fontweight='bold')

            # Plot Meltome Tm if available
            if self.data_type == DataType.FLIP:
                ax.scatter(self.params.tm_flip, 0.5, color="purple", marker='x', s=50, zorder=5, label='Meltome Melting Point ')

                # Add the Coordinate Label
                ax.annotate(f'Tm = {self.params.tm_flip:.2f}', 
                            xy=(self.params.tm_flip, 0.5), 
                            xytext=(12, -5), 
                            textcoords='offset points',
                            fontsize=8, 
                            color='red', 
                            fontweight='bold')
            
            # Get the existing handles (the Fit line)
            handles, labels = ax.get_legend_handles_labels()

            ax.set_ylim(0)
            ax.set_xlabel('Temperature (°C)')
            ax.set_title(f"Melting Behaviour of {self.params.pid} with R2 = {self.params.r_squared:.3f}, RMSE = {self.params.rmse:.3f}")
            
            if plot_exp and not self.melting_data.empty:
                # Add our grouped dots tuple to the handles
                ax.legend(
                    handles=[handles[0], tuple(legend_dots), handles[1]], 
                    labels=[labels[0], "Experimental Replicates", labels[1]],
                    handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)},
                    loc='upper right'
                )
                # Resize plot for experimental data
                ax.set_xlim(self.melting_data.Temperature.min() - 5, 100)
                
            else:
                ax.legend(loc='upper right')
                
            
            plt.show()
        except Exception as e:
            self.logger.error(f"Error while plotting figure : {e}")
        
### Functions
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

def get_common_parser(description = "Script Description", epilog = None):
    parser = argparse.ArgumentParser(description=description,
        epilog=textwrap.dedent(epilog) if epilog else None,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # Required arguments
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to input data file"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=".",
        help="Path to output directory for results"
    )
    
    parser.add_argument(
        "-f", "--format",
        type=str,
        required=True,
        choices=['generic', 'mass_spec', 'longf', "flip"],
        help="Define input format and how it will handle by the program"    
    )
    
    # parser.add_argument(
    #     "-p", "--parallel",
    #     action="store_true",
    #     default=True,
    #     help="Defines if input file is processed with parallelization. Defaults to True. Only for FLIP & GENERIC data format."
    # )
    
    parser.add_argument(
        "--n_chunks",
        type=int,
        default=100,
        help="Number of chunks to split the data into for processing. Only for FLIP & GENERIC data format."
    )
    
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=10,
        help="Number of jobs (rows to process) in parallel. Only works if --parallel is set to True."
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level)"
    )
    
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress log output (WARNING level only)"
    )
    
    return parser

def main():
    # Timestamp
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M")
        
    description = "Curve Fitting Tool with External Function"
    epilog = """
    Use example (with longf_example.csv):\n
    $ python .\main.py -i .\example\longf_example.csv -v -f longf \n
    Run main script with longf_example.csv as input, output is CWD and format to process data is longf.
    Will create a main_$timestamp.log file, ./result_main_$timestamp directory where curve results will be stored. \n \n \n
    """
    parser = get_common_parser(description=description, epilog=epilog)
        
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        LOG_LEVEL = logging.DEBUG
    elif args.quiet:
        LOG_LEVEL = logging.WARNING
    else:
        LOG_LEVEL = logging.INFO
    
    LOG_FILE = f'main_{timestamp_str}.log'
    LOG_PATH = os.path.join(args.output, LOG_FILE)
    setup_logging(LOG_PATH, LOG_LEVEL)
    
    output_path = args.output
    result_file = f'results_main_{timestamp_str}'
    output_path = os.path.join(output_path, result_file)
    
    logging.info(f"Log level: {logging.getLevelName(LOG_LEVEL)}")
    
    # Main process for data coming from FLIP Meltome
    if args.format == 'flip':
        data_handler = MeltomeAtlasHandler(args.input, output_path, LOG_LEVEL)
        data_handler.process_parallel(args.n_chunks, args.n_jobs)
    # Main process for data coming from raw ms data     
    if args.format == 'mass_spec':
        return NotImplemented
    # Main process from data coming from longF format
    if args.format == 'longf':
        data_handler = LongFormatHandler(args.input, output_path, LOG_LEVEL)
        data_handler.process(args.n_jobs)
    # Generic processing for data
    if args.format == 'generic':
        return NotImplemented
    else:
        return NotImplemented
    
    
    return 0

if __name__=="__main__":
    main()