import os
import datetime
import logging

from main import setup_logging, LongFormatHandler

import numpy as np
import pandas as pd

def main():
    # Timestamp
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M")
    
    LOG_LEVEL = logging.DEBUG
    LOG_FILE = f'longf_example_{timestamp_str}.log'
    LOG_PATH = os.path.join('', LOG_FILE)
    setup_logging(LOG_PATH, LOG_LEVEL)
    
    # Set output directory, default './results_meltome_$timestamp/'
    output_path = ''
    result_file = f'results_longf_example_{timestamp_str}'
    output_path = os.path.join(output_path, result_file)
    
    logging.info(f"Log level: {logging.getLevelName(LOG_LEVEL)}")
    
    # Load & Process example dataset
    data_handler = LongFormatHandler(os.path.join('example', 'longf_example.csv'), output_path, LOG_LEVEL)
    data_handler.process()
    
    # Load generated results and compare them to expected output
    result_df = pd.read_csv(os.path.join(output_path, 'curve_fit.csv'))
    expected_df = pd.read_csv(os.path.join('example', 'expected_output', 'longf_curve_fit.csv'))
    
    # Check general properties
    assert len(result_df) == len(expected_df), "Mismatching number of entries"
    
    # Drop rows with NANs
    result_df.dropna(axis=0, inplace=True)
    expected_df.dropna(axis=0, inplace=True)
    
    # Check for proteins entries
    assert (result_df.pid == expected_df.pid).all(), "Mismatching proteins entries"
    
    # Check for Tm prediction
    assert (result_df.tm_pred == expected_df.tm_pred).all(), "Mismatching predicted Tm"
    
    logging.info("All good!")
    
    return 0

if __name__ == "__main__":
    main()