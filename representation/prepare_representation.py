from representation_functions import *
import os
import sys
import pandas as pd
from mendeleev.fetch import fetch_table
import logging

#Initialize logger
logging.basicConfig(filename='representation_log.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s:%(message)s')


# read the csv file
def read_csv_file(file_path):
  try:
    df = pd.read_csv(file_path)
  except Exception as e:
    logging.error(f"Error reading file: {file_path} - {e}")
    sys.exit(1)
  return df