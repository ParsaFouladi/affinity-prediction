from representation_functions import *
import os
import sys
import pandas as pd
from mendeleev.fetch import fetch_table
import logging
import re

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

# Get the desired PDB code binding affinity
def get_binding_affinity_info(pdb_code, df):
  try:
    # In order to catch 'Kd=10uM' or 'Ki=10uM' we need to use regex
    # Here we want to catch the binding affinity value and the unit and the type of the binding affinity
    # We are looking for a number followed by a unit (uM, nM, mM, pM, M) and the type of the binding affinity (Ki, Kd)
    
    # First we need to get the binding affinity value
    binding_affinity = float(re.search(r'(\d+\.?\d*)', df[df['PDB code'] == pdb_code]['binding data'].values[0])[0])

    # Next we need to get the unit of the binding affinity
    binding_unit = re.search(r'(uM|nM|mM|pM|M)', df[df['PDB code'] == pdb_code]['binding data'].values[0])[0]

    # Finally we need to get the type of the binding affinity
    binding_type = re.search(r'(Ki|Kd)', df[df['PDB code'] == pdb_code]['binding data'].values[0])[0]
  except Exception as e:
    logging.error(f"Error getting binding affinity for {pdb_code} - {e}")
    sys.exit(1)
  return binding_affinity, binding_unit, binding_type