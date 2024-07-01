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
try:
  #Add logging here
  

  ptable = fetch_table("elements")
  cols = ["symbol",
      "vdw_radius"
  ]
  ptable=ptable[cols]
  ptable.dropna(inplace=True)
  van_dict=ptable.set_index('symbol').T.to_dict('index')['vdw_radius']
  logging.info("Fetched van der Waals radii from Mendeleev database")
except:
  #Add logging here
  van_dict={'H': 110.00000000000001,'He': 140.0,'Li': 182.0,'Be': 153.0,'B': 192.0,'C': 170.0,'N': 155.0,'O': 152.0,'F': 147.0,'Ne': 154.0,'Na': 227.0,'Mg': 173.0,
 'Al': 184.0,'Si': 210.0,'P': 180.0,'S': 180.0,'Cl': 175.0,'Ar': 188.0,'K': 275.0,'Ca': 231.0,'Sc': 215.0,'Ti': 211.0,'V': 206.99999999999997,'Cr': 206.0,'Mn': 204.99999999999997,
 'Fe': 204.0,'Co': 200.0,'Ni': 197.0,'Cu': 196.0,'Zn': 200.99999999999997,'Ga': 187.0,'Ge': 211.0,'As': 185.0,'Se': 190.0,'Br': 185.0,'Kr': 202.0,'Rb': 303.0,'Sr': 249.00000000000003,
 'Y': 231.99999999999997,'Zr': 223.0,'Nb': 218.00000000000003,'Mo': 217.0,'Tc': 216.0,'Ru': 213.0,'Rh': 210.0,'Pd': 210.0,'Ag': 211.0,'Cd': 218.00000000000003,'In': 193.0,
 'Sn': 217.0,'Sb': 206.0,'Te': 206.0,'I': 198.0,'Xe': 216.0,'Cs': 343.0,'Ba': 268.0,'La': 243.00000000000003,'Ce': 242.0,'Pr': 240.0,'Nd': 239.0,'Pm': 238.0,'Sm': 236.0,'Eu': 235.0,
 'Gd': 234.0,'Tb': 233.0,'Dy': 231.0,'Ho': 229.99999999999997,'Er': 229.0,'Tm': 227.0,'Yb': 225.99999999999997,'Lu': 224.00000000000003,
 'Hf': 223.0,'Ta': 222.00000000000003,'W': 218.00000000000003,'Re': 216.0,'Os': 216.0,'Ir': 213.0,'Pt': 213.0,'Au': 214.0,'Hg': 223.0,'Tl': 196.0,'Pb': 202.0,'Bi': 206.99999999999997,
 'Po': 197.0,'At': 202.0,'Rn': 220.00000000000003,'Fr': 348.0,'Ra': 283.0,'Ac': 247.00000000000003,'Th': 245.00000000000003,'Pa': 243.00000000000003,'U': 241.0,'Np': 239.0,
 'Pu': 243.00000000000003,'Am': 244.0,'Cm': 245.00000000000003,'Bk': 244.0,'Cf': 245.00000000000003,'Es': 245.00000000000003,'Fm': 245.00000000000003,'Md': 246.0,'No': 246.0,'Lr': 246.0}

logging.info("Used default van der Waals radii")

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

    # get the resoltion of the structure
    resolution = float(df[df['PDB code'] == pdb_code]['resolution'].values[0])
  except Exception as e:
    logging.error(f"Error getting binding affinity for {pdb_code} - {e}")
    sys.exit(1)
  return binding_affinity, binding_unit, binding_type, resolution

