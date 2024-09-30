# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 22:21:17 2023

@author: Matloob Khushi
"""
import pandas as pd
import requests

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv('uniprot.csv', header=0)

# Loop through the DataFrames
for i in range(len(df)):

    # Get the URL from the first column
    entry = df.iloc[i, 0]
    filename = "AF-" + entry + "-F1-model_v4.pdb"
    url = "https://alphafold.ebi.ac.uk/files/" + filename
    # Fetch the file from the URL
    response = requests.get(url)

    # Save the file in the subfolder
    with open('PDB_files/' + filename , 'wb') as f:
        print("downloading ... " + url)
        f.write(response.content)
