# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 22:21:17 2023

@author: Matloob Khushi
"""
import pandas as pd
import requests

# Read the CSV file into a Pandas DataFrame
df = pd.read_excel('uniprot.xlsx')
# add a new column to the DataFrame that checks if the alhaFold model is available
df['AlphaFold'] = False
# Loop through the DataFrames
for i in range(len(df)):

    # Get the URL from the first column
    entry = df.iloc[i, 0]
    filename = "AF-" + entry + "-F1-model_v4.pdb"
    url = "https://alphafold.ebi.ac.uk/files/" + filename
    # Fetch the file from the URL
    response = requests.get(url)
    if response.status_code == 200:
        #Update the AlhaFold column to True
        df.iloc[i, 7] = True

        # Save the file in the subfolder
        with open('PDB_files/' + filename , 'wb') as f:
            print("downloading ... " + url)
            f.write(response.content)

# Save the updated DataFrame to a new excel file
df.to_excel('uniprot_with_alphafold.xlsx', index=False)