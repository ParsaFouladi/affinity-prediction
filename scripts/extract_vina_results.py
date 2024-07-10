import csv
import os
import re
import sys

def extract_information_from_pdbqt(input_file):
    result_line = None
    with open(input_file, 'r') as file:
        for line in file:
            if line.startswith('REMARK VINA RESULT:'):
                result_line = re.findall(r"[-+]?\d*\.\d+|\d+", line.strip())
                break
    return result_line

def process_files_in_folder(folder_path, output_csv):
    # Open CSV file for writing
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['File Name', 'Remark Vina Result'])

        # Iterate over files in the folder
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.pdbqt'):
                file_path = os.path.join(folder_path, file_name)

                # Extract information from the current file
                result_values = extract_information_from_pdbqt(file_path)

                # Write extracted information to CSV file
                csv_writer.writerow([file_name, ' '.join(result_values)])

def main():
    # input_folder = '/mainfs/home/mk6n23/mkdata/docked_complexes/job_test'  # Replace 'input.pdbqt' with the path to your input file
    # output_csv = '/mainfs/home/mk6n23/mkdata/docked_complexes/new_parsa_set/csv_output/twenty_box_output.csv'    # Output CSV file name

    input_folder = sys.argv[1]
    output_csv = sys.argv[2]

    # Process files in the specified folder
    process_files_in_folder(input_folder, output_csv)

    print('Information extracted and saved to {output_csv}.')

if __name__ == "__main__":
    main()