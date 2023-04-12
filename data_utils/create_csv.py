# Code to create csv files from .mat files of annotations


import csv
import os
import argparse
import scipy.io 



def create_csv(args):
    mat = scipy.io.loadmat(args.mat_file)

    annotations = mat['annotations'][0]
    image_names = [str(annotation[-1][0]) for annotation in annotations]
    class_labels = [int(annotation[-2][0][0]) - 1 for annotation in annotations]

    data = list(zip(image_names, class_labels))

    file_name = os.path.join(args.dest_path, args.csv_name)

    with open(file_name, 'w', newline = '') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['name', 'class_id'])
        writer.writerows(data)

    print(f"Successfully created {args.csv_name}.")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mat_file", type = str, help = "annotation file of the data", required = True)
    parser.add_argument("--dest_path", type = str, help = "destination location of the csv file", required = True)
    parser.add_argument("--csv_name", type = str, help = "name of the output csv file", required = True)

    args = parser.parse_args()

    create_csv(args)