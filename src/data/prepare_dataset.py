import os
import sys
import argparse
import pandas
import numpy as np
from tqdm import tqdm
sys.path.append(os.getcwd())

from src.data.prepare_aus import *

def get_data(extracted_folder_path):
    data = get_data_from_openface(extracted_folder_path)
    
    processed_data = []
    for dt in tqdm(data):
        image_path  = dt[0]
        au_file_path = dt[1]
        emotions_vec = pandas.read_csv(au_file_path)
        origin_au_vec = emotions_vec.filter(regex=r'AU\d+_r').to_numpy()[0]
        processed_data.append([image_path, origin_au_vec])
    
    return processed_data

def get_dataset(processed_data):
    len_data = len(processed_data)
    dataset = []
    
    for i in tqdm(range(len_data)):
        idx = np.random.randint(0, len_data-1)
        desired_au_vec = np.copy(processed_data[idx][1])
        desired_au_vec += np.random.uniform(-0.1, 0.1, desired_au_vec.shape)
        dataset.append([processed_data[i][0], processed_data[i][1], np.round(desired_au_vec, 2)])
        
    return dataset
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aus_path', type=str, default=None, help="processed folder when extract the au from openface")
    parser.add_argument('--labelfile_path', type=str, default=None, help="path of file labels.txt extracted")
    
    args = parser.parse_args()
    
    if args.aus_path and args.labelfile_path:
        processed_data = get_data(args.aus_path)
        dataset = get_dataset(processed_data)
        
        labels_filepath = os.path.join(args.labelfile_path, 'labels.txt')
        with open(labels_filepath, 'w') as f:
            for dt in dataset:
                dt[1] = list(dt[1])
                dt[2] = list(dt[2])
                f.write(f"{dt[0]}\t{dt[1]}\t{dt[2]}\n")
                
    elif args.aus_path is None:
        print("Can't find the path to aus folder")
    else:
        print("Can't file the path to save the labels.txt file")
        
if __name__ == '__main__':
    main()
    