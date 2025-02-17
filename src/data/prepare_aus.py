import os
import sys
from tqdm import tqdm
sys.path.append(os.getcwd())

def get_data_from_openface(extracted_folder_path):
    # get the .txt file
    txt_files = [file for file in os.listdir(extracted_folder_path) if file.endswith('.txt')]
    
    data = []
    # read the data in .txt file
    for i, txt_file in tqdm(enumerate(txt_files)):
        txt_file_path = os.path.join(extracted_folder_path, txt_file)
        with open(txt_file_path, 'r') as f:
            image_infor = f.readlines()
        
        try:
            image_path = image_infor[0][(len(os.getcwd())+7):-1]
            output_csv = image_infor[6][11:-1]
            output_csv = os.path.join(extracted_folder_path, output_csv)
            data.append([image_path, output_csv])
        except:
            pass
        
    return data