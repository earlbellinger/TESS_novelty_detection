'''
Script to prepare the data into a suitable format for training 
'''

import pickle
import numpy as np 
import torch 
import matplotlib.pyplot as plt 

from torch.utils.data import Dataset, DataLoader
from utils import extract_file_names, extract_data, convert_datalist, split_to_chunk, normalise

class AutoEncoderDataset(Dataset): 

    def __init__(self, data): 
        '''
            data - input list of lightcurves 
        '''
        torch_data_list = [] 
        torch_label_list = []

        for item in data: 

            #Prepare the main data
            torch_data_list.append(
                    torch.from_numpy(item).float()
                    ) 

            #Prepare the labelled data 
            torch_label_list.append(
                    torch.from_numpy(item).float()
                    )

        self.data = torch_data_list
        self.label = torch_label_list 

    def __len__(self): 
        return len(self.data) 

    def __getitem__(self, idx):   
        return (self.data[idx], self.label[idx])

if __name__ == "__main__": 

    #Parameters for the dataset 
    chunk_size = 200

    #Load in the input data 
    dirs = extract_file_names("/home/alex/Projects/Unsupervised/kepler_q9_variability/") 

    data = extract_data(dirs) 

    data = split_to_chunk(data, chunk_size) 
    
    datalist = convert_datalist(data) 

    datalist = normalise(datalist) 
 
    data_arr = np.vstack(datalist) 

    with open("autoencoder_dataset.pkl", "wb") as f:
        pickle.dump(data_arr, f) 
        print("Written ae_dataset.pkl")



    ### Plotting
    #for i in range(0, 100): 
    #    print(datalist[i].shape) 
    #    plt.figure()
    #    plt.title('Segment %i' % i)
    #    plt.scatter(range(0, chunk_size), datalist[i])
    #    plt.ylabel('Un-normalised flux') 
    #    plt.xlabel('Data point index') 
    #    plt.show()
    #    plt.close('all') 
    ###

    #Prepare a torch dataset
    #ae_dataset = AutoEncoderDataset(datalist) 
    
    #Save the dataset with pickle
    #with open("ae_dataset.pkl", "wb") as f:
    #    pickle.dump(ae_dataset, f) 
    #    print("Written ae_dataset.pkl")

