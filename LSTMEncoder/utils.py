'''
Utility file containing functions to read in and prep lightcurves for training

Contains: 
    - stack_jagged_array -> create square array from uneven time-series
    - extract_file_names -> create a map of the directory 
    - extract_data -> create data arrays from the files 

'''

import numpy as np 
import fitsio 

from os.path import basename, join
from os import walk
from glob import glob

def split_to_chunk(data, chunk, keys=None): 
    '''
    data - dictionary of data produced by extract_data
    keys - the list of objects to use for dataset
    ''' 

    if not(keys): keys = data.keys() #if empty use all keys for data

    for key in keys: #iterate through the classes and split lcs 
        
        class_data = [] 

        for item in data[key]["FLUX"]: 

            #split each lightcurve into chunks of size chunk 
            for i in range(0, item.shape[0], chunk): 

                if (i + chunk) < item.shape[0]:
                    class_data.append(item[i:(i+chunk)]) 
                else:
                    padding = chunk - (item.shape[0] - i)
                    
                    # Diagnostics for padding commented out
                    #print(item.shape[0], i) 
                    #print(padding) 
                    #x = item[i:item.shape[0]] 
                    #print('pre', x.shape)
                    #x = np.pad(x, padding) 
                    #print('pad', x.shape) 

                    class_data.append(np.pad(item[i:item.shape[0]], (0, padding)))
    
        #Append the new dataset to dictionary 
        data[key]["FLUX"] = class_data 

    return data

def convert_datalist(data, keys=None, verbose=True ):
    '''
    data - dictionary of data produced by extract_data 
    keys - the list of objects to use for dataset
    '''

    if not(keys): keys = data.keys() #if empty use all keys for data

    datalist = [] 
    for key in keys: 
        if verbose: print("Converting %s to list" % key)  
        for item in data[key]["FLUX"]: datalist.append(item) 

    return datalist

def normalise(datalist):
    
    normlist = [] #normalised data

    for item in datalist:

        item[np.isnan(item)] = np.nanmean(item) 
        normed = (item - np.nanmean(item))/np.nanstd(item) 
        normlist.append(normed) 

    return normlist 

def stack_jagged_array(listarr): 
    '''
    Create a rectangular array by padding with nans  
    '''

    #find maximum size of arr
    max_arr = -1

    for item in listarr:
        if len(item) > max_arr: max_arr = len(item) 

    #define a 2-d array nans
    arr = np.zeros((len(listarr), max_arr)) #this is a hefty array 
    arr = arr * np.nan #convert everything to nans 

    #iterate through list and fill 2-d array
    for i, item in enumerate(listarr): 
        arr[i,0:len(item)] = item[:] 

    return arr 

def extract_file_names(dir_, ext=".txt", verbose=True): 
    '''
    Build up a dictionary of all the files in the directory specified by 
    dir_. The dictionary keys are sub-directories of dir_.
    '''

    #Output dictionary containing list of files in each dir 
    dir_structure = {} 

    #Walk through the directory and map to a dict
    for topdir_, subdir_, file_ in walk(dir_):
        
        if verbose: print('Found directory: %s' % topdir_)

        #Add a new key for each dict
        dir_structure[basename(topdir_)] =  [] 

        #iterate through the files and add everything the endswith ext 
        for fname in file_:
            
            #match the extension
            if fname.lower().endswith(ext):
                
                #append the file to the dict
                dir_structure[basename(topdir_)].append(join(topdir_, fname)) 
    
    return dir_structure 

def extract_data(dir_map, verbose=True): 
    '''
    Build up a dictionary of the complete dataset, using the dictionary of keys
    producted by the extract_file_name functions.
    '''
    
    #Data dictionary 
    data = {}

    #Iterate through the keys and extract all the data in the directory
    for key in dir_map.keys():

        if verbose: print("Working on dataset %s" % key) 

        hjd, flux = [], [] 

        #Iterate through the files 
        for file_ in dir_map[key]:

            #Load the data from the file 
            lc_data = np.loadtxt(file_) 
       
            hjd.append(lc_data[:, 0]) 
            flux.append(lc_data[:, 1])

    #Check to make sure directory isn't empty
        if (len(hjd) > 0) and (len(flux) > 0): 

            #Create a single hjd and flux array for each directory
            data[key] = {}
            data[key]["HJD"] = stack_jagged_array(hjd)
            data[key]["FLUX"] = stack_jagged_array(flux) 

    return data 

if __name__ == "__main__": 

    dirs = extract_file_names("/home/alex/Projects/Unsupervised/kepler_q9_variability/") 

    data = extract_data(dirs) 

    for key in data.keys(): 
        print(data[key]["HJD"].shape)
        print(len(dirs[key]))
