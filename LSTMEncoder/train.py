import numpy as np
import torch 
import torch.nn as nn
import pickle 

from torch.utils.data import DataLoader
from vanilla_autoencoder import VanillaAE
from dataprep import AutoEncoderDataset

if __name__ == "__main__": 

    #Setup DEVICE
    seed = 99 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    print(device)

    #Setup the model
    INPUT_DIM = 200 #must be bigger than 1 
    model = VanillaAE(INPUT_DIM)
    #loss_function  = nn.BCELoss(reduce=False) 
    loss_function = nn.MSELoss()
    print("Model initialised.") 

    # Some information about the model 
    # print(torch.cuda.memory_cached(device=device))
    #print(torch.cuda.memory_allocated(device=device))
   
    num_parameters = 0 
    for parameter in model.parameters():
             num_parameters += parameter.view(-1).size()[0]
    print("Number of parameters: %i" % num_parameters) 
    print(model) 

    # Load the data and split into sets 
    with open("ae_dataset.pkl", "rb") as f:
        data = pickle.load(f) #load the autoencoderdatasets 
 
    #Split the dataset into train and test
    #training dataset size, testing dataset size
    lengths = [int(np.floor(0.99*len(data))), int(np.ceil(0.01*len(data)))] 
    train_data, test_data = torch.utils.data.random_split(data, lengths)
    print("Training data length %i, test data lenght %i" % (lengths[0], lengths[1])) 
    
    training_data = DataLoader(train_data, batch_size=100, shuffle=True, drop_last=True) 
    testing_data = DataLoader(test_data, batch_size=1, shuffle=True, drop_last=True) 

    print("Training data loaded.")
    #Setup the optimizer 
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
   
    #Begin the training 
    model.train() 

    for epoch in range(25): 

        for i, (labels, timeseries) in enumerate(training_data):

            #print(sum(np.isnan(labels)))
            #print(labels, timeseries) 

            #print(labels, timeseries)
            #print(labels.size(), timeseries.size()) 
            model.zero_grad()
           
            prediction = model(timeseries) 
            prediction = prediction.view_as(labels) 
            
            loss = loss_function(prediction, labels) 

            loss.backward()
            optimizer.step()          
        print("Epoch %i, batch %i, loss: %f" % (epoch, i, loss)) 
         
        #if epoch % 10 == 0.0: 
        #    torch.save(model, "model_%i.pkt" % epoch)
        #    torch.save(h, "hidden_%i.pkt" % epoch) 
        #    print("Model saved!")

    #Show the test results - this needs to be moved from here 
    #import matplotlib; matplotlib.use("qt4cairo")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    
    print('here')

    with PdfPages("vanilla_autoencoder.pdf") as pdf:

        for j, (test_label, test_timeseries) in enumerate(testing_data): 
        
            print(j) 
            if j < 500: 

                plt.figure(figsize=(8, 6))
                plt.scatter(range(0, 200), test_timeseries, label="Input")
                plt.scatter(range(0, 200), model(test_timeseries).detach().numpy(), color="r", label="AE output") 
                plt.xlabel('Data point index') 
                plt.ylabel('Normalised flux')
                plt.legend()
                plt.grid('on')
                plt.title('Test example #%i' % j) 
                plt.tight_layout()
                pdf.savefig()
                plt.close('all') 




        



