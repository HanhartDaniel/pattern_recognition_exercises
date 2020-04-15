"""
CNN with 3 conv layers and a fully connected classification layer
PATTERN RECOGNITION EXERCISE:
Fix the three lines below marked with PR_FILL_HERE


CNN -> Convolution neural network normaly consist of multiple convolutional layers followd
with a pooling layer (mostly max_pooling). afterwards fully connected layers
advantages to MLP are:
    -2D, 3D Neurol construct
    -shared weights
    -lokal conectivity



"""
import torch
import torch.nn as nn
import numpy as np
import time



class Flatten(nn.Module):
    """
    Flatten a convolution block into a simple vector.

    Replaces the flattening line (view) often found into forward() methods of networks. This makes it
    easier to navigate the network with introspection
    """
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class PR_CNN(nn.Module):
    """
    Simple feed forward convolutional neural network

    Attributes
    ----------
    expected_input_size : tuple(int,int)
        Expected input size (width, height)
    conv1 : torch.nn.Sequential
    conv2 : torch.nn.Sequential
    conv3 : torch.nn.Sequential
        Convolutional layers of the network
    fc : torch.nn.Linear
        Final classification fully connected layer

    """

    def __init__(self, **kwargs):
        """
        Creates an CNN_basic model from the scratch.

        Parameters
        ----------
        output_channels : int
            Number of neurons in the last layer
        input_channels : int
            Dimensionality of the input, typically 3 for RGB
        """
        super(PR_CNN, self).__init__()

        # PR_FILL_HERE: Here you have to put the expected input size in terms of width and height of your input image
        self.expected_input_size = (28,28)

        # First layer
        self.conv1 = nn.Sequential(
            # PR_FILL_HERE: Here you have to put the input channels, output channels ands the kernel size
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=3),
            nn.LeakyReLU()
        )

        # Classification layer
        self.fc = nn.Sequential(
            Flatten(),
            # PR_FILL_HERE: Here you have to put the output size of the linear layer. DO NOT change 1536!
            nn.Linear(1536, 10)
        )

    def forward(self, x):
        """
        Computes forward pass on the network

        Parameters
        ----------
        x : Variable
            Sample to run forward pass on. (input to the model)

        Returns
        -------
        Variable
            Activations of the fully connected layer
        """
        x = self.conv1(x)
        x = self.fc(x)
        return x






def main():
   
    start_t = time.time()
    train = open(r"mnist_train.csv","r")
    train_set_data = list()
    train_set_label = list()
   
    reduced_set = "yes"
    
    if reduced_set == "yes":
        
    
        
        for i in train: 
            if len(train_set_data) < 1000:   
                train_set_data.append(list(map(int,(i.strip().split(","))))[1:])
                label = list(map(int,(i.strip().split(","))))[0]
                lab = list()
                for j in range(0,10):
                    if label == j:
                        lab.append(1)
                    else:
                        lab.append(0)
                train_set_label.append(lab)
          
        train.close()
        train_set_data_np = np.asarray(train_set_data)
        train_set_data_picture = list()
        for i in range(0, len(train_set_data_np)):
            train_set_data_picture.append(train_set_data_np[i].reshape(28,28))
        
        train_set_data_np = train_set_data_np
        train_set_label_np = np.asarray(train_set_label)
        
        test = open(r"mnist_test.csv","r")
        test_set_data = list()
        test_set_label = list()
    
        
        
        for i in test:
            if len(test_set_data) < 1000:   
                test_set_data.append(list(map(float,(i.strip().split(","))))[1:])
                label = list(map(int,(i.strip().split(","))))[0]
                lab = list()
                for j in range(0,10):
                    if label == j:
                        lab.append(1)
                    else:
                        lab.append(0)
                test_set_label.append(lab)
            
        test.close()
        test_set_data_np = np.asarray(test_set_data)
        test_set_data_picture = list()
        for i in range(0, len(test_set_data_np)):
            test_set_data_picture.append(test_set_data_np[i].reshape(28,28))
        test_set_data_np = test_set_data_np
        test_set_label_np = np.asarray(test_set_label)
        
        
 
    else:
        for i in train: 
            train_set_data.append(list(map(int,(i.strip().split(","))))[1:])
            label = list(map(int,(i.strip().split(","))))[0]
            lab = list()
            for j in range(0,10):
                if label == j:
                    lab.append(1)
                else:
                    lab.append(0)
            train_set_label.append(lab)
      
        train.close()
        train_set_data_np = np.asarray(train_set_data)
        train_set_data_np = train_set_data_np.reshape(28,28)
        train_set_label_np = np.asarray(train_set_label)
        
        test = open(r"mnist_test.csv","r")
        test_set_data = list()
        test_set_label = list()
    
    
        for i in test: 
            test_set_data.append(list(map(float,(i.strip().split(","))))[1:])
            label = list(map(int,(i.strip().split(","))))[0]
            lab = list()
            for j in range(0,10):
                if label == j:
                    lab.append(1)
                else:
                    lab.append(0)
            test_set_label.append(lab)
            
        test.close()
        test_set_data_np = np.asarray(test_set_data)
        test_set_data_np = test_set_data_np.reshape(28,28)
        test_set_label_np = np.asarray(test_set_label)
        
    print("reading in Data finished after ", time.time()-start_t, " seconds")
    print( train_set_data_np.shape)
    print( test_set_data_np.shape)

    Network = PR_CNN()

    train_set_data_picture = np.asarray(train_set_data_picture)
   
    train_set_data_picture = train_set_data_picture[:,np.newaxis,:,:]
    
    print(train_set_data_picture.shape)
    
    Network.forward(torch.Tensor(train_set_data_picture))
    

main()