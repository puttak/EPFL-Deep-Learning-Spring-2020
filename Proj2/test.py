#Imports
import torch
import math
from mini_torch import *

#Helper function generating 'size' inputs and targets
def dataset_generator(size):
    input_=torch.rand((1000,2))
    target=(torch.sum((input_-0.5)**2,axis=1)*(2*math.pi)<1)
    return input_,torch.stack([~target,target]).T*1

#Helper function computing the accuracy of our predictions
def accuracy(pred,target):
    return 1.*torch.sum((torch.max(pred,0)[1])==(torch.max(target,0)[1]))/target.shape[1]

#Useful constants
input_size=1000
batch_size=25 #Good tradoff, gives high convergence speed
epochs=4000 #High number, gives enough time for the model to converge
gamma=0.01 # Learning rate shared by all 'Linear' modules

#Setting a seed for reproducibility
torch.manual_seed(0)

#Generating our train and test datasets
train_input, train_targets = dataset_generator(input_size)
test_input, test_targets = dataset_generator(input_size)

#Initializing our model
model=Sequential([Linear(2,25,gamma),Tanh(),Linear(25,25,gamma),Tanh(),Linear(25,25,gamma),Tanh(),Linear(25,2,gamma)])

#Training our model
for epoch in range(epochs):
    
    #Initializing a loss accumulator to compure the average loss of an SGD iteration
    avg_loss=0
    
    #Running an SGD iteration
    for b in range(math.ceil(input_size/batch_size)):
        
        #Computing the lower and upper indices corresponding to the current batch
        low=b*batch_size
        high=min((b+1)*batch_size,input_size)
        
        #Getting the input and target batches
        input_batch=train_input[low:high].T
        target_batch=train_targets[low:high].T
        
        #Forward step
        pred=model.forward(input_batch)
        
        #Computing the loss
        loss_func=LossMSE()
        loss=loss_func.forward(pred,target_batch)
        
        #Backward step
        grad=loss_func.backward()
        model.backward(grad)
        
        #Updating the loss accumulator
        avg_loss=avg_loss+loss.item()
    
    #Dividing the accumulated batch losses to get the average loss of the SGD step
    avg_loss=avg_loss/(input_size//batch_size+1)
    
    #Printing the results of the current iteration
    print("The average training loss at iteration {} is {}".format(epoch+1,avg_loss))

#Computing the final training loss
pred=model.forward(train_input.T)
final_loss=LossMSE().forward(pred,train_targets.T)
print("The final training loss is ", final_loss)

#Computing the final test loss
pred=model.forward(test_input.T)
final_loss=LossMSE().forward(pred,test_targets.T)
print("The test loss is ", final_loss)

#Printing the test accuracy to check the actual classification capabilities of the model 
print("The test accuracy is ", accuracy(pred,test_targets.T))