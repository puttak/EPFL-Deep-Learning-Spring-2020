#Imports
import torch
import math

#Deactivating autograd
torch.set_grad_enabled(False)

#############################################LAYER MODULES##############################################

#Fully Connected Layer
class  Linear:
    
    def __init__(self,input_size,output_size,gamma=1e-02):
        
        #Initializing the layer's weights
        self.W=torch.empty(output_size,input_size).normal_()
        
        #Initializing the layer's biases
        self.b=torch.empty(output_size,1).normal_()
        
        #Initializing the learning rate
        self.gamma=gamma
        
        #Initializing the gradients (for param())
        self.gradW=None
        self.gradb=None
    
    def  forward(self , x):
        
        #Memorizing the input for the backpropagation
        self.x_old=x
        
        return torch.mm(self.W,x)+self.b
        
    def backward(self , grads):
        
        #Computing the gradient of the loss with respect to W, b and x
        self.gradW=torch.mm(grads,self.x_old.T)
        self.gradb=torch.sum(grads,axis=1).unsqueeze(1)
        gradx=torch.mm(self.W.T,grads)
        
        #Updating the weights and biases (SGD)
        self.W=self.W-self.gamma*self.gradW
        self.b=self.b-self.gamma*self.gradb
        
        return gradx
        
    def  param(self):
        
        return  [(self.W,self.gradW),(self.b,self.gradb)]
    
#Concatenation Of Layers
class  Sequential:
    
    def __init__(self,layers):
        
        #Getting the sequence of layers
        self.layers=layers
    
    def  forward(self , x):
        
        #Input of the next layer (x for the first)
        layer_input=x
        
        #Forwarding the input through each layer in sequence
        for layer in self.layers:
            layer_input=layer.forward(layer_input)
        
        #Returning the output of the last layer
        return layer_input
        
    def backward(self , gradwrtoutput):
        
        #Gradient with respect to the output of the last traversed layer (gradwrtoutput at the beginning)
        grad=gradwrtoutput
        
        #Backward propagating the gradient from the last layer to the first
        for layer in reversed(self.layers):
            grad=layer.backward(grad)
            
        #Returning the gradient of the first layer
        return grad
    
    def  param(self):
        
        params=[]
        
        #Getting the param list of each layer
        for layer in self.layers:
            
            for param in layer.param():
                
                params.append(param)
        
        return  params

#############################################ACTIVATION MODULES##############################################

#Rectifier Activation Module
class  ReLU:
    
    def  forward(self , s):
        
        #Saving s for the backward
        self.s=s
        
        #Setting all negative values to 0
        s[s<0]=0
        
        return s
        
    def backward(self , gradx):
        
        #Computing the gradient with respect to s
        grads=gradx*(self.s>0) #∇s max(s,0) = s>0
        
        return grads
        
    def  param(self):
        return  []

#Hyperbolic Tangent Activation Module
class  Tanh:
        
    
    def  forward(self , s):
        
        #Saving s for the backward
        self.s=s
        
        return 1-2/(1+math.exp(1)**(2*s))
        
    def backward(self , gradx):
        
        #Computing the gradient with respect to s
        grads=gradx*(4/(math.exp(1)**(self.s)+math.exp(1)**(-self.s))) #∇s tanh(s) = 4/(e^s+e^-s)
        
        return grads
        
    def  param(self):
        return  []

#Sigmoid Activation Module
class  sigmoid:
        
    
    def  forward(self , s):
        
        #Saving s for the backward
        self.s=s
        
        return 1/(1+math.exp(1)**(-self.s))
        
    def backward(self , gradx):
        
        #Computing the gradient with respect to s
        #∇s sigmoid(s) = sigmoid(s)*(1-sigmoid(s))
        grads=gradx*((1/(1+math.exp(1)**(-self.s)))*(1-1/(1+math.exp(1)**(-self.s)))) 
        
        return grads
        
    def  param(self):
        return  []
    
#############################################LOSS MODULES##############################################
    
#Mean Squared Error Loss Module
class  LossMSE:
    
    def  forward(self , s, y):
        
        #Saving both s and y for the backpropagation
        self.s=s
        self.y=y
        return torch.mean((s-y)**2)
        
    def backward(self):
        
        #Computing the gradient of the loss with respect to s
        grads=2*(self.s-self.y)/(self.s.shape[1]*self.s.shape[0]) #∇s MSE(s,y)=2*(s-y)/N
        return grads
        
    def  param(self):
        return  []
    
#Mean Absolute Error Loss Module
class  LossMAE:
    
    def  forward(self , s, y):
        
        #Saving both s and y for the backpropagation
        self.s=s
        self.y=y
        
        #Computing the difference between the actual output and the target output
        delta=s-y
        
        #Inverting the signs when the value is negative to get the absolute values
        delta[delta<0]=-delta[delta<0]
        
        return torch.mean(delta)
        
    def backward(self):
        
        #Computing the difference between the actual output and the target output
        delta=self.s-self.y
        
        #Computing the gradient of the loss with respect to s
        grads=((delta<0)*(-1.)+(delta>=0))/(self.s.shape[1]*self.s.shape[0]) #∇s MAE(s,y)=sign(s-y)/N
        return grads
        
    def  param(self):
        return  []