import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

import matplotlib.pyplot as plt
import dlc_practical_prologue as prologue

from networks import *
from helper_functions import train_model, compute_nb_errors, get_stats

# Adam with learning rate 0.001 (default) ; batch size 25 ; number of epochs 100 ; 20 ROUNDS
torch.manual_seed(0)

# Setting the number of rounds and models
nb_rounds = 20
nb_models = 4

# Display information messages about which models are trained
if not prologue.args.trainall:
    print("Training only NetWSAL, the best architecture, on {:d} rounds".format(nb_rounds))
    print("To train all 4 defined architectures instead, run test.py with the --trainall flag")
    nb_models = 1
else:
    print("Training NetBase, NetWS, NetAL and NetWSAL, the 4 architectures, on {:d} rounds".format(nb_rounds))

# Setting the hyperparameters
criterion = nn.CrossEntropyLoss()
mini_batch_size = 25
nb_epochs = 100

# Creating the 2D tensor of recorded error rates
test_errors = torch.zeros(nb_models, nb_rounds)

# Setting x-axis and y-axis of the plot
fig = plt.figure()
sub_plt = fig.add_subplot(1, 1, 1)
sub_plt.set_xlabel('Round Number', labelpad=10)
sub_plt.set_ylabel('Error rate (%)', labelpad=10)
sub_plt.set_yscale('linear')

# Executing computation
for j in range(nb_rounds):
    
    print('Starting round {:d}'.format(j + 1))

    # Generate train and test data sets with 1000 samples
    nb_samples = 1000
    train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(nb_samples)
    
    # Setting graph parameters
    if not prologue.args.trainall:
        graph_param = [
            (0, NetWSAL()),
        ]
    else:
        graph_param = [
            (0, NetBase()),
            (1, NetWS()),
            (2, NetAL()),
            (3, NetWSAL()),
        ]

    # Loop to compute loss for each model
    for i, mod in graph_param:
        optimizer = optim.Adam(mod.parameters())
        test_error = get_stats(mod, optimizer, criterion, mini_batch_size, nb_epochs,
                               train_input, train_target, train_classes, test_input, test_target, test_classes)
        test_errors[i][j] = test_error
        
    print('Round {:d} done'.format(j + 1))

# Setting colors and labels
if not prologue.args.trainall:
    colors = ['tab:orange']
    labels = ['WS + AL']
else:
    colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']
    labels = ['Base', 'WS', 'AL', 'WS + AL']

# Plotting accuracy
for i in range(nb_models):
    sub_plt.plot(range(1, nb_rounds + 1), test_errors[i], color=colors[i], label=labels[i])

# Including legend to the plot
sub_plt.legend()

# Formatting axis
plt.xticks(range(1, nb_rounds + 1))
if not prologue.args.trainall:
    plt.yticks(range(0, 30, 5))

# Setting list of trained models
if not prologue.args.trainall:
    nets = ['NetWSAL']
else:
    nets = ['NetBase', 'NetWS  ', 'NetAL  ', 'NetWSAL']

# Printing mean and standard deviation of each model test error rates
print('\nMean of error rates for each model is : \n' + ''.join(
    '- {} : {:.10f}\n'.format(net_name, mean_err) for net_name, mean_err in zip(nets, test_errors.mean(1))))
print('Standard Deviation of error rates for each model is : \n' + ''.join(
    '- {} : {:.10f}\n'.format(net_name, std_err) for net_name, std_err in zip(nets, test_errors.std(1))))

# Saving figure
if not prologue.args.trainall:
    plt.savefig('NetWSAL_architecture_performance.png')
    print("Graph for NetWSAL architecture performance saved under the name NetWSAL_architecture_performance.png")
else:
    plt.savefig('All_architectures_performance.png')
    print("Graph for all architectures performance saved under the name All_architectures_performance.png")
