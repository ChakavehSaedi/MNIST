"""
This code loads the MNIST database and the corresponding train and test subsets.
Trains a neural network
Shows the resutls
"""

import os

from data_prepare import data_handler
from model import net_handler

import argparse

################################################
# set up arguments
################################################
parser = argparse.ArgumentParser()
parser.add_argument("-inp", "--input_path", default="/data/input/")
parser.add_argument("-op", "--output_path", default="/data/output/")
parser.add_argument("-prt", "--pre_trained", default=False)
parser.add_argument("-e", "--epochs", default=40)
parser.add_argument("-bs", "--batch_size", default=4)
parser.add_argument("-cnn", "--cnn_info", default=[(1, 10, 5), (10, 20, 2)])
parser.add_argument("-ln", "--linear", default=[(500, 50), (50, 10)])
parser.add_argument("-v", "--verbos", default=False)


args = parser.parse_args()


net_param = {}

net_param['cnn']= args.cnn_info
net_param['linear']=args.linear

epochs = args.epochs
verbos = args.verbos
out_dir = os.getcwd() + args.output_path
log_file = out_dir + "log.txt"
pretrained = args.pre_trained


################################################
# Praparing data
################################################
print(" * Loading train and test data")
with open(log_file,'w') as log:
    log.write(" * Loading train and test data\n")

data_h = data_handler()

################################################
# building the net
################################################
print(" * Building the netwrok")
with open(log_file,'a') as log:
    log.write(" * Building the netwrok\n")
network_h = net_handler(net_param, out_dir, pretrained)

################################################
# train the net
################################################
if not pretrained:
    print(" * Training the netwrok")
    with open(log_file, 'a') as log:
        log.write(" * Training the netwrok\n")
    train_acc, train_loss = network_h.train_net(epochs, data_h.train_loader, verbos, log_file)

################################################
# test the net
################################################
print(" * Testing the netwrok")
with open(log_file, 'a') as log:
    log.write(" * Testing the netwrok\n")
test_acc, train_loss = network_h.test_net(data_h.test_loader, data_h.classes, data_h.plot, log_file)


