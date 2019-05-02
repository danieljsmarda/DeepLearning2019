import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F

import time
import copy

""" FRAMEWORK FOR INPUT AS TWO SINGLE CHANNEL IMAGES """
""" In this framework, the network is first trained to recognize the digits of each image from each pair and with the help of the class labels. To do so, we use the class labels provided and use a CrossEntropyLoss to maximize the response of the correct digit. Once the network can predict the digits, we compare the digits and define if they are a pair or not """

nb_classes = 10
nb_input_channels = 1

def prep_input_vanilla(input_):
    """ Transforms the 1 image 2 channels input in 2 images 1 channel """
    new_input = input_.view(-1,1,14,14)
    return new_input

######################################################################

def train_model_1C(model, train_input, train_classes, optimizer, mini_batch_size=1000, nb_epochs=300):
    """ Network is a classifier: it is trained to predict the digit from the image """
    criterion = torch.nn.CrossEntropyLoss()
    train_input = prep_input_vanilla(train_input)
    train_target = train_classes.flatten() # the target are the class labels 
    nb_samples = len(train_input)
    
    since = time.time()
    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for e in range(0, nb_epochs):
        
        for phase in ['train', 'val']:
            if phase == 'train': model.train()
            else: model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            for b in range(0, train_input.size(0), mini_batch_size):
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(train_input.narrow(0, b, mini_batch_size))
                    target = train_target.narrow(0, b, mini_batch_size)
                    
                    loss = criterion(output, target)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * train_input.size(0)
                running_corrects += torch.sum(torch.max(output, 1)[1] == target)       

            epoch_loss = running_loss / nb_samples
            epoch_acc = running_corrects.double() / nb_samples
            
            if (e % 100 == 99):
                print('phase: %s, epoch: %d, loss: %.5f, acc: %.4f' %
                      (phase, e+1, epoch_loss, epoch_acc))
                
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                
    time_elapsed = time.time() - since
    print('Training complete in %.0f min %.0f s' % (time_elapsed // 60, time_elapsed % 60))
    print('Best val acc: %.4f' % (best_acc))
    
    model.load_state_dict(best_model_wts)
    return model, time_elapsed

######################################################################

def compute_nb_errors(model, input_, target, mini_batch_size=1000):
    nb_errors = 0  
    for b in range(0, input_.size(0), mini_batch_size):
        output = model(input_.narrow(0, b, mini_batch_size))
        target_classes = target.narrow(0, b, mini_batch_size)
        _, predicted_classes = torch.max(output, 1)
        nb_errors += (predicted_classes != target_classes).sum().item()      
    return nb_errors

def compare_pairs(model, input_):    
    tensor_a = torch.max(model(input_[:,0,:,:].view(-1,1,14,14)), 1)[1]
    tensor_b = torch.max(model(input_[:,1,:,:].view(-1,1,14,14)),1)[1]
    return torch.le(tensor_a, tensor_b)

def test_model_1C(model, test_input, test_target, test_classes, mini_batch_size=1000):
    model.eval()
    test_input_vanilla = prep_input_vanilla(test_input)
    test_classes_target = test_classes.flatten()
    
    # Number of digits incorrectly identified
    nb_errors_digits = compute_nb_errors(model, test_input_vanilla, test_classes_target, mini_batch_size)
    # Test accuracy on task = predicting digits
    acc_digits = 1 - nb_errors_digits / len(test_input_vanilla)
    
    # Number of wrong predictions (first digit less than or equal to the second)
    test_output_pairs = compare_pairs(model, test_input).type(torch.LongTensor)
    nb_errors_pairs = (test_output_pairs != test_target).sum().item()
    # Test accuracy on task = comparison of pairs
    acc_pairs = 1 - nb_errors_pairs / len(test_input)
    
    return acc_digits, acc_pairs

######################################################################
import csv

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def check_overwrite(filename, model, row_to_write):
    overwrite = False
    with open(filename, 'r') as readFile:
        reader = csv.reader(readFile)
        row_list = list(reader)
        for index, row in enumerate(row_list):
            if row[0] == model.name: 
                row_list[index] = row_to_write
                overwrite = True           
                break
    with open(filename, 'w') as writeFile:
        print("Overwriting file")
        writer = csv.writer(writeFile)
        writer.writerows(row_list)
    writeFile.close()
    readFile.close()
    return overwrite
    
def write_to_csv(filename, model, test_results):
    nb_params = count_parameters(model)
    row = [model.name, nb_params, round(test_results[0], 2), 
           round(test_results[2], 4), round(test_results[3], 4), 
           round(test_results[4], 4), round(test_results[5], 4)]
    
    try: file = open(filename, 'r')
    except FileNotFoundError:
        csvData = [['Model', 'Number of parameters', 'Training time', 
                    'Mean digits accuracy (test set)', 'Std digits accuracy', 
                    'Mean accuracy (test set)', 'Std accuracy']]
        with open('1channel2images.csv', 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(csvData)
        csvFile.close()
        return
        
    overwrite = check_overwrite(filename, model, row)
    if overwrite == False:    
        with open(filename, 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()

######################################################################

def compute_properties(lst):
    mean = sum(lst) / len(lst)
    variance = sum([(e-mean)**2 for e in lst]) / (len(lst)-1)
    return mean, variance ** (1/2)

def multiple_training_runs(model, nb_runs, optimizer, train_input, train_classes,
                           test_input, test_target, test_classes, mini_batch_size=1000, nb_epochs=300):
    list_time = []
    list_acc_digits = []
    list_acc_pairs = []
    
    initial_model_wts = copy.deepcopy(model.state_dict())
            
    for i in range(nb_runs):
        model.load_state_dict(initial_model_wts)
        model, time_elapsed = train_model_1C(model, train_input, train_classes, optimizer, mini_batch_size=mini_batch_size, nb_epochs=nb_epochs)
        list_time.append(time_elapsed)
        
        acc_digits, acc_pairs = test_model_1C(model, test_input, test_target, test_classes)
        list_acc_digits.append(acc_digits)
        list_acc_pairs.append(acc_pairs)
        
    mean_time, std_time = compute_properties(list_time)
    mean_acc_digits, std_acc_digits = compute_properties(list_acc_digits)
    mean_acc_pairs, std_acc_pairs = compute_properties(list_acc_pairs)
    
    return mean_time, std_time, mean_acc_digits, std_acc_digits, mean_acc_pairs, std_acc_pairs
