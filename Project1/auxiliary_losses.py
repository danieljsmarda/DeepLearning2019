import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F

import time
import copy

""" FRAMEWORK FOR USING AUXILIARY LOSSES """

nb_classes_digits = 10
nb_classes_pairs = 2

def train_model_aux(model, train_input, train_target, train_classes, optimizer, mini_batch_size=1000, nb_epochs=300):
    
    criterion1 = torch.nn.CrossEntropyLoss()
    criterion2 = torch.nn.BCEWithLogitsLoss()
    
    alpha = 0.3
    
    print("train_input.shape = ", train_input.shape)
    
    train_target = train_target.type(torch.FloatTensor).view(-1, 1)
    
    print("train_classes.shape = ", train_classes.shape)
    print("train_classes[0] = ", train_classes[0])

    print("train_target.shape = ", train_target.shape)
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
                    x0, x1, x = model(train_input.narrow(0, b, mini_batch_size))
                    target = train_target.narrow(0, b, mini_batch_size)
                    classes = train_classes.narrow(0, b, mini_batch_size)
                    
                    loss0 = criterion1(x0, classes[:,0])
                    loss1 = criterion1(x1, classes[:,1])
                    loss2 = criterion2(x, target)
                    loss = loss2 + alpha*(loss0+loss1)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * train_input.size(0)
                output_to_prediction = torch.ge(torch.sigmoid(x), 0.5)
                running_corrects += torch.sum(output_to_prediction == target.type(torch.ByteTensor))      

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

def test_model_aux(model, test_input, test_target, mini_batch_size=1000):
    model.eval()
   
    # Number of wrong predictions (first digit less than or equal to the second)
    _, _, test_output = model(test_input)
    output_to_prediction = torch.ge(torch.sigmoid(test_output), 0.5).flatten()
    print("output_to_pred = ", output_to_prediction[:10])
    print("target = ", test_target[:10])
    nb_errors_pairs = torch.sum(output_to_prediction != test_target.type(torch.ByteTensor)).item()
    print("nb_errors = ", nb_errors_pairs)
    acc_pairs = 1 - nb_errors_pairs / len(test_input)
    print("acc_pairs = ", acc_pairs)
    return acc_pairs

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
           round(test_results[2], 4), round(test_results[3], 4)]
    
    try: file = open(filename, 'r')
    except FileNotFoundError:
        csvData = [['Model', 'Number of parameters', 'Training time', 
                    'Mean accuracy (test set)', 'Std accuracy']]
        with open('auxiliary_losses.csv', 'w') as csvFile:
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


def multiple_training_runs(model, nb_runs, optimizer, train_input, train_target, train_classes, test_input, test_target, test_classes, mini_batch_size=1000, nb_epochs=300):
    list_time = []
    list_acc_pairs = []
    
    initial_model_wts = copy.deepcopy(model.state_dict())
            
    for i in range(nb_runs):
        model.load_state_dict(initial_model_wts)
        model, time_elapsed = train_model_aux(model, train_input, train_target, train_classes, optimizer, mini_batch_size=mini_batch_size, nb_epochs=nb_epochs)
        list_time.append(time_elapsed)
        
        acc_pairs = test_model_aux(model, test_input, test_target)
        list_acc_pairs.append(acc_pairs)
        
    mean_time, std_time = compute_properties(list_time)
    mean_acc_pairs, std_acc_pairs = compute_properties(list_acc_pairs)
    
    return mean_time, std_time, mean_acc_pairs, std_acc_pairs
