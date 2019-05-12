from torch import FloatTensor, LongTensor, Tensor
import torch
import random 
import math
import time


#Setting the seed.
random.seed(7) 


# Data creation/handling ------------------------------------------------------

def generate_disc_data(n=1000):
    """
    Generates a dataset with a uniformly sampled data in the range [0,1] in two dimensions, with labels beeing 1 inside
    a circle with radius 1/sqrt(2*pi) and labels with 1 on the inside and 0 on the outside of the circle.
    
    Output:
    inputs  : nx2 dimension FloatTensor
    targets : nx1 dimension LongTensor with range [0,1] 
    """
    
    inputs = torch.rand(n,2)
    distance = torch.norm((inputs - torch.Tensor([[0.5, 0.5]])).abs(), 2, 1, True)
    targets = distance.mul(math.sqrt(2*math.pi)).sub(1).sign().sub(1).div(2).abs().long()
    return inputs, targets


def generate_linear_data(n=1000):
    """
    Generates an example dataset that can be seperated linearly
    
    Output:
    inputs  : nx2 dimension FloatTensor
    targets : nx1 dimension LongTensor with range [0,1] 
    """
    
    inputs = torch.rand(n,2)
    targets = torch.sum(inputs, dim=1).sub(0.9).sign().sub(1).div(2).abs().long().view(-1, 1)
    return inputs, targets



def split_data(inputs, targets, train_part, val_part, test_part):
    """
    Splits dataset into training, validation and test set
    
    Output:
    train-, validation- and test inputs  : (percentage * n)x2 dimension FloatTensor
    train-, validation- and test targets : (percentage * n)x1 dimension LongTensor
    """
    training_size = math.floor(inputs.size()[0] * train_part)
    train_data = inputs.narrow(0, 0, training_size)
    train_targets = targets.narrow(0, 0, training_size)

    val_size = math.floor(inputs.size()[0] * val_part)
    validation_data = inputs.narrow(0, training_size, val_size)
    validation_targets = targets.narrow(0, training_size, val_size)

    test_size = math.floor(inputs.size()[0] * test_part)
    test_data = inputs.narrow(0, training_size+val_size, test_size)
    test_targets = targets.narrow(0, training_size+val_size, test_size)
    
    
    
    return train_data, train_targets, validation_data, validation_targets, test_data, test_targets

    
def convert_labels(input, target):
    """
    Convertes targets to labels of -1 and 1.
    
    Output:
    one_hot_labels : nx2 dimension FloatTensor 
    """
    new_target = input.new(target.size(0), target.max() + 1).fill_(-1)
    new_target.scatter_(1, target.view(-1, 1), 1.0)
    return new_target


# Modules ------------------------------------------------------------------------

class Module (object) :
    """
    Base class for other neural network modules to inherit from
    """
    
    def __init__(self):
        self._author = 'YoussefJanjar'
    
    def forward ( self , * input ) :
        """ `forward` should get for input, and returns, a tensor or a avr_calle of tensors """
        raise NotImplementedError
        
    def backward ( self , * gradwrtoutput ) :
        """
        `backward` should get as input a tensor or a avr_calle of tensors containing the gradient of the loss 
        with respect to the module’s output, accumulate the gradient wrt the parameters, and return a 
        tensor or a avr_calle of tensors containing the gradient of the loss wrt the module’s input.
        """
        raise NotImplementedError
        
    def param ( self ) :
        """ 
        `param` should return a list of pairs, each composed of a parameter tensor, and a gradient tensor 
        of same size. This list should be empty for parameterless modules (e.g. activation functions). 
        """
        return []


class Linear(Module):
    """
    Fully connected layer.
    
    Outputs:
    forward  :  FloatTensor of size m (m: number of units)
    backward :  FloatTensor of size m (m: number of units)
    """
    def __init__(self, dimention, output_dim, epsilon=1):
        super().__init__()
       
        self.weight = Tensor(output_dim, dimention).normal_(mean=0, std=epsilon)
        self.bias = Tensor(output_dim).normal_(0, epsilon)
        self.x = 0
        self.dl_weights = Tensor(self.weight.size())
        self.dl_bias = Tensor(self.bias.size())
         
    def forward(self, input):
        self.x = input
        return self.weight.mv(self.x) + self.bias
    
    def backward(self, grd_of_output):
        self.dl_weights.add_(grd_of_output.view(-1,1).mm(self.x.view(1,-1)))
        self.dl_bias.add_(grd_of_output)
        return self.weight.t().mv(grd_of_output)
    
    def param (self):
        return [(self.weight, self.dl_weights), (self.bias, self.dl_bias)]
        

class Tanh(Module):
    """
    Activation module: Tanh 
    
    Outputs:
    forward  :  FloatTensor of size m (m: number of units)
    backward :  FloatTensor of size m (m: number of units)
    """
    
    def __init__(self):
        super().__init__()
        self.s = 0
        
    def forward(self, input):
        self.s = input
        
        # We run Tanh function elementwise on all of input.
        tanh_vector = []
        for x in input:
            tanh = (2/ (1 + math.exp(-2*x))) -1
            tanh_vector.append(tanh)
        tanh_vector = torch.FloatTensor(tanh_vector)
        return tanh_vector
    
    def backward(self, grd_of_output):
        return 4 * ((self.s.exp() + self.s.mul(-1).exp()).pow(-2)) * grd_of_output
    
    def param (self):
        return [(None, None)]

        
class ReLu(Module):
    """
    Activation module: ReLu
    
    Outputs:
    forward  :   FloatTensor of size m (m: number of units)
    backward :   FloatTensor of size m (m: number of units)
    """
    def __init__(self):
        super().__init__()
        self.s = 0
    #Applies the ReLu function to the input.    
    def forward(self, input):
        self.s = input
        relu = input.clamp(min=0)
        return relu
    
    def backward(self, grd_of_output):
        relu_input = self.s
        last = relu_input.sign().clamp(min=0)
        curr_grad = grd_of_output * last
        return curr_grad    

    def param (self):
        return [(None, None)]  
        
        
class SGD():
    """
    SGD optimizer
    """
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr
    
    def step(self):
        """Weight Update. """

        for module in self.params:
            for avr_cal in module:
                weight, grad = avr_cal

                #Sanity check.
                if (weight is None) or (grad is None):
                    continue
                else:
                    #We update the weight.
                    weight.add_(-self.lr * grad)
    
    def zero_grad(self):
        """Clears the gradients in all the modules parameters"""

        for module in self.params:
            for avr_cal in module:  
                weight, grad = avr_cal

                 #Sanity check.
                if (weight is None) or (grad is None):
                    continue
                else:
                    #We set the gradient to 0.
                    grad.zero_()
                


# Sequential -------------------------------------------------------------------------------    

class Sequential(Module):
    """
    Combinaition of multiple modules/layers sequentially.
    
    Outputs:
    parameters :  List object containing List objects with the parameters of the modules in the Sequential instance. 
    """

    def __init__(self, *args):

        super().__init__()
        self.modules = []
        args = list(args)[0]

        for ind, module in enumerate(args):
            self.add_module(str(ind), module)

    def add_module(self, ind, module):

        self.ind = module
        self.modules.append(self.ind)

        return module
    
    def forward(self, input):

        out = input

        for module in self.modules:
            out = module.forward(out)

        return out
    
    def backward(self, grd_of_output):

        reversed_modules = self.modules[::-1]
        out = grd_of_output

        for module in reversed_modules:
            out = module.backward(out)
    
    def param ( self ) :

        parameters = []

        for module in self.modules:
            parameters.append(module.param())

        return parameters

        
        
# Lossfunction -----------------------------------------------------------------------

def loss(pred,target):

    #MMSE Loss.
    
    return (pred - target.float()).pow(2).sum()

def derivative_loss(pred,target):

    #Derivative of MMSE Loss.
   
    return 2*(pred - target.float())




# Training and Testing of model ------------------------------------------------------

def train_model(train_data, train_targets, test_data, test_targets, model, learning_rate, epochs):
    """
    Trains the model and outputs the training and validation error.

    Output:
    model       :  Sequential object
    train error :  List object 
    test error  :  List object 
    """   
    # make train targets and test targets to 1-hot vector
    train_targets = convert_labels(train_data, train_targets)
    test_targets = convert_labels(test_data, test_targets)    
    
    
    # define optimizer
    sgd = SGD(model.param(), lr=learning_rate)
    
    # constants
    nb_sample = train_data.size(0)
    dimention = train_data.size(1)

    nb_classes = train_targets.size(1)
    
    
    test_error_values = []
    train_error_values = []


    start = time.time()

    for epoch in range(epochs):
        
        # Training -------------------------------------------------------------------------------
        training_loss = 0
        nb_train_errors = 0

        # iterate through samples and accumelate derivatives
        for n in range(0, nb_sample):
            # clear gradiants 1.(outside loop with samples = GD) 2.(inside loop with samples = SGD).
            sgd.zero_grad()
            
            ### In order to get nb_train_errors, check how many true_targetly classified.
            
            # Get index of true_target , by taking argmax.
            curr_TT = train_targets[n]
            train_targets_list = [curr_TT[0], curr_TT[1]]
            true_target = train_targets_list.index(max(train_targets_list))
            
            # We compute the output of our model
            output = model.forward(train_data[n])
            
            # Get the predicted output, by taking argmax.
            output_list = [output[0], output[1]]

            #We compute our prediction.
            prediction = output_list.index(max(output_list))
            
            # Check if predicted true_targetly if it's the case we increase the number of errors.
            if int(true_target) != int(prediction) : nb_train_errors += 1


            ### We compute the loss loss 
            training_loss = training_loss + loss(output, train_targets[n].float())
            d_loss = derivative_loss(output, train_targets[n].float())

            #We backprop that loss 
            model.backward(d_loss)

            ### Gradient step 1.(outside loop with samples = GD) 2.(inside loop with samples = SGD)
            #We update the weights.
            sgd.step()
        #We append the training accuracy for this epoch.
        train_acc = (100 * nb_train_errors) / train_data.size(0)   
        train_error_values.append(train_acc)


        # Validation --------------------------------------------------------------------------------
        nb_valid_errors = 0
        

        # Here we do the exact same thing as for the training but without calculating the loss or updating the weights.
        for n in range(0, test_data.size(0)):
            
            
            curr_TestT = test_targets[n]
            test_targets_list = [curr_TestT[0], curr_TestT[1]]
            true_target = test_targets_list.index(max(test_targets_list)) 
            
                  
            
            output = model.forward(test_data[n])
            output_list = [output[0], output[1]]
            prediction = output_list.index(max(output_list))
            if int(true_target) != int(prediction) : nb_valid_errors += 1
        

        training_accuracy = 100-((100 * nb_train_errors) / train_data.size(0))
        validation_accuracy =  100-((100 * nb_valid_errors) / test_data.size(0))

        # Here we print the performance values to keep track of how the training is going.
        if epoch%(epochs*0.03) == 0:
            print('We are at epoch : {:d};  Training loss: {:.02f};   Training accuracy: {:.02f}%;   Validation accuracy {:.02f}%.'
              .format(epoch,
                      training_loss,
                      training_accuracy,
                      validation_accuracy))
        test_error_values.append((100 * nb_valid_errors) / test_data.size(0))


    end = time.time()
    training_time = int(end-start)
    print("The training time was : {:3}".format(time.strftime('%H:%M:%S', time.gmtime(training_time))))

    return model, train_error_values, test_error_values


def test_model(model, test_data, test_targets):
    """
    Tests the performance of the model on the test_set.
    """   
    
    # Converts the test labels.
    test_targets = convert_labels(test_data, test_targets)    
    
    test_error_values = []
    
    nb_test_errors = 0

    for n in range(0, test_data.size(0)):


        ### In order to get nb_train_errors, check how many true_targetly classified
        curr_TestT = test_targets[n]
        test_targets_list = [curr_TestT[0], curr_TestT[1]]
        true_target = test_targets_list.index(max(test_targets_list)) # argmax

        ### Find which one is predicted of the two outputs, by taking argmax            
        output = model.forward(test_data[n])
        output_list = [output[0], output[1]]
        prediction = output_list.index(max(output_list))
        if int(true_target) != int(prediction) : nb_test_errors += 1

    test_accuracy = (100-((100 * nb_test_errors) / test_data.size(0)))
    print('Here is the accuracy of our model on the Testing_set: {:.02f}%'.format(test_accuracy))
    test_error_values.append((100 * nb_test_errors) / test_data.size(0))
    return


