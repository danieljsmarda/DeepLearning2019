import matplotlib.pyplot as plt

#### Function that draws the evolution of the loss during training.
def visualize_loss(model,loss_model,lr):
    plt.plot(loss_model[0],loss_model[1], color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Evolution of the loss during training with learning rate {:>3} of the model {:>5}.'.format(lr,model.__class__.__name__))
    plt.show()


#### Function that prints the results of the prediction one the model is trained.
def print_results(model,optimizer,learning_rate,NB_EPOCHS,MINI_BATCH_SIZE, train_input, train_target,test_input, test_target):
    train_err = model.test_model(model, train_input, train_target, MINI_BATCH_SIZE) / train_input.size(0) * 100 
    test_err =  model.test_model(model, test_input, test_target, MINI_BATCH_SIZE) / test_input.size(0) * 100  
    print('model: {:6}, optimizer: {:6}, learning rate: {:6}, num epochs: {:3}, '
                    'mini batch size: {:3}, train error: {:5.2f}%, test error: {:5.2f}%'.format(model.__class__.__name__,optimizer.__name__,learning_rate,NB_EPOCHS,MINI_BATCH_SIZE,train_err,test_err
                    )
                )
