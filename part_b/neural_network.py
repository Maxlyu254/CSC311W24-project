from utils import *
from torch.autograd import Variable
from metadata_load import *

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    ############### Part B: append metadata to the training matrix #### Significant reduction to the training accuracy
    load_metadata = True
    if load_metadata:
        student_metadata = load_csv_to_matrix(os.path.join("../data", "student_meta.csv"))
        tsfmd_meta = transform_metadata(student_metadata)
        train_matrix = np.hstack((train_matrix, tsfmd_meta))

    zero_train_matrix = train_matrix.copy()
    ############### Part B: changed NaN handling to value of 0.5 #### No significant improvement over test accuracy
    change_nan_handling = True
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0.5 if change_nan_handling else 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                               #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        # out = inputs

        # Pass the input through the first layer and sigmoid activation function
        encoded = torch.sigmoid(self.g(inputs))
        
        # Pass the encoded form through the last layer and sigmoid activation function
        reconstructed = torch.sigmoid(self.h(encoded))
        return reconstructed
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################

        # return out
def train_without_lamb(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function. 
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]
    # Initialize some variables to find the best valid accuracy and epoch.
    best_val_acc = 0
    best_epoch = 0
    epochs_no_improve = 0
    train_losses = []
    valid_losses = []

    for epoch in range(0, num_epoch):
        # Training steps
        train_loss = 0.
        
        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.)
            loss.backward()

            train_loss += loss.item()
            optimizer.step()
        
        # Validation Steps
        valid_acc, valid_loss = evaluate(model, zero_train_data, valid_data) 

         # Update training and validation losses for plotting
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
            best_epoch = epoch
            epochs_no_improve = 0 # reset the counter if performance improved
        else:
            epochs_no_improve += 1  # increment counter if no improvement

        if epochs_no_improve >= 8:
                print("Early stopping at epoch {}: Validation accuracy has not improved in {} epochs.".format(epoch, 8))
                break  # Early stopping
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))

    # print("For this epoch, the best acc", best_val_acc)
    return train_losses, valid_losses, best_epoch, best_val_acc

def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function. 
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]
    # Initialize some variables to find the best valid accuracy and epoch.
    best_val_acc = 0
    best_epoch = 0
    epochs_no_improve = 0
    train_losses = []
    valid_losses = []

    for epoch in range(0, num_epoch):
        train_loss = 0.
        
        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.) + (lamb / 2) * model.get_weight_norm()
            loss.backward()

            train_loss += loss.item()
            optimizer.step()
        
        valid_acc, valid_loss = evaluate(model, zero_train_data, valid_data) 

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
            best_epoch = epoch
            epochs_no_improve = 0 # reset the counter if performance improved
        else:
            epochs_no_improve += 1  # increment counter if no improvement

        if epochs_no_improve >= 8:
                print("Early stopping at epoch {}: Validation accuracy has not improved in {} epochs.".format(epoch, 8))
                break  # Early stopping
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))

    # print("For this epoch, the best acc", best_val_acc)
    return train_losses, valid_losses, best_epoch, best_val_acc

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0
    valid_loss = 0.

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)

        output = model(inputs)
    
        loss = torch.sum((output[0][valid_data["question_id"][i]] - valid_data["is_correct"][i]) ** 2.)

        valid_loss += loss.item()
        
        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total), valid_loss 

import matplotlib.pyplot as plt

def plot_metrics(train_losses, valid_losses, best_epoch):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(valid_losses, label='Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Losses')
    plt.title('Validation Losses Over Epochs')
    plt.legend()

    plt.show()

def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.
#------ Part(b) and Part(c)
     # Set model hyperparameters.
    k_values = [10, 50, 100, 200, 500]  # Potential k values
    learning_rates = [0.01, 0.03, 0.05]  # Potential learning rates
    num_epoch = 100
    lamb = None  # Regularization parameter, not used in this optimization

    best_overall_val_acc = 0
    best_hyperparameters = {}
    best_train_losses = None
    best_valid_losses = None

    # Starting with a baseline k and learning rate
    baseline_k = 100  # Example baseline
    baseline_lr = 0.01  # Example baseline

    # Neighboring Hyperparameter Optimization for k
    for k in [baseline_k // 2, baseline_k, baseline_k * 2]:  # Exploring neighbors
        if k not in k_values: continue  # Skip if the new k value is not in the predefined list
        # Keeping the learning rate constant at baseline during k optimization
        model = AutoEncoder(num_question=zero_train_matrix.shape[1], k=k)
        train_losses, valid_losses, best_epoch, best_val_acc = train_without_lamb(model, baseline_lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
        if best_val_acc > best_overall_val_acc:
            best_overall_val_acc = best_val_acc
            best_hyperparameters = {'k': k, 'learning_rate': baseline_lr, 'lambda': lamb, 'best_epoch': best_epoch}
            best_train_losses = train_losses
            best_valid_losses = valid_losses

    # Using the best k found, now optimize learning rate
    best_k = best_hyperparameters['k']
    for lr in [baseline_lr / 2, baseline_lr, baseline_lr * 2]:  # Exploring neighbors
        if lr not in learning_rates: continue  # Skip if the new lr value is not in the predefined list
        model = AutoEncoder(num_question=zero_train_matrix.shape[1], k=best_k)
        train_losses, valid_losses, best_epoch, best_val_acc = train_without_lamb(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
        if best_val_acc > best_overall_val_acc:
            best_overall_val_acc = best_val_acc
            best_hyperparameters.update({'learning_rate': lr, 'best_epoch': best_epoch})  # Update only learning rate and epoch
            best_train_losses = train_losses
            best_valid_losses = valid_losses

    # Report best hyperparameters found
    print("Best Hyperparameters found through Neighboring Optimization:")
    for key, value in best_hyperparameters.items():
        print(f"{key}: {value}")


    model = AutoEncoder(num_question=zero_train_matrix.shape[1], k=best_hyperparameters['k'])
    train_without_lamb(model, best_hyperparameters['learning_rate'], best_hyperparameters['lambda'], train_matrix, zero_train_matrix, valid_data, best_epoch)

    test_acc, _ = evaluate(model, zero_train_matrix, test_data)

    print(f"(b): The test accuracy is {test_acc}")
    
    plot_metrics(best_train_losses, best_valid_losses, best_epoch)
#--------------- Part (d)
    k = best_hyperparameters['k']
    # If 8 consecutive accuracies are not as high as the previous one, then stop the gradient descent.
    # Set optimization hyperparameters.
    lr = best_hyperparameters['learning_rate']
    num_epoch = best_hyperparameters['best_epoch']
    lamb = [0.001, 0.01, 0.1, 1]

    best_overall_val_acc = 0
    b_hyperparameters = {}

    for each_lamb in lamb:
                # Instantiated models
        model = AutoEncoder(num_question=zero_train_matrix.shape[1], k=k)
                    # Training models
        train_losses, valid_losses, best_epoch, best_val_acc = train(model, lr, each_lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
                # Find the maximum acc
        if best_val_acc > best_overall_val_acc:
            best_overall_val_acc = best_val_acc

            b_hyperparameters = {
                            'k': k,
                            'learning_rate': lr,
                            'lambda': each_lamb,
                            'best_epoch': num_epoch, 
                            'best_overall_train_acc': best_overall_val_acc
                        }

    # Report the best hyperparameters
    print("report for part d")
    print("Best Hyperparameters:")
    for key, value in b_hyperparameters.items():
        print(f"{key}: {value}")
    
    best_num_epoch = b_hyperparameters['best_epoch']
    lamb = b_hyperparameters['lambda']
    test_k = b_hyperparameters['k']
    test_lr = b_hyperparameters['learning_rate']
    test_lamb = b_hyperparameters['lambda']

    model = AutoEncoder(num_question=zero_train_matrix.shape[1], k=test_k)
    train(model, test_lr, test_lamb, train_matrix, zero_train_matrix, valid_data, best_num_epoch)

    valid_acc, _ = evaluate(model, zero_train_matrix, valid_data)
    test_acc, _ = evaluate(model, zero_train_matrix, test_data)
    print(f"(d): The valid accuracy is {valid_acc}")
    print(f"(d): The test accuracy is {test_acc}")

    


    # train(model, lr, lamb, train_matrix, zero_train_matrix,
    #       valid_data, num_epoch)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
