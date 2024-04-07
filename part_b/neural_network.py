from utils import *
from torch.autograd import Variable
from metadata_load import *

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch
import time


# These are a list of switches to turn on/off optimization in part B.
DO_LOAD_METADATA = True
DO_CHANGE_NAN_HANDLING = True
DO_NEIGHBOR_HYPERPARAMETERS = True
DO_EARLY_RETURN = True

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
    if DO_LOAD_METADATA:
        student_metadata = load_csv_to_matrix(os.path.join("../data", "student_meta.csv"))
        tsfmd_meta = transform_metadata(student_metadata)
        train_matrix = np.hstack((train_matrix, tsfmd_meta))

    zero_train_matrix = train_matrix.copy()
    ############### Part B: changed NaN handling to value of 0.5 #### No significant improvement over test accuracy
    zero_train_matrix[np.isnan(train_matrix)] = 0.5 if DO_CHANGE_NAN_HANDLING else 0
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

        ############### Part B: early return #### Significant reduce the training time
        if DO_EARLY_RETURN and epochs_no_improve >= 8:
            print("Early stopping at epoch {}   Validation accuracy: {}".format(epoch, valid_acc))
            break  # Early stopping
        print("Epoch: {}".format(epoch), end="\r")

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

        if DO_EARLY_RETURN and epochs_no_improve >= 8:
            print("Early stopping at epoch {}   Validation accuracy: {}                   ".format(epoch, valid_acc))
            break  # Early stopping
        print("Epoch: {}".format(epoch), end="\r")

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
    
    ############### Part B: append metadata to the training matrix #### Significant reduction to the training accuracy

    part1_start_timestamp = time.time()
    if DO_NEIGHBOR_HYPERPARAMETERS: 
        num_epoch = 100
        lamb = None  # Regularization parameter, not used in this optimization

        best_overall_val_acc = 0
        validation_dict = dict()
        best_hyperparameters = {}
        best_train_losses = None
        best_valid_losses = None

        best_k = 40  # start with k and lr in the middle of the grid
        best_lr = 0.02
        while 2 <= best_k <= 500 and 0.001 <= best_lr <= 0.5:
            k_before_explore = best_k
            lr_before_explore = best_lr
            exploration_list = [(best_k, best_lr), (best_k // 2, best_lr), (best_k * 2, best_lr), (best_k, best_lr / 2), (best_k, best_lr * 2)]
            for curr_k, curr_lr in exploration_list:
                if (curr_k, curr_lr) not in validation_dict:
                    print(f"Exploring k={curr_k}, lr={curr_lr}")
                    # Instantiated models
                    model = AutoEncoder(num_question=zero_train_matrix.shape[1], k=curr_k)
                    # Training models
                    train_losses, valid_losses, best_epoch, best_val_acc = train_without_lamb(model, curr_lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
                    validation_dict[(curr_k, curr_lr)] = (train_losses, valid_losses, best_epoch, best_val_acc)
                else:
                    print(f"Retrieved k={curr_k}, lr={curr_lr}")
                    train_losses, valid_losses, best_epoch, best_val_acc = validation_dict[(curr_k, curr_lr)]
                # Find the maximum acc
                if best_val_acc > best_overall_val_acc:
                    best_overall_val_acc = best_val_acc

                    best_hyperparameters = {
                                'k': curr_k,
                                'learning_rate': curr_lr,
                                'lambda': lamb,
                                'best_epoch': best_epoch, 
                                'best_overall_train_acc': best_overall_val_acc
                            }
                    best_k = curr_k
                    best_lr = curr_lr
                    best_train_losses = train_losses
                    best_valid_losses = valid_losses
            # If k and lr didn't change over the process, break
            if best_k == k_before_explore and best_lr == lr_before_explore:
                break
            else:
                print(f"completed one round of exploration, moving to ({best_k}, {best_lr}).")
    else:
        k = [10, 50, 100, 200, 500]
        # If 8 consecutive accuracies are not as high as the previous one, then stop the gradient descent.
        # Set optimization hyperparameters.
        lr = [0.01, 0.03, 0.05]
        num_epoch = 100
        lamb = None

        best_overall_val_acc = 0
        best_hyperparameters = {}
        best_train_losses = None
        best_valid_losses = None
        # Loop for each k value
        for each_k in k:
            # Loop for each learning rate
            for each_lr in lr:
                    # Instantiated models
                model = AutoEncoder(num_question=zero_train_matrix.shape[1], k=each_k)
                        # Training models
                train_losses, valid_losses, best_epoch, best_val_acc = train_without_lamb(model, each_lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
                    # Find the maximum acc
                if best_val_acc > best_overall_val_acc:
                    best_overall_val_acc = best_val_acc

                    best_hyperparameters = {
                                'k': each_k,
                                'learning_rate': each_lr,
                                'lambda': lamb,
                                'best_epoch': best_epoch, 
                                'best_overall_train_acc': best_overall_val_acc
                            }
                    best_train_losses = train_losses
                    best_valid_losses = valid_losses
    
    part1_end_timestamp = time.time()

    # Report best hyperparameters found
    print("Best Hyperparameters found through Neighboring Optimization:")
    for key, value in best_hyperparameters.items():
        print(f"\t{key}: {value}")
    print(f"Time taken looking for k and lr: {part1_end_timestamp - part1_start_timestamp} seconds.")


    model = AutoEncoder(num_question=zero_train_matrix.shape[1], k=best_hyperparameters['k'])
    train_without_lamb(model, best_hyperparameters['learning_rate'], best_hyperparameters['lambda'], train_matrix, zero_train_matrix, valid_data, best_epoch)

    test_acc, _ = evaluate(model, zero_train_matrix, test_data)

    print(f"(b): The test accuracy is {test_acc}")
    
    plot_metrics(best_train_losses, best_valid_losses, best_epoch)
#--------------- Part (d)
    part2_start_timestamp = time.time()
    k = best_hyperparameters['k']
    # If 8 consecutive accuracies are not as high as the previous one, then stop the gradient descent.
    # Set optimization hyperparameters.
    lr = best_hyperparameters['learning_rate']
    num_epoch = best_hyperparameters['best_epoch']
    lamb = [0, 0.0001, 0.0003, 0.001, 0.003, 0.01]

    best_overall_val_acc = 0
    b_hyperparameters = {}

    for each_lamb in lamb:
        print(f"Exploring lamb: {each_lamb}")
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
    part2_end_timestamp = time.time()

    # Report the best hyperparameters
    print("report for part d")
    print("Best Hyperparameters:")
    for key, value in b_hyperparameters.items():
        print(f"\t{key}: {value}")
    print(f"Time taken looking for lambda: {part2_end_timestamp - part2_start_timestamp} seconds.")
    
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
