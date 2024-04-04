from utils import *
from matplotlib import pyplot as plt
import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    for i in range(len(data["user_id"])):
        user = data["user_id"][i]
        question = data["question_id"][i]
        correct = data["is_correct"][i]
        log_lklihood += correct * (theta[user] - beta[question]) - np.log((1 + np.exp(theta[user] - beta[question])))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    theta_update = np.zeros(theta.shape)
    beta_update = np.zeros(beta.shape)

    for i in range(len(data['user_id'])):
        user = data['user_id'][i]
        question = data['question_id'][i]
        correct = data['is_correct'][i]
        theta_update[user] += (correct - sigmoid(theta[user] - beta[question]))
        beta_update[question] += (sigmoid(theta[user] - beta[question]) - correct)

    theta += lr * theta_update
    beta += lr * beta_update
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros(len(data["user_id"]))
    beta = np.zeros(len(data["question_id"]))

    val_acc_lst = []
    train = []
    validation = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        train.append(neg_lld)
        neg_lld_validation = neg_log_likelihood(data=val_data, theta=theta, beta=beta)
        validation.append(neg_lld_validation)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train, validation


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lrt_rate = 0.01
    iterations = 40
    theta, beta, val_acc_lst, train, validation = irt(train_data, val_data, lrt_rate, iterations)
    plt.plot(range(1, iterations + 1), train)
    plt.ylabel("Training Negative Log-Likelihood")
    plt.xlabel("Training Iterations")
    plt.title("Training Negative Log-Likelihood vs. Training Iterations")
    plt.savefig('ir_train')
    plt.show()

    plt.plot(range(1, iterations + 1), validation)
    plt.ylabel("Validation Negative Log-Likelihood")
    plt.xlabel("Validation Iterations")
    plt.title("Validation Negative Log-Likelihood vs. Validation Iterations")
    plt.savefig('ir_val')
    plt.show()

    print(f'Validation accuracy: {val_acc_lst[-1]}')
    print(f'Test accuracy: {evaluate(test_data, theta, beta)}')
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    j_1 = beta.argmax()
    j_2 = beta.argmin()
    j_3 = np.random.randint(1500)
    theta = np.sort(theta)
    plt.plot(theta, sigmoid((theta - beta[j_1])), color='red')
    plt.plot(theta, sigmoid((theta - beta[j_2])), color='blue')
    plt.plot(theta, sigmoid((theta - beta[j_3])), color='green')
    plt.savefig('ir_three')
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
