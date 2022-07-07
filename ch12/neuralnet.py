import numpy as np
import sys

class NeuralNetMLP(object):
    """ Feedforward neural network / Multi-layer perceptron classifier.

    Parameters
    ------------
    n_hidden : int (default: 30)
        Number of hidden units.
    l2 : float (default: 0.)
        Lambda value for L2-regularization.
        No regularization if l2=0. (default)
    epochs : int (default: 100)
        Number of passes over the training set.
    eta : float (default: 0.001)
        Learning rate.
    shuffle : bool (default: True)
        Shuffles training data every epoch if True to prevent circles.
    minibatch_size : int (default: 1)
        Number of training examples per minibatch.
    seed : int (default: None)
        Random seed for initializing weights and shuffling.

    Attributes
    -----------
    eval_ : dict
      Dictionary collecting the cost, training accuracy,
      and validation accuracy for each epoch during training.

    """
    def __init__(self, n_hidden=30,
                 l2=0., epochs=100, eta=0.001,
                 shuffle=True, minibatch_size=1, seed=None):

        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size

    def _onehot(self, y, n_classes):
        """Encode labels into one-hot representation

        Parameters
        ------------
        y : array, shape = [n_examples]
            Target values.
        n_classes : int
            Number of classes

        Returns
        -----------
        onehot : array, shape = (n_examples, n_labels)

        """
        onehot = np.zeros((n_classes, y.shape[0]))
        for idx, val in enumerate(y.astype(int)):
            onehot[val, idx] = 1.
        return onehot.T

    def _sigmoid(self, z):
        """Compute sigmoid function """
        # It maps input of -inf to +inf to output of 0 to 1
        # -inf -> 0
        # 0 -> 0.5
        # +inf -> 1
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def _forward(self, X):
        """Compute forward propagation step"""

        # Step 1: Compute Z_h
        # dim{X} = n_examples (100) x m (784)
        # dim{w_h} = m (784) x d (100)
        # dim{z_h} = n_examples (100) x d (100)      
        z_h = np.dot(X, self.w_h) + self.b_h # see page 391

        # Step 2: Compute a_h using the activation function which is the sigmoid function in this case
        a_h = self._sigmoid(z_h) # these are nonlinear "improved features"

        # Step 3: net input of output layer
        # [n_examples, n_hidden] dot [n_hidden, n_classlabels]
        # -> [n_examples, n_classlabels]

        # Step 3: Compute Z_out
        # dim{a_h} = n_examples (100) x d (100)
        # dim{w_out} = d (100) x t (10)   
        # dim{z_out} = n_examples (100) x t (10)       
        z_out = np.dot(a_h, self.w_out) + self.b_out

        # Step 4: Compute a_out using the activation function which is the sigmoid function in this case
        # a_out is the estimated output in one-hot format 
        a_out = self._sigmoid(z_out)

        return z_h, a_h, z_out, a_out

    def _compute_cost(self, y_enc, output):
        """Compute cost function.

        Parameters
        ----------
        y_enc : array, shape = (n_examples, n_labels)
            one-hot encoded class labels.
        output : array, shape = [n_examples, n_output_units]
            Activation of the output layer (forward propagation)

        Returns
        ---------
        cost : float
            Regularized cost

        """
        L2_term = (self.l2 *
                   (np.sum(self.w_h ** 2.) +
                    np.sum(self.w_out ** 2.)))

        term1 = -y_enc * (np.log(output))
        term2 = (1. - y_enc) * np.log(1. - output)
        cost = np.sum(term1 - term2) + L2_term
        
        # If you are applying this cost function to other
        # datasets where activation
        # values may become more extreme (closer to zero or 1)
        # you may encounter "ZeroDivisionError"s due to numerical
        # instabilities in Python & NumPy for the current implementation.
        # I.e., the code tries to evaluate log(0), which is undefined.
        # To address this issue, you could add a small constant to the
        # activation values that are passed to the log function.
        #
        # For example:
        #
        # term1 = -y_enc * (np.log(output + 1e-5))
        # term2 = (1. - y_enc) * np.log(1. - output + 1e-5)
        
        return cost

    def predict(self, X):
        """Predict class labels

        Parameters
        -----------
        X : array, shape = [n_examples, n_features]
            Input layer with original features.

        Returns:
        ----------
        y_pred : array, shape = [n_examples]
            Predicted class labels.

        """
        z_h, a_h, z_out, a_out = self._forward(X)
        y_pred = np.argmax(z_out, axis=1)
        return y_pred

    def fit(self, X_subtrain, y_subtrain, X_valid, y_valid):
        """ Learn weights from training data.

        Parameters
        -----------
        X_subtrain : array, shape = [n_examples, n_features]
            Input layer with original features.
        y_subtrain : array, shape = [n_examples]
            Target class labels.
        X_valid : array, shape = [n_examples, n_features]
            Sample features for validation during training
        y_valid : array, shape = [n_examples]
            Sample labels for validation during training

        Returns:
        ----------
        self

        """

        # Initialization
        n_output = np.unique(y_subtrain).shape[0]  # number of class labels (10: 0~9)
        n_features = X_subtrain.shape[1] # number of features (784 pixels)
        epoch_strlen = len(str(self.epochs))  # number of characters for epochs (used for displaying the progress)
        self.eval_ = {'cost': [], 'subtrain_acc': [], 'valid_acc': []}

        # Initialize weights for input -> hidden
        self.b_h = np.zeros(self.n_hidden) # this is w(h)_0,1..w(h)_0,d in the figure on page 388. dim{b_h} = d = 100
        self.w_h = self.random.normal(loc=0.0, scale=0.1,
                                      size=(n_features, self.n_hidden)) # dim{w_h} = 784 x 100 = <number of features> x d

        # Initialize weights for hidden -> output
        self.b_out = np.zeros(n_output) # this is w(out)_0,1..w(out)_0,t in the figure on page 388. dim{b_out} = t = 10
        self.w_out = self.random.normal(loc=0.0, scale=0.1,
                                        size=(self.n_hidden, n_output)) # dim{w_out} = 100 x 10 = d x t

        # Encode output into one-hot format (a group of bits among which the legal combinations of values are only those with a single high (1) bit and all the others low (0))
        # y_subtrain has a format of 0 ~ 9
        # y_subtrain_enc has a format of [1000000000], [0100000000], [0010000000], etc
        y_subtrain_enc = self._onehot(y_subtrain, n_output)

        # Iterate over epochs
        for i in range(self.epochs):
            # Prepare indices for the subtraining samples for this epoch
            indices = np.arange(X_subtrain.shape[0])
            if self.shuffle:
                self.random.shuffle(indices)

            # Iterate over minibatches throughout all samples in the subtraining data set
            for start_idx in range(0, indices.shape[0] - self.minibatch_size +
                                   1, self.minibatch_size):
                batch_idx = indices[start_idx:start_idx + self.minibatch_size] # indices for the subtraining samples for this batch

                # Perform forward propagation
                z_h, a_h, z_out, a_out = self._forward(X_subtrain[batch_idx])

                ##################
                # Backpropagation (p418)
                ##################

                # Compute the error (predicted - actual) for the output layer
                # dim{delta_out} = n_examples (100) x t (10)     
                delta_out = a_out - y_subtrain_enc[batch_idx] # the error vector (predicted - actual)

                # Compute the derivative of the sigmoid activation function (needed in later steps)
                # dim{a_h} = n_examples (100) x d (100)     
                # dim{sigmoid_derivative_h} = n_examples (100) x d (100)
                sigmoid_derivative_h = a_h * (1. - a_h) # page 418 and 419 for equations.  "*" is element-wise multiplication

                # Compute the error (predicted - actual) for the hidden layer
                # dim{delta_h} = n_examples (100) x d (100)   
                delta_h = (np.dot(delta_out, self.w_out.T) * sigmoid_derivative_h) # equation is in page 418, but not sure where it came from...

                # Compute the gradient for hidden layer weights
                # dim{grad_w_h} = m (784) x d (100)
                # dim{grad_b_h} = d (100)
                grad_w_h = np.dot(X_subtrain[batch_idx].T, delta_h) # equation is in page 420, but not sure where it came from...
                grad_b_h = np.sum(delta_h, axis=0)

                # Compute the gradient for output layer weights
                # dim{grad_w_out} = d (100) x t (10)
                # dim{grad_b_out} = t (10)
                grad_w_out = np.dot(a_h.T, delta_out) # equation is in page 420, but not sure where it came from...
                grad_b_out = np.sum(delta_out, axis=0)

                # Regularization and weight updates
                delta_w_h = (grad_w_h + self.l2*self.w_h)
                delta_b_h = grad_b_h # bias is not regularized
                self.w_h -= self.eta * delta_w_h
                self.b_h -= self.eta * delta_b_h

                delta_w_out = (grad_w_out + self.l2*self.w_out)
                delta_b_out = grad_b_out  # bias is not regularized
                self.w_out -= self.eta * delta_w_out
                self.b_out -= self.eta * delta_b_out

            #############
            # Evaluation
            #############

            # Evaluation after each epoch during training
            # This section is not necessary
            z_h, a_h, z_out, a_out = self._forward(X_subtrain)
            
            cost = self._compute_cost(y_enc=y_subtrain_enc,
                                      output=a_out)

            y_subtrain_pred = self.predict(X_subtrain)
            y_valid_pred = self.predict(X_valid)

            subtrain_acc = ((np.sum(y_subtrain == y_subtrain_pred)).astype(np.float) /
                         X_subtrain.shape[0])
            valid_acc = ((np.sum(y_valid == y_valid_pred)).astype(np.float) /
                         X_valid.shape[0])

            sys.stderr.write('\r%0*d/%d | Cost: %.2f '
                             '| Subtrain/Valid Acc.: %.2f%%/%.2f%% ' %
                             (epoch_strlen, i+1, self.epochs, cost,
                              subtrain_acc*100, valid_acc*100))
            sys.stderr.flush()

            self.eval_['cost'].append(cost)
            self.eval_['subtrain_acc'].append(subtrain_acc)
            self.eval_['valid_acc'].append(valid_acc)

        return self