from rnn_utils import *


def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    # Retrieve parameters from "parameters"
    Wf = parameters["Wf"]  # forget gate weight
    bf = parameters["bf"]
    Wi = parameters["Wi"]  # update gate weight (notice the variable name)
    bi = parameters["bi"]  # (notice the variable name)
    Wc = parameters["Wc"]  # candidate value weight
    bc = parameters["bc"]
    Wo = parameters["Wo"]  # output gate weight
    bo = parameters["bo"]
    Wy = parameters["Wy"]  # prediction weight
    by = parameters["by"]

    # Retrieve dimensions from shapes of xt and Wy
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    # Concatenate a_prev and xt
    concat = np.concatenate((a_prev, xt), axis=0)

    # Compute values for ft (forget gate), it (update gate),
    # cct (candidate value), c_next (cell state),
    # ot (output gate), a_next (hidden state)
    ft = sigmoid(np.dot(Wf, concat) + bf)        # forget gate
    it = sigmoid(np.dot(Wi, concat) + bi)        # update gate
    cct = np.tanh(np.dot(Wc, concat) + bc)       # candidate value
    c_next = ft * c_prev + it * cct    # cell state
    ot = sigmoid(np.dot(Wo, concat) + bo)        # output gate
    a_next = ot * np.tanh(c_next)    # hidden state

    # Compute prediction of the LSTM cell
    yt_pred = softmax(np.dot(Wy, a_next) + by)

    # store values needed for backward propagation in cache
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache


def lstm_forward(x, a0, parameters):
    # Initialize "caches", which will track the list of all the caches
    caches = []

    # saving parameters['Wy'] in a local variable
    Wy = parameters['Wy']
    # Retrieve dimensions from shapes of x and parameters['Wy']
    n_x, m, T_x = x.shape
    n_y, n_a = Wy.shape

    # initialize "a", "c" and "y" with zeros
    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))
    y = np.zeros((n_y, m, T_x))

    # Initialize a_next and c_next
    a_next = a0
    c_next = np.zeros((n_a, m))

    # loop over all time-steps
    for t in range(T_x):
        # Get the 2D slice 'xt' from the 3D input 'x' at time step 't'
        xt = x[:, :, t]
        # Update next hidden state, next memory state, compute the prediction, get the cache
        a_next, c_next, yt, cache = lstm_cell_forward(
            xt, a_next, c_next, parameters)
        # Save the value of the new "next" hidden state in a
        a[:, :, t] = a_next
        # Save the value of the next cell state
        c[:, :, t] = c_next
        # Save the value of the prediction in y
        y[:, :, t] = yt
        # Append the cache into caches
        caches.append(cache)

    # store values needed for backward propagation in cache
    caches = (caches, x)

    return a, y, c, caches
