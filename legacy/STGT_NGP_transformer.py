import tensorflow as tf
import numpy as np
import numpy
import pandas as pd
import os
from tensorflow import keras
from tensorflow.keras import layers
from scipy.sparse.linalg import eigs
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--L", type=int, help="Number of transformer blocks", default=3)
parser.add_argument("--H", type=int, help="Number of attention heads per block", default=3)
parser.add_argument("--dp", type=str, help="Path to average speed data", default='None')
parser.add_argument("--amp", type=str, help="Path to adjacency matrix", default='None')
parser.add_argument("--d", type=str, help="PeMS-Bay or PeMSD7(M)", default='PeMS-Bay')

args = parser.parse_args()

# Choose Parameters
num_blocks = args.L
num_ah = args.H
data_path_data = args.dp
data_path_am = args.amp
data_which = args.d


# Input the prediction Interval and look-back window and output the dataset
class DataLoader:
    def __init__(self, data_path, look_back_window, prediction_interval, data_which):
        base_dd = data_path
        self.lb = int(look_back_window)
        self.pi = int(prediction_interval)

        if data_which == "PeMS-Bay":
            df = pd.read_hdf(base_dd)
        else:
            df = pd.read_csv(base_dd)
        self.data = df.values

    def make_dataset(self):
        # Convert the data_list element into dataset using prediction interval and look-back window info
        total_samples = self.data.shape[0] - self.lb - self.pi + 1
        sample_dim = self.data.shape[1]

        # Create the dataset
        dataset_x = np.zeros((total_samples, self.lb, sample_dim))
        dataset_y = np.zeros((total_samples, self.pi, sample_dim))

        for j in range(total_samples):
            dataset_x[j, :, :] = self.data[j:(j + self.lb), :]
            dataset_y[j, :, :] = self.data[(j + self.lb):(j + self.lb + self.pi), :]

        return dataset_x, dataset_y

    def generate_dataset(self):
        return self.make_dataset()



# Multi-head Graph Attention with Q, K, V
class SpatialConvLayer(tf.keras.layers.Layer):
    def __init__(self, Ks, c_in, c_out, kernel, **kwargs):
        super(SpatialConvLayer, self).__init__(**kwargs)
        self.Ks = Ks
        self.c_in = c_in
        self.c_out = c_out
        self.kernel = kernel

        w_input_init = tf.random_normal_initializer()
        self.w_input = tf.Variable(initial_value=w_input_init(shape=(1, 1, c_in, c_out), dtype="float32"),
                                   trainable=True)

        ws_init = tf.random_normal_initializer()
        self.ws = tf.Variable(initial_value=ws_init(shape=(Ks * c_in, c_out), dtype="float32"), trainable=True)

        bs_init = tf.zeros_initializer()
        self.bs = tf.Variable(initial_value=bs_init(shape=(c_out,), dtype="float32"), trainable=True)

    # def build(self, input_shape):

    def call(self, inputs):
        _, T, n, _ = inputs.get_shape().as_list()
        if self.c_in > self.c_out:
            # bottleneck down-sampling
            x_input = tf.nn.conv2d(inputs, self.w_input, strides=[1, 1, 1, 1], padding='SAME')
        elif self.c_in < self.c_out:
            # if the size of input channel is less than the output,
            # padding x to the same size of output channel.
            # Note, _.get_shape() cannot convert a partially known TensorShape to a Tensor.
            x_input = tf.concat([inputs, tf.zeros([tf.shape(inputs)[0], T, n, self.c_out - self.c_in])], axis=3)
        else:
            x_input = inputs

        # x_gconv = self.gconv(tf.reshape(inputs, [-1, n, self.c_in]), self.ws) + self.bs

        n = tf.shape(self.kernel)[0]
        x_tmp = tf.reshape(tf.transpose(tf.reshape(inputs, [-1, n, self.c_in]), [0, 2, 1]), [-1, n])
        x_mul = tf.reshape(tf.matmul(x_tmp, self.kernel), [-1, self.c_in, self.Ks, n])
        x_ker = tf.reshape(tf.transpose(x_mul, [0, 3, 1, 2]), [-1, self.c_in * self.Ks])
        x_gconv = tf.reshape(tf.matmul(x_ker, self.ws), [-1, n, self.c_out]) + self.bs

        x_gc = tf.reshape(x_gconv, [-1, T, n, self.c_out])
        return tf.nn.relu(x_gc[:, :, :, 0:self.c_out] + x_input)


def scaled_laplacian(W):
    '''
    Normalized graph Laplacian function.
    :param W: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
    :return: np.matrix, [n_route, n_route].
    '''
    # d ->  diagonal degree matrix
    n, d = np.shape(W)[0], np.sum(W, axis=1)
    # L -> graph Laplacian
    L = -W
    L[np.diag_indices_from(L)] = d
    for i in range(n):
        for j in range(n):
            if (d[i] > 0) and (d[j] > 0):
                L[i, j] = L[i, j] / np.sqrt(d[i] * d[j])
    # lambda_max \approx 2.0, the largest eigenvalues of L.
    lambda_max = eigs(L, k=1, which='LR')[0][0].real
    return np.mat(2 * L / lambda_max - np.identity(n))


def cheb_poly_approx(L, Ks, n):
    '''
    Chebyshev polynomials approximation function.
    :param L: np.matrix, [n_route, n_route], graph Laplacian.
    :param Ks: int, kernel size of spatial convolution.
    :param n: int, number of routes / size of graph.
    :return: np.ndarray, [n_route, Ks*n_route].
    '''
    L0, L1 = np.mat(np.identity(n)), np.mat(np.copy(L))

    if Ks > 1:
        L_list = [np.copy(L0), np.copy(L1)]
        for i in range(Ks - 2):
            Ln = np.mat(2 * L * L1 - L0)
            L_list.append(np.copy(Ln))
            L0, L1 = np.matrix(np.copy(L1)), np.matrix(np.copy(Ln))
        # L_lsit [Ks, n*n], Lk [n, Ks*n]
        return np.concatenate(L_list, axis=-1)
    elif Ks == 1:
        return np.asarray(L0)
    else:
        raise ValueError(f'ERROR: the size of spatial kernel must be greater than 1, but received "{Ks}".')


def first_approx(W, n):
    '''
    1st-order approximation function.
    :param W: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
    :param n: int, number of routes / size of graph.
    :return: np.ndarray, [n_route, n_route].
    '''
    A = W + np.identity(n)
    d = np.sum(A, axis=1)
    sinvD = np.sqrt(np.mat(np.diag(d)).I)
    # refer to Eq.5
    return np.mat(np.identity(n) + sinvD * A * sinvD)


def weight_matrix(file_path, sigma2=0.1, epsilon=0.5, scaling=True):
    '''
    Load weight matrix function.
    :param file_path: str, the path of saved weight matrix file.
    :param sigma2: float, scalar of matrix W.
    :param epsilon: float, thresholds to control the sparsity of matrix W.
    :param scaling: bool, whether applies numerical scaling on W.
    :return: np.ndarray, [n_route, n_route].
    '''
    try:
        W = pd.read_csv(file_path, header=None).values
        # W = W.astype('float64')
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')

    # check whether W is a 0/1 matrix.
    if set(np.unique(W)) == {0, 1}:
        print('The input graph is a 0/1 matrix; set "scaling" to False.')
        scaling = False

    if scaling:
        n = W.shape[0]
        W = W / 10000.
        W2, W_mask = W * W, np.ones([n, n]) - np.identity(n)
        # refer to Eq.10
        return np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask
    else:
        return W


# Multi-head attention with Q, K, V
class multiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, key_dim, num_heads):
        super(multiHeadAttention, self).__init__()
        self.key_dim = key_dim
        self.num_heads = num_heads

    def build(self, input_shape):
        self.WqL = []
        self.WkL = []
        self.WvL = []

        for i in range(self.num_heads):
            Wq_init = tf.random_normal_initializer()
            Wq = tf.Variable(initial_value=Wq_init(shape=(int(input_shape[-1]), self.key_dim), dtype="float32"),
                             trainable=True)
            self.WqL.append(Wq)

            Wk_init = tf.random_normal_initializer()
            Wk = tf.Variable(initial_value=Wk_init(shape=(int(input_shape[-1]), self.key_dim), dtype="float32"),
                             trainable=True)
            self.WkL.append(Wk)

            Wv_init = tf.random_normal_initializer()
            Wv = tf.Variable(initial_value=Wv_init(shape=(int(input_shape[-1]), int(input_shape[-1])), dtype="float32"),
                             trainable=True)
            self.WvL.append(Wv)

        Wlt_init = init = tf.random_normal_initializer()
        self.Wlt = tf.Variable(
            initial_value=Wlt_init(shape=((self.num_heads * int(input_shape[-1])), int(input_shape[-1])),
                                   dtype="float32"), trainable=True)

    def call(self, inputs):

        # inputs : batch_size x time_steps x dim
        x = inputs

        # transform for generating Q,K,V : (batch_size * time_steps) x dim
        x_tran = tf.reshape(x, [-1])
        x_tran = tf.reshape(x_tran, [-1, int(inputs.shape.as_list()[-1])])

        a_xL = []

        # Generate Query, Key and Value corresponding to each attention head
        for i in range(self.num_heads):
            # Query : batch_size x time_steps x dq
            xq = tf.matmul(x_tran, self.WqL[i])
            xq = tf.reshape(xq, [-1, int(inputs.shape.as_list()[-2]), int(xq.shape.as_list()[-1])])

            # Key : batch_size x time_steps x dk
            xk = tf.matmul(x_tran, self.WkL[i])
            xk = tf.reshape(xk, [-1, int(inputs.shape.as_list()[-2]), int(xk.shape.as_list()[-1])])

            # Value : batch_size x time_steps x dv
            xv = tf.matmul(x_tran, self.WvL[i])
            xv = tf.reshape(xv, [-1, int(inputs.shape.as_list()[-2]), int(xv.shape.as_list()[-1])])

            # Transposing each key in a batch (xk_t : batch_size x dk x time_steps)
            xk_t = tf.transpose(xk, perm=[0, 2, 1])

            # Computing scaled dot product self attention of each time step in each training sample (s_a : batch_size x time_steps x time_steps)
            s_a = tf.math.multiply(tf.keras.layers.Dot(axes=(1, 2))([xk_t, xq]), (1 / self.key_dim))

            # Applying Softmax Layer to the self attention weights for proper scaling (sft_s_a : batch_size x time_steps x time_steps)
            sft_s_a = tf.keras.layers.Softmax(axis=2)(s_a)

            # Computing attention augmented values for each time step and each training sample (a_x : batch_size x time_steps x dim)
            a_xL.append(tf.keras.layers.Dot(axes=(1, 2))([xv, sft_s_a]))

        # Concatenate and applying linear transform for making dimensions compatible
        a_x = tf.concat(a_xL, -1)

        # Transform to shape a_x_tran : ((batch_size x time_steps) x (dim x num_heads))
        a_x_tran = tf.reshape(a_x, [-1])
        a_x_tran = tf.reshape(a_x_tran, [-1, (self.num_heads * int(inputs.shape.as_list()[-1]))])

        # Get the dimensions compatible after applying linear transform
        a_x_tran = tf.matmul(a_x_tran, self.Wlt)
        a_x_tran = tf.reshape(a_x_tran, [-1, int(inputs.shape.as_list()[-2]), int(inputs.shape.as_list()[-1])])

        return a_x_tran


# Load Dataset
dl = DataLoader(data_path_data, 12, 9, data_which)
data = dl.generate_dataset()
X = data[0]
Y = data[1]

X = np.expand_dims(X, axis=3)

# Randomly re-shuffling the data and ground truths
np.random.seed(2)
reorder = np.random.permutation(X.shape[0])
X = X[reorder, :, :, :]
Y = Y[reorder, :, :]

# Dividing data into training, validation and test sets
net_data_size = int(X.shape[0])
train_size = int(np.ceil(0.7 * net_data_size))
val_size = int(np.ceil(0.15 * net_data_size))
test_size = net_data_size - train_size - val_size

X_tr = X[0:train_size, :, :, :]
Y_tr = Y[0:train_size, 0, :]
X_val = X[train_size:(train_size + val_size), :, :, :]
Y_val = Y[train_size:(train_size + val_size), 0, :]
X_test = X[(train_size + val_size):, :, :, :]
Y_test = Y[(train_size + val_size):, :, :]

# Perform data normalization (use train set mean and variance)
X_mean = np.mean(X_tr, axis=0)
X_var = np.sqrt(np.sum(np.square(np.subtract(X_tr, X_mean)), axis=0) / X_tr.shape[0])
X_tr = np.divide(np.subtract(X_tr, X_mean), X_var)
X_var = np.where(X_var > 0, X_var, 1)
X_val = np.divide(np.subtract(X_val, X_mean), X_var)
X_test = np.divide(np.subtract(X_test, X_mean), X_var)

X_tr = np.squeeze(X_tr)
X_val = np.squeeze(X_val)
X_test = np.squeeze(X_test)


# Transformer Block implemented as a Layer
# Only Graph PST
class TransformerBlockF(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlockF, self).__init__()
        # self.att = MultiHeadGraphAttention(embed_dim, num_heads, adj_mat_path_d2)
        self.att = multiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        # Calculate graph kernel
        W = weight_matrix(data_path_am)
        Ks = 3  # Ks: int, kernel size of spatial convolution.
        L = scaled_laplacian(W)
        Lk = cheb_poly_approx(L, Ks, n)
        graph_kernel = tf.cast(tf.constant(Lk), tf.float32)

        self.scl = SpatialConvLayer(Ks, 1, 1, graph_kernel)
        self.el = layers.Dense(embed_dim)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = tf.expand_dims(out1, axis=3)
        ffn_output = self.scl(ffn_output)
        ffn_output = tf.squeeze(ffn_output, [3])

        ffn_output = self.el(ffn_output)

        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class PositionEmbeddingLayer(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super(PositionEmbeddingLayer, self).__init__(**kwargs)
        self.position_embedding_layer = layers.Embedding(
            input_dim=(sequence_length), output_dim=output_dim
        )
        self.sequence_length = sequence_length

    def call(self, inputs):
        position_indices = tf.range(self.sequence_length)  # tf.range(1, self.sequence_length + 1, 1)
        embedded_words = inputs
        embedded_indices = self.position_embedding_layer(position_indices)
        return embedded_words + embedded_indices


def masked_mape_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)


# Creating the model (position embedding layer)
num_road_seg = X_tr.shape[2]

n = num_road_seg

num_attn_heads = num_ah
hidden_layer_dim = 64  # Hidden layer size in feed forward network inside transformer

# Initializing the transformer blocks
num_transformer_blocks = num_blocks
transformer_blocks = []

for i in range(num_transformer_blocks):
    transformer_blocks.append(TransformerBlockF(num_road_seg, num_attn_heads, hidden_layer_dim))

# Model
inputs = layers.Input(shape=(X_tr.shape[1], X_tr.shape[2],))
x = inputs

# Trainable Embedding
embedding_layer = PositionEmbeddingLayer(12, num_road_seg)
x = embedding_layer(x)

for i in range(num_transformer_blocks):
    x = transformer_blocks[i](x)

x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(X_tr.shape[2])(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()


# Learning Rate Scheduler
def scheduler(epoch, lr):
    exp = np.floor((1 + epoch) / 100)
    alpha = 0.0018 * (0.75 ** exp)
    return float(alpha)


# setup callbacks
callbacks = [tf.keras.callbacks.LearningRateScheduler(scheduler)]

model.compile(optimizer="RMSprop", loss="mse", metrics=["mse", "mae"])
history = model.fit(
    X_tr, Y_tr, batch_size=64, epochs=1500, validation_data=(X_val, Y_val), callbacks=callbacks, verbose=2
)

# Unroll before calculating testing MAPE (for pred intervals [1, 2, 3, 6, 9])
# Predict upto 45mins
mape_list = []
mae_list = []
rmse_list = []
pred_int_list = [int(0), int(1), int(2), int(5), int(8)]

X_test = np.expand_dims(X_test, axis=3)
X_test_copy = X_test  # Keep copy to restore original value after this run
for un_l in range(9):
    # For every testing sample we must predict the output at each time step and compute mape for pred_int_list steps

    # Make predictions for the next step
    y_pred_nxt = model.predict(np.squeeze(X_test), verbose=0)

    # Compute mape for predicton intervals in pred_int_list
    if un_l in pred_int_list:
        ground_truth = Y_test[:, un_l, :]
        test_mape = 100 * masked_mape_np(y_pred_nxt, ground_truth,
                                         null_val=0)  # np.mean(abs(y_pred_nxt - ground_truth) / (abs(ground_truth)+0.01))
        test_mae = np.mean(abs(y_pred_nxt - ground_truth))
        test_se = np.sqrt(np.mean(np.square(y_pred_nxt - ground_truth)))
        mape_list.append(test_mape)
        mae_list.append(test_mae)
        rmse_list.append(test_se)

    # Update the testing data: delete the speeds of the first time step and insert predicted value at the end
    # Undo the normalization
    X_test = np.add(np.multiply(X_test, X_var), X_mean)

    # Update the test data
    X_test = np.concatenate((X_test[:, 1:, :, :], np.expand_dims(y_pred_nxt, axis=[1, 3])), axis=1)

    # Reapply normalization
    X_test = np.divide(np.subtract(X_test, X_mean), X_var)

# Display Testing Results
print('\n')
for pr_id in range(len(pred_int_list)):
    print('Test MAPE values (%) for prediction interval ' + str(5 * (pred_int_list[pr_id] + 1)) + 'min:')
    print(mape_list[pr_id])

    print('\n')

    print('Test MAE values for prediction interval ' + str(5 * (pred_int_list[pr_id] + 1)) + 'min:')
    print(mae_list[pr_id])

    print('\n')

    print('Test RMSE values for prediction interval ' + str(5 * (pred_int_list[pr_id] + 1)) + 'min:')
    print(rmse_list[pr_id])

    print('\n')