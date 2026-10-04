# Three-Layer Neural Network on MNIST

## 1. Objective

The objectives of this assignment are:

1. To derive the backpropagation equation for the weight matrix \(W_3\).
2. To train a three-linear-layer neural network on the MNIST dataset and report its classification accuracy.

## 2. Backpropagation for \(W_3\)

For the third linear layer,

\[
z_3=W_3h_2+b_3
\]

where \(h_2\) is the output of the previous layer.

Using the chain rule,

\[
\frac{\partial L}{\partial W_3}
=
\frac{\partial L}{\partial z_3}
\frac{\partial z_3}{\partial W_3}.
\]

Since

\[
\frac{\partial z_3}{\partial W_3}=h_2^T,
\]

we obtain

\[
\boxed{
\frac{\partial L}{\partial W_3}
=
\frac{\partial L}{\partial z_3}h_2^T
}
\]

For Softmax with Cross-Entropy Loss,

\[
\frac{\partial L}{\partial z_3}=p-y,
\]

therefore,

\[
\boxed{
\frac{\partial L}{\partial W_3}
=
(p-y)h_2^T
}
\]

In PyTorch, this gradient is automatically calculated by `loss.backward()`.

## 3. Neural Network and Training

The MNIST dataset contains \(28\times28\) grayscale handwritten digit images with 10 classes, from 0 to 9. Each image is flattened into a 784-dimensional vector.

The three-linear-layer network is

\[
784 \rightarrow 196 \rightarrow 49 \rightarrow 10.
\]

ReLU activation is applied after the first and second linear layers. The final layer outputs 10 logits corresponding to the 10 digit classes.

The training settings are:

- Loss function: Cross-Entropy Loss
- Optimizer: SGD
- Learning rate: 0.01
- Momentum: 0.9
- Batch size: 64
- Number of epochs: 10

## 4. Result

The training loss decreased continuously during training:

\[
0.537 \rightarrow 0.192 \rightarrow \cdots \rightarrow 0.032.
\]

After 10 epochs, the model achieved:

\[
\boxed{\text{Test Accuracy}=97.86\%}
\]

The decreasing training loss shows that the network learned effectively from the MNIST training data.

## 5. Conclusion

A three-linear-layer neural network was successfully trained on the MNIST dataset. After 10 epochs, the model achieved a test accuracy of **97.86%**. The experiment also demonstrates how backpropagation computes the gradient of \(W_3\) and how PyTorch performs this process automatically during training.
