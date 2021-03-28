# NLP_nano_udacity
## Coding Exercises

### PyTorch
#### Lesson 1. Intro to PyTorch
In this lesson, we learn the basic syntax of tensor operations in PyTorch, such as `torch.randn`, `torch.exp`, `torch.mm`, `torch.from_numpy`, `torch.utils`, etc. Importantly, to build feed-forward network, we learn how to use module `nn` in PyTorch, to create a subclass of `nn.Module` and to use `nn.functional`, `nn.Sequential`. Then, we can define the loss. For example, we can call `nn.NLLLoss`. After this, the gradient can be calculated by running `.backward` method on a variable. The last piece to start training is taking PyTorch's `optim` package. For instance, we use `optim.SGD` for stochastic gradient descent. So the usual steps to train is 
```
optim.zero_grad()
model.forward(batch_x)
loss=criterion(output,labels)
loss.backward()
optimizer.step()
```

Besides, we learn inference/validations (`loss.item(), probabilities.topk(1, dim=1), with torch.no_grad(), model.eval(), model.train()`), saving and loading models (`models.densenet121, datasets.ImageFolder, torch.utils.data.DataLoader, transform.compose()`) through the eight exercises. 
The applications includes two Digit-MNIST, Fashion-MNIST, Cat-Dog-Classification. At the end, we will use the popular pretrained neural network and transfer it to the Cat-Dog-Classification problem. 
To run with the 8 exercises, cpu is enough for reasonable training and testing. To play with the Cat-Dog-Classification task, we need to download the dataset [here](https://www.kaggle.com/c/dogs-vs-cats).
More tips on using `CUDA` are taught at last.


#### Lesson 2. Embeddings \& Word2Vec
In this lesson, we learn about embeddings in neural networks by Implementing Word2Vec. It is worth noted that `torch.nn.Embedding(num_embeddings, embedding_dim)` is a sparse layer type. It takes `num_embeddings` and `embedding_dim`, which are respectively the number of rows, and the number of columns in the embedding lookup table.

Two architectures for implementing Word2Vec:
* CBOW (Continuous Bag-Of-Words)
* Skip-gram

We use TSNE to visualiza the result of the embeddings.

The **Negative_Sampling_Exercise.ipynb** aims to accelerate the training process. Since very few weights before softmax layer are updated in a meaninful way, we approximate the loss from softmax layer. And we do this by updating a small subset of all the weights at once. But then we update only a small number of incorrect or noise targets around 100 so as opposed to 60,000.

#### Lesson 3. Implementation of RNN \& LSTM
In this lesson, we learn how to use RNNs and LSTMs to process sequential data. A simple time series example is provided. And we further build a LSTM model to predict character-level sentence prediction. 