a deep learning library written from scratch in c. 

some usage examples from `minst.c`: 
```c
Network *net = network_create();
network_add_layer(net, layer_create(LINEAR(784, 512)));
network_add_layer(net, layer_create(RELU()));
Optimizer *opt = optimizer_create(net, ADAM(0.005f, 0.9f, 0.999f, 1e-8f));
network_train(net, opt, train_images, train_labels, 3, 64, LOSS_CROSS_ENTROPY, 1);
float accuracy = network_accuracy(predictions, test_labels);
```
