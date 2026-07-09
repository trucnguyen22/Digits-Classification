import mnist_loader
from network_oop import NeuralNetwork, Optimizer, Trainer

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

network = NeuralNetwork([784, 30, 10])
optimizer = Optimizer(learning_rate=3.0)
trainer = Trainer(network, optimizer)

trainer.train(training_data, epochs=30, mini_batch_size=10, test_data=test_data)