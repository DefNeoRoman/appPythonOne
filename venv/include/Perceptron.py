import  numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))



trainingInputs = np.array([[0, 0, 1],
                          [1, 1, 1],
                          [1, 0, 1],
                          [0, 1, 1]])


trainigOutputs = np.array([[0, 1, 1, 0]]).T

np.random.seed(1)


synapticWeights = 2 * np.random.random((3, 1)) - 1


print("Случайные инициализирующиеся веса: ")
print(synapticWeights)

#Метод обратного распространения

for i in range(20000):
    inputLayer = trainingInputs
    outputs = sigmoid(np.dot(inputLayer, synapticWeights))




print("результат:")
print(outputs)


