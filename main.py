#!/usr/bin/env python3
import nn

def main():
    eps = 1e-1
    learning_rate = 1

    training_data = nn.Matrix.from_list(nn.OR)

    input = training_data.sub_matrix(4, 2)
    output = training_data.sub_matrix(4, 1, 0, 2)

    arch = [2, 2, 1]
    network = nn.NeuralNetwork(arch)
    gradient = nn.NeuralNetwork(arch)

    network.randomize()

    print("cost(backprop): %s"%(network.cost(input, output)))
    for i in range(0, 5 * 1000):
        #network.finite_diff(gradient, eps, input, output)
        network.back_propagation(gradient, input, output)
        network.learn(gradient, learning_rate)

    print("cost(backprop): %s"%network.cost(input, output))

    for i in range(0, 2):
        for j in range(0, 2):
            network.input[0][0] = i
            network.input[0][1] = j
            network.forward()
            print("%s | %s = %s" % (i, j, network.output(0, 0)))

main()
