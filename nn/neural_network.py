#!/usr/bin/env python3
from .matrix import Matrix

class NeuralNetwork():
    def __init__(self, arch):
        self.a = [] #len = len(arch)
        self.w = [] #len = len(arch) - 1
        self.b = [] #len = len(arch) - 1
        self.a.append(Matrix(1, arch[0]))
        for i in range(1, len(arch)):
            self.w.append(Matrix(self.a[i-1].cols, arch[i]))
            self.b.append(Matrix(1, arch[i]))
            self.a.append(Matrix(1, arch[i]))
        self.input = self.a[0].data

    def output(self, row, col):
        """Get output layer of the neural network"""
        return self.a[-1].data[row][col]

    def zero_network(self):
        for el in self.a:
            el.fill(0)
        for el in self.b:
            el.fill(0)
        for el in self.w:
            el.fill(0)

    def randomize(self, start=0, stop=1):
        """Randomize every layer of the neural network"""
        for weight in self.w:
            weight.randomize(start, stop)

        for bias in self.b:
            bias.randomize(start, stop)

        for activation in self.a:
            activation.randomize(start, stop)

    def forward(self):
        """forward data to the network"""
        for i in range(0, len(self.w)):
            self.a[i+1] = self.a[i] * self.w[i]
            self.a[i+1].sum(self.b[i])
            self.a[i+1].sigmoid()

    def finite_diff(self, gradient, eps, training_input, training_output):
        """calulate gradient with finite differense (care only for learning, use backprop for regular use)"""
        saved = 0.0
        c = self.cost(training_input, training_output)

        for i in range(0, len(self.w)):
            #weights
            for row in range(0, self.w[i].rows):
                for col in range(0, self.w[i].cols):
                    saved = self.w[i].data[row][col]
                    self.w[i].data[row][col] += eps
                    gradient.w[i].data[row][col] = (self.cost(training_input, training_output) - c) / eps
                    self.w[i].data[row][col] = saved
            #biases
            for row in range(0, self.b[i].rows):
                for col in range(0, self.b[i].cols):
                    saved = self.b[i].data[row][col]
                    self.b[i].data[row][col] += eps
                    gradient.b[i].data[row][col] = (self.cost(training_input, training_output) - c) / eps
                    self.b[i].data[row][col] = saved

    def back_propagation(self, gradient, training_input, training_output):
        """calculate gradient with back propagation"""
        assert training_input.rows == training_output.rows, "number of rows have to be equal in training data"
        # i - current sample
        # l - current layer
        # j - current activation
        # k - previous activation

        gradient.zero_network()


        for i in range(0, training_input.rows):
            self.a[0].copy(Matrix.from_list([training_input.data[i]]))
            self.forward()

            for y in range(0, len(self.a) - 1):
                gradient.a[y].fill(0)

            for j in range(0, training_output.cols):
                gradient.a[-1].data[0][j] = self.a[-1].data[0][j] - training_output.data[i][j]


            #iterate over activations
            for l in range(len(self.a) - 1, 0, -1):
                #print("l:%s"%(l))
                for j in range(0, self.a[l].cols):
                    a = self.a[l].data[0][j]
                    da = gradient.a[l].data[0][j]
                    gradient.b[l-1].data[0][j] += 2*da*a*(1 - a)
                    for k in range(0, self.a[l-1].cols):
                        pa = self.a[l-1].data[0][k]
                        weight = self.w[l-1].data[k][j]
                        gradient.w[l-1].data[k][j] += 2*da*a*(1-a)*pa
                        gradient.a[l-1].data[0][k] += 2*da*a*(1-a)*weight

        for i in range(0, len(gradient.a) - 1):
            for j in range(0, gradient.w[i].rows):
                for k in range(0, gradient.w[i].cols):
                    gradient.w[i].data[j][k] /= training_input.rows
            for j in range(0, gradient.b[i].rows):
                for k in range(0, gradient.b[i].cols):
                    gradient.b[i].data[j][k] /= training_input.rows

    def cost(self, training_input, training_output):
        """calculate cost of the model"""
        assert training_input.rows == training_output.rows, "rows must have equal amounts of rows"
        assert training_output.cols == self.a[-1].cols, "output and XOR output must have the same number of cols"

        c = 0.0
        for row in range(0, training_input.rows):
            x = training_input.data[row]
            y = training_output.data[row]

            self.a[0].copy(Matrix.from_list([x]))
            self.forward()

            for i in range(0, training_output.cols):
                d = self.a[-1].item_at(0, i) - y[i]
                c += d*d

        return c/training_input.rows

    def learn(self, gradient, learning_rate):
        """apply gradient to network"""
        for i in range(0, len(self.w)):
            #weights
            for j in range(0, self.w[i].rows):
                for k in range(0, self.w[i].cols):
                    self.w[i].data[j][k] -= (learning_rate * gradient.w[i].data[j][k])
            #biases
            for j in range(0, self.b[i].rows):
                for k in range(0, self.b[i].cols):
                    self.b[i].data[j][k] -= (learning_rate * gradient.b[i].data[j][k])

    def __str__(self):
        """return string represenation of the network"""
        return_string = "nn=[\n"
        for i in range(0, len(self.w)):
            return_string += "w[" + str(i) + "] = "
            return_string += self.w[i].__str__() + " "
            return_string += "\n]"
            return_string += "b[" + str(i) + "] = "
            return_string += self.b[i].__str__() + " "
            return_string += "\n]"
        for i in range(0, len(self.a)):
            return_string += "a[" + str(i) + "] = "
            return_string += self.a[i].__str__() + " "
            return_string += "\n]"

        return return_string
