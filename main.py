import numpy as np
import scipy.special


class NetWork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learn_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learn_rate = learn_rate
        '''
        链接矩阵初始化，范围为（-0.5，0.5)
        self.wih = (np.random.rand(self.hidden_nodes, self.input_nodes)) - 0.5
        self.who = (np.random.rand(self.output_nodes, hidden_nodes)) - 0.5
        '''
        self.wih = np.random.normal(0.0, pow(self.hidden_nodes), (self.hidden_nodes, input_nodes))
        self.wih = np.random.normal(0.0, pow(self.output_nodes), (self.output_nodes, hidden_nodes))

        self.activation_function = lambda x: scipy.special.expit(x)  # sigmoid函数

    def train(self, input_list, targets_list):
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_error = targets - final_outputs
        self.who += self.learn_rate * np.dot((output_error * final_outputs * (1.0 - final_outputs)),
                                             np.transpose(hidden_inputs))
        self.wih += self.learn_rate * np.dot((output_error * hidden_outputs * (1.0 - hidden_outputs)),
                                             np.transpose(inputs))

    def query(self, input_list):
        # 接受网络的输入，返回神经网络的输出
        inputs = np.array(input_list, ndmin=2).T  # ndmin设置数组维数，T为转置
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs




if __name__ == 'main':
    pass
