import numpy as np
import matplotlib.pyplot
import scipy.special

image_num = 10
label_num = 10


def load_train_images():
    image_arrays = []
    with open('train-images.idx3-ubyte', 'rb') as f:
        file = f.read()
        # image_num = int(file[4:8].hex(), 16)
        for i in range(image_num):
            image_pixels = [pixel for pixel in file[16 + 784 * i:16 + 784 * (i + 1)]]
            image_array = np.array(image_pixels, dtype=np.uint8).reshape(28, 28)
            image_arrays.append(image_array)
        f.close()
    return image_arrays


def load_train_labels():
    label_array = []
    with open('train-labels.idx1-ubyte', 'rb') as f:
        file = f.read()
        # label_num = int(file[4:8].hex(), 16)
        for i in range(label_num):
            label_array.append(file[8 + i])
        f.close()
    return label_array


def load_test_images():
    image_arrays = []
    with open('t10k-images.idx3-ubyte', 'rb') as f:
        file = f.read()
        # image_num = int(file[4:8].hex(), 16)
        for i in range(image_num):
            image_pixels = [pixel for pixel in file[16 + 784 * i:16 + 784 * (i + 1)]]
            image_array = np.array(image_pixels, dtype=np.uint8).reshape(28, 28)
            image_arrays.append(image_array)
        f.close()
    return image_arrays


def load_test_labels():
    label_array = []
    with open('t10k-labels.idx1-ubyte', 'rb') as f:
        file = f.read()
        # label_num = int(file[4:8].hex(), 16)
        for i in range(label_num):
            label_array.append(file[8 + i])
        f.close()
    return label_array


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
        self.wih = np.random.normal(0.0, pow(self.hidden_nodes,-0.5), (self.hidden_nodes, self.input_nodes))
        self.who = np.random.normal(0.0, pow(self.output_nodes,-0.5), (self.output_nodes, self.hidden_nodes))

        self.activation_function = lambda x: scipy.special.expit(x)  # sigmoid函数

    def train(self, input_list, targets_list):
        inputs = np.transpose(input_list)
        targets = np.transpose(targets_list)

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

    def test(self, test_images_list, test_labels_list):
        score = 0
        for index, image in enumerate(test_images_list):
            res = self.query(image)
            if np.argmax(res) == test_labels_list[index]:
                score += 1
        return score / len(test_labels_list)

    def save_who_matrix(self):
        np.savetxt('who', self.who)

    def load_who_matrix(self):
        self.who = np.load_text('who')

    def save_wih_matrix(self):
        np.savetxt('wih', self.wih)

    def load_wih_matrix(self):
        self.wih = np.load_text('wih')


if __name__ == '__main__':
    # test_images = [1 / 255 * 0.99 * image + 0.01 for image in load_test_images()]
    # test_labels = load_test_labels()
    train_images = [1 / 255 * 0.99 * image + 0.01 for image in load_train_images()]
    train_label = load_train_labels()
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10
    learning_rate = 0.3
    network = NetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    for index, image in enumerate(train_images):
        target = np.zeros(output_nodes) + 0.01
        target[train_label[index]] = 0.99
        network.train(image, target)
    network.save_who_matrix()
    network.save_wih_matrix()
