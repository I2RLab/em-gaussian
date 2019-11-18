import numpy as np


class TrainingData:
    def __init__(self):
        self.input_output_dict = dict()
        self.input_num_units = 11
        self.input_training = []
        self.ouput_training = []

    def io_sequence_generator(self):
        self.input_training = np.array([[10, 10, 10], [10, 10, 9.], [10, 10, 8.], [10, 10, 1.], [10, 9, 1], [10, 8, 7], [10, 8, 8], [10, 10, 10],
                                        [10, 10, 1.], [10, 10, 1.], [10, 10, 1.], [10, 10, 1.], [10, 10, 1.], [10, 10, 1.], [10, 10, 1.], [10, 10, 1.],
                                        [8., 10, 4.], [1., 10, 5.], [1., 10, 5.], [1., 10, 5.], [1., 10, 6.], [2., 10, 7.], [3., 10, 8.], [4., 10, 9.],
                                        [5., 10, 10], [5., 10, 9.], [5., 10, 10], [8., 10, 10], [8., 10, 10], [8., 10, 10], [8., 10, 10], [9., 10, 10],
                                        [10, 1., 10], [10, 1., 10], [10, 1., 10], [10, 1., 10], [10, 5., 10], [10, 4., 10], [10, 4., 10], [10, 6., 10],])

        self.ouput_training = np.array([[1], [1], [1], [2], [1], [1], [1], [1],
                                        [4], [4], [4], [4], [4], [4], [4], [4],
                                        [4], [2], [2], [2], [2], [2], [2], [2],
                                        [2], [1], [1], [2], [2], [2], [1], [1],
                                        [3], [3], [3], [3], [3], [3], [3], [3]])

        return np.array(self.input_training), np.array(self.ouput_training)


if __name__ == '__main__':

    TD = TrainingData()
    [i, o] = TD.io_sequence_generator()
    print(i, '\n', o)
