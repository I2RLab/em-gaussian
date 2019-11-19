import numpy as np
import matplotlib.pyplot as plt


class TrainingData:
    def __init__(self):
        self.input_output_dict = dict()
        self.input_num_units = 11
        self.input_training = []
        self.output_training = []

    def io_sequence_generator(self):
        self.input_training = np.array([[10, 10, 10], [10, 10, 9.], [10, 10, 8.], [10, 10, 8.], [10, 9., 8.], [10, 9., 7.], [10, 9., 6.], [10, 8., 6.],
                                        [10, 10, 6.], [10, 10, 6.], [10, 10, 6.], [10, 10, 4.], [10, 10, 4.], [10, 10, 2.], [10, 10, 1.], [10, 10, 1.],
                                        [1., 10, 10], [1., 10, 9.], [1., 10, 8.], [1., 10, 8.], [1., 10, 7.], [2., 10, 7.], [4., 10, 9.], [4., 10, 9.],
                                        [10, 3., 8.], [9., 5., 8.], [10, 3., 9.], [10, 5., 9.], [10, 4., 10], [10, 4., 10], [10, 3., 10], [9., 1., 10],
                                        [10, 8., 10], [10, 9., 10], [4., 10, 10], [4., 10, 10], [4., 9., 10], [10, 1., 10], [10, 2., 10], [10, 1., 10]])

        self.output_training = np.array([[1], [1], [1], [1], [1], [1], [1], [1],
                                         [4], [4], [4], [4], [4], [4], [4], [4],
                                         [2], [2], [2], [2], [2], [2], [2], [2],
                                         [3], [3], [3], [3], [3], [3], [3], [3],
                                         [1], [1], [1], [2], [2], [2], [3], [3]])


        ax1 = plt.subplot(211)
        plt.plot(self.input_training[:,0], "X")
        plt.plot(self.input_training[:,1], "v")
        plt.plot(self.input_training[:,2], "P")
        plt.ylabel('Performance level')
        ax1.set_xticks(np.arange(0, 40, 1))
        ax1.set_yticks(np.arange(1, 11, 1))
        plt.grid(markevery=3)
        plt.legend(('Y = 2', 'Y = 3', 'Y = 4'))

        ax2 = plt.subplot(212)
        plt.plot(self.output_training)
        plt.ylabel('Mode')
        plt.xlabel('time')
        ax2.set_yticks(np.arange(1,5,1))
        plt.grid()
        plt.savefig('test_data_plot.png')
        plt.show()


        return np.array(self.input_training), np.array(self.output_training)


if __name__ == '__main__':
    TD = TrainingData()
    [i, o] = TD.io_sequence_generator()
    print(i, '\n', o)
