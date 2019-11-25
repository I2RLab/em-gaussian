import numpy as np
import matplotlib.pyplot as plt

class TrainingData:
    def __init__(self):
        self.input_output_dict = dict()
        self.input_num_units = 11
        self.input_training = []
        self.output_training = []

    def io_sequence_generator(self):
        t = 0

        for i1 in range(1, self.input_num_units):
            for i2 in range(1, self.input_num_units):
                for i3 in range(1, self.input_num_units):
                    u = [i1, i2, i3]
                    indexes = [(i, x) for i, x in enumerate(u) if x == min(u)]

                    for i_r in indexes:
                        if i_r[1] < 7:
                            self.input_output_dict[t, i1, i2, i3] = i_r[0] + 2
                            self.input_training.append([i1, i2, i3])
                            self.input_training.append([i1, i2, i3])
                            self.input_training.append([i1, i2, i3])
                            self.output_training.append([i_r[0] + 2])
                            self.output_training.append([i_r[0] + 2])
                            self.output_training.append([i_r[0] + 2])
                        else:
                            self.input_output_dict[t, i1, i2, i3] = 1
                            self.input_training.append([i1, i2, i3])
                            self.input_training.append([i1, i2, i3])
                            self.input_training.append([i1, i2, i3])
                            self.input_training.append([i1, i2, i3])
                            self.input_training.append([i1, i2, i3])
                            self.output_training.append([1])
                            self.output_training.append([1])
                            self.output_training.append([1])
                            self.output_training.append([1])
                            self.output_training.append([1])

                        t += 1
        '''
        # plot data
        ax1 = plt.subplot(211)
        plt.plot(self.input_training[:], "X")
        # plt.plot(self.input_training[1][:], "v")
        # plt.plot(self.input_training[2][:], "P")
        plt.ylabel('Performance level')
        # ax1.set_xticks(np.arange(0, 40, 1))
        # ax1.set_yticks(np.arange(1, 11, 1))
        plt.grid(markevery=3)
        plt.legend(('Y = 2', 'Y = 3', 'Y = 4'))

        ax2 = plt.subplot(212)
        plt.plot(self.output_training)
        plt.ylabel('Mode')
        plt.xlabel('time')
        ax2.set_yticks(np.arange(1, 5, 1))
        plt.grid()

        plt.show()
        '''

        return np.array(self.input_training), np.array(self.output_training)


if __name__ == '__main__':

    TD = TrainingData()
    [i,o] = TD.io_sequence_generator()
    print(i,'\n', o)