import numpy as np


class TrainingData:
    def __init__(self):
        self.input_output_dict = dict()
        self.input_num_units = 11
        self.input_training = []
        self.ouput_training = []

    def io_sequence_generator(self):
        t = 0
        for i1 in range(1, self.input_num_units):
            for i2 in range(1, self.input_num_units):
                for i3 in range(1, self.input_num_units):
                    u = [i1, i2, i3]
                    indexes = [(i, x) for i, x in enumerate(u) if x == min(u)]
                    # print(u, indexes)
                    for i_r in indexes:
                        if i_r[1] < 8:
                            self.input_output_dict[t, i1, i2, i3] = i_r[0] + 2
                            self.input_training.append([i1, i2, i3])
                            self.input_training.append([i1, i2, i3])
                            self.ouput_training.append([i_r[0] + 2])
                            self.ouput_training.append([i_r[0] + 2])
                        else:
                            self.input_output_dict[t, i1, i2, i3] = 1
                            self.input_training.append([i1, i2, i3])
                            self.input_training.append([i1, i2, i3])
                            self.ouput_training.append([1])
                            self.ouput_training.append([1])

                        t += 1

        # print('input_output_dict')
        # print(self.input_output_dict)
        return np.array(self.input_training), np.array(self.ouput_training)


if __name__ == '__main__':

    TD = TrainingData()
    [i,o] = TD.io_sequence_generator()
    print(i,'\n', o)