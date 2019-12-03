import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=700)
np.set_printoptions(precision=3, edgeitems=50)

class TrainingData:
    def __init__(self):
        self.input_output_dict = dict()
        self.input_num_units = 11
        self.input_training = list()
        self.output_training = list()
        self.output_f_training = list()

    def feedback_generator(self, input, output):
        feedback_base5 = ''
        for r, u_r in enumerate(input):
            if output - 2 == r:  # manual mode
                if u_r > 7:
                    f_r = '3'
                elif u_r > 5:
                    f_r = '2'
                elif u_r > 3:
                    f_r = '1'
                else:
                    f_r = '0'
            else:               # autonomous mode
                if u_r > 7:
                    f_r = '4'
                elif u_r > 5:
                    f_r = '3'
                elif u_r > 3:
                    f_r = '2'
                else:
                    f_r = '1'

            feedback_base5 += f_r
        # print('input {}   output {}   f_5 {}   f_125 {}'.format(input, output, feedback_base5, int(feedback_base5, 5)))

        return int(feedback_base5, 5)






    def io_sequence_generator(self):
        t = 0

        for i1 in reversed(range(1, self.input_num_units)):
            for i2 in reversed(range(1, self.input_num_units)):
                for i3 in reversed(range(1, self.input_num_units)):
                    input3 = [i1, i2, i3]
                    indexes = [(i, x) for i, x in enumerate(input3) if x == min(input3)]
                    for i_r in indexes:
                        if i_r[1] < 8:
                            output_y = i_r[0] + 2
                            feedback = self.feedback_generator(input3, output_y)
                            self.input_output_dict[t, i1, i2, i3] = output_y

                            self.input_training.append([i1, i2, i3])
                            self.output_training.append(output_y)
                            self.output_f_training.append(feedback)

                            self.input_training.append([i1, i2, i3])
                            self.output_training.append(output_y)
                            self.output_f_training.append(feedback)

                        if i_r[1] > 6:
                            output_y = 1
                            feedback = self.feedback_generator(input3, output_y)
                            self.input_output_dict[t, i1, i2, i3] = 1

                            self.input_training.append([i1, i2, i3])
                            self.output_training.append(output_y)
                            self.output_f_training.append(feedback)

                            self.input_training.append([i1, i2, i3])
                            self.output_training.append(output_y)
                            self.output_f_training.append(feedback)

                            self.input_training.append([i1, i2, i3])
                            self.output_training.append(output_y)
                            self.output_f_training.append(feedback)
                            #
                            # self.input_training.append([i1, i2, i3])
                            # self.output_training.append(output_y)
                            # self.output_f_training.append(feedback)

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

        return np.array(self.input_training), np.array(self.output_training), np.array(self.output_f_training)


if __name__ == '__main__':

    TD = TrainingData()
    [i, o, o_f] = TD.io_sequence_generator()
    # print(i,'\n', o, '\n', o_f)


    # for output in range(1, 5):
    #     for input1 in range(1,11):
    #         for input2 in range(1,11):
    #             for input3 in range(1,11):
    #                 TD.feedback_generator([input1, input2, input3], output)