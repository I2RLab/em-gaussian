import pygame
import numpy as np
import xlsxwriter

# Define some colors.
BLACK = pygame.Color('black')
WHITE = pygame.Color('white')
GREEN = pygame.Color('green')

workbook = xlsxwriter.Workbook('IO_sample.xlsx')

worksheet = workbook.add_worksheet()


# This is a simple class that will help us print to the screen.
# It has nothing to do with the joysticks, just outputting the
# information.
class TextPrint(object):
    def __init__(self):
        self.reset()
        self.font = pygame.font.Font(None, 20)

    def tprint(self, screen, textString):
        textBitmap = self.font.render(textString, True, BLACK)
        screen.blit(textBitmap, (self.x, self.y))
        self.y += self.line_height

    def reset(self):
        self.x = 10
        self.y = 10
        self.line_height = 15

    def indent(self):
        self.x += 10

    def unindent(self):
        self.x -= 10


pygame.init()

# Set the width and height of the screen (width, height).
screen = pygame.display.set_mode((1500, 700))

pygame.display.set_caption("My Game")

# Loop until the user clicks the close button.
done = False

# Used to manage how fast the screen updates.
clock = pygame.time.Clock()
ticks = pygame.time.get_ticks()
# Initialize the joysticks.
pygame.joystick.init()

# Get ready to print.
textPrint = TextPrint()

points = np.array([[0, 500], [1, 500]])

trig = False
input_sample = list()
output_sample = list()
agent = 0

input_flag, output_flag = False, False


# -------- Main Program Loop -----------
while not done:


    ticks = pygame.time.get_ticks()
    #
    # EVENT PROCESSING STEP
    #
    # Possible joystick actions: JOYAXISMOTION, JOYBALLMOTION, JOYBUTTONDOWN,
    # JOYBUTTONUP, JOYHATMOTION
    for event in pygame.event.get(): # User did something.
        if event.type == pygame.QUIT: # If user clicked close.
            done = True # Flag that we are done so we exit this loop.
        elif event.type == pygame.JOYBUTTONDOWN:
            print("Joystick button pressed.")
        elif event.type == pygame.JOYBUTTONUP:
            print("Joystick button released.")


    #
    # DRAWING STEP
    #
    # First, clear the screen to white. Don't put other drawing commands
    # above this, or they will be erased with this command.
    screen.fill(WHITE)
    textPrint.reset()

    # pygame.draw.line(screen, BLACK, [0, 500], [1500, 500], 5)

    # Get count of joysticks.
    joystick_count = pygame.joystick.get_count()

    textPrint.tprint(screen, "Number of joysticks: {}".format(joystick_count))
    textPrint.indent()

    # For each joystick:
    for i in range(joystick_count):
        joystick = pygame.joystick.Joystick(i)
        joystick.init()

        textPrint.tprint(screen, "Joystick {}".format(i))
        textPrint.indent()

        # Get the name from the OS for the controller/joystick.
        name = joystick.get_name()
        textPrint.tprint(screen, "Joystick name: {}".format(name))

        # Usually axis run in pairs, up/down for one, and left/right for
        # the other.
        axes = joystick.get_numaxes()
        textPrint.tprint(screen, "Number of axes: {}".format(axes))
        textPrint.indent()

        for i in range(axes):
            axis = joystick.get_axis(i)

            textPrint.tprint(screen, "Axis {} value: {:>6.3f}".format(i, axis))
            if i == 1:
                if input_flag:
                    new_point = np.array([[points[-1][0] + 1, 500 + 10 * axis]])
                    points = np.concatenate((points, new_point))
                    pygame.draw.lines(screen, BLACK, False, points)
                    if trig and len(input_sample) < 300:
                        input_sample.append(round((-axis + 1) / 2  * 4, 0) / 5 + 0.2)

                    if len(input_sample) >= 300 and trig:
                        print('input_sample')
                        print(input_sample)
                        trig = False
                        worksheet.write_column(0,agent,input_sample)

                        input_sample = list()
                        agent += 1
                        if agent > 2:
                            workbook.close()
                if output_flag:
                    new_point = np.array([[points[-1][0] + 1, 600 + 10 * axis]])
                    points = np.concatenate((points, new_point))
                    pygame.draw.lines(screen, GREEN, False, points)
                    if trig and len(output_sample) < 300:
                        output_sample.append(round((-axis + 1) * 4, 0))

                    if len(output_sample) >= 300 and trig:
                        print('output_sample')
                        print(output_sample)
                        trig = False
                        worksheet.write_column(0, 3, output_sample)

                        output_sample = list()
                        workbook.close()




        textPrint.unindent()

        buttons = joystick.get_numbuttons()
        textPrint.tprint(screen, "Number of buttons: {}".format(buttons))
        textPrint.indent()

        for i in range(buttons):
            button = joystick.get_button(i)
            if i == 0:
                if button == 1:
                    trig = True
            if i == 3:
                if button == 1:
                    input_flag = True
                    output_flag = False
            if i == 4:
                if button == 1:
                    input_flag = False
                    output_flag = True


            textPrint.tprint(screen, "Button {:>2} value: {}".format(i, button))
        textPrint.unindent()

        hats = joystick.get_numhats()
        textPrint.tprint(screen, "Number of hats: {}".format(hats))
        textPrint.indent()

        # Hat position. All or nothing for direction, not a float like
        # get_axis(). Position is a tuple of int values (x, y).
        for i in range(hats):
            hat = joystick.get_hat(i)
            textPrint.display(screen, "Hat {} value: {}".format(i, str(hat)))
        textPrint.unindent()

        textPrint.unindent()

    #
    # ALL CODE TO DRAW SHOULD GO ABOVE THIS COMMENT
    #

    # Go ahead and update the screen with what we've drawn.
    pygame.display.flip()

    # Limit to 20 frames per second.
    clock.tick(20)


# Close the window and quit.
# If you forget this line, the program will 'hang'
# on exit if running from IDLE.
pygame.quit()
