from config import CONFIG
from datagen import DataGenerator
from utils import *
import matplotlib.pyplot as plt


class ConvShapeCalculator:
    def __init__(self, input_shape):
        self.shape = input_shape

    def apply_conv(self, kernel_size, stride=1):
        self.shape = (self.shape - kernel_size) // stride + 1

    def output_shape(self):
        return self.shape


def main():
    calc = ConvShapeCalculator(96)
    calc.apply_conv(3, 1)
    calc.apply_conv(3, 2)
    calc.apply_conv(3, 1)
    calc.apply_conv(3, 2)
    calc.apply_conv(3, 1)
    calc.apply_conv(3, 2)
    calc.apply_conv(3, 1)
    calc.apply_conv(3, 2)
    print(calc.output_shape())


if __name__ == '__main__':
    datagen = DataGenerator(CONFIG.HR_DIR,
                            CONFIG.INPUT_SHAPE,
                            down_sample_scale=CONFIG.DOWN_SAMPLE_SCALE,
                            batch_size=CONFIG.BATCH_SIZE)
    Image.fromarray((datagen[0][0][0] * 255).astype(np.uint8)).show()
