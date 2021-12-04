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
    main()