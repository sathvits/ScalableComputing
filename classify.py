#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
from PIL import Image
import numpy
import string
import random
import argparse
from tflite_runtime.interpreter import Interpreter
# import tensorflow as tf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='Model name to use for classification', type=str)
    parser.add_argument('--captcha-dir', help='Where to read the captchas to break', type=str)
    parser.add_argument('--output', help='File where the classifications should be saved', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    args = parser.parse_args()

    if args.model_name is None:
        print("Please specify the TFLite model to use")
        exit(1)

    if args.captcha_dir is None:
        print("Please specify the directory with captchas to break")
        exit(1)

    if args.output is None:
        print("Please specify the path to the output file")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    print("Classifying captchas with symbol set {" + captcha_symbols + "}")

    with open(args.output, 'w') as output_file:
        print("Entered loop")

        # Load TFLite model and allocate tensors
        interpreter = Interpreter(model_path=args.model_name + '.tflite')      # for Rasberry with tflite_runtime

        print("Interpreter created", interpreter)
        # interpreter = tf.lite.Interpreter(model_path=args.model_name + '.tflite')      # for Anaconda with Tensorflow Lite
        interpreter.allocate_tensors()

        # Print TFLite output details
        output_details = interpreter.get_output_details()
        

        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        for x in os.listdir(args.captcha_dir):
            # Load image and preprocess it
            raw_data = Image.open(os.path.join(args.captcha_dir, x))
            if raw_data is None:
                break
            rgb_data = raw_data.convert("RGB")
            image = numpy.array(rgb_data) / 255.0
            image = image.astype(numpy.float32)

            # print("Before : ", image)
            image = numpy.expand_dims(image, axis=0)

            # Set the tensor to the input data
            interpreter.set_tensor(input_details[0]['index'], image)

            # print("Trying to invoke:")
            # Run inference
            interpreter.invoke()

            mapper = {
                0: "0",
                1: "1",
                2: "2",
                3: "3",
                4: "4",
                5: "5",
                6: "6",
                7: "7",
                8: "8",
                9: "9",
                10: "A",
                11: "B",
                12: "C",
                13: "D",
                14: "E",
                15: "F",
                16: "G",
                17: "H",
                18: "I",
                19: "J",
                20: "K",
                21: "L",
                22: "M",
                23: "N",
                24: "O",
                25: "P",
                26: "Q",
                27: "R",
                28: "S",
                29: "T",
                30: "U",
                31: "V",
                32: "W",
                33: "X",
                34: "Y",
                35: "Z"
            }


            output_data = [interpreter.get_tensor(output['index']) for output in output_details]
            finalString = ""
            
            for output in output_data:
                finalString += mapper[numpy.argmax(numpy.squeeze(output))]

            print("finalString: ", finalString)

            output_file.write(x + ", " + finalString + "\n")
            
            print('Classified ' + x)

if __name__ == '__main__':
    main()
