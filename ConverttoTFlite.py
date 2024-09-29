import tensorflow as tf
import numpy as np

# Load the Keras model
model = tf.keras.models.load_model('C:/Users/User/Desktop/Scalable Computing/sample-code - Pillow/test.h5')

# Print the model output details before conversion
print("Keras Model Output Shapes:")
for output in model.outputs:
    print(output.shape)

# Convert to TFLite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open("test.tflite", "wb") as f:
    f.write(tflite_model)

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Print TFLite output details
output_details = interpreter.get_output_details()
print("TFLite Model Output Details:")
for output in output_details:
    print(output['shape'], output['dtype'])
