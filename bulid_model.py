import tensorflow as tf
import keras

# Constants
EPOCHS = 5
BATCH_SIZE = 16
MODEL_H5_PATH = "kws_model.h5"
MODEL_TFLITE_PATH = "kws_model.tflite"


def build_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(44, 40, 1)),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid') 
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def train_and_export(X, y):

    model = build_model()
    print("\n--- Model Summary ---")
    model.summary()

    print("\n--- Starting Training ---")
    model.fit(X, y,
          validation_split=0.2,
          shuffle=True,
          epochs=EPOCHS,
          batch_size=BATCH_SIZE)

    model.save(MODEL_H5_PATH)
    print(f"✅ Model saved to {MODEL_H5_PATH}")

    print("\n--- Converting to TFLite ---")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(MODEL_TFLITE_PATH, "wb") as f:
        f.write(tflite_model)
    print(f"✅ Model exported to {MODEL_TFLITE_PATH}")

    print("\n--- Validating TFLite Model ---")
    interpreter = tf.lite.Interpreter(model_path=MODEL_TFLITE_PATH)
    interpreter.allocate_tensors()
    print("✅ TFLite model loaded and ready.")

