# train_pose_model.py

import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras import backend as K
from load_pose_data import X_train, y_train

MODEL_PATH = 'pose_model.h5'

def create_pose_model(input_shape):
    inputs = Input(shape=input_shape)

    x = Conv2D(4, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(512, (3, 3), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    outputs = Dense(16, activation='linear')(x)  # Adjusted to 16 units

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

input_shape = X_train[0].shape
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    model = create_pose_model(input_shape)
model.compile(optimizer='adam', loss='mse')

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
model.fit(X_train, y_train, validation_split=0.1, epochs=1000, batch_size=2, callbacks=[reduce_lr])
model.save(MODEL_PATH)

K.clear_session()
