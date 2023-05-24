import tensorflow as tf
from keras import backend as K
from utils import load_config_partI


def train_model(train_dataset, valid_dataset, total_train, total_val, config):

    config = load_config_partI()

    ###################################
    # Create Model
    ###################################

    target_shape = config['batch_size']

    IMG_WIDTH = 32
    IMG_HEIGHT = 32
    IMG_CHANNELS = 1

    # Inputs
    inputs_img = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    # inputs_location = tf.keras.layers.Input((3,))

    def flux(inputs_img, kernel_size=config['kernel_size'], dropout=config['dropout'], padding=config['padding']):
        # Input layer
        x = inputs_img

        # Convolutional Layer 1
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=(kernel_size, kernel_size), padding=padding,
                                   activation='elu',
                                   name='conv2d')(x)
        x = tf.keras.layers.Dropout(dropout)(x)  # Add dropout layer after convolutional layer 1

        # Max Pooling Layer 1
        x = tf.keras.layers.MaxPool2D((2, 2), name='max_pooling2d')(x)

        # Convolutional Layer 2
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=(kernel_size, kernel_size), padding=padding,
                                   activation='elu',
                                   name='conv2d_1')(x)
        x = tf.keras.layers.Dropout(dropout)(x)  # Add dropout layer after convolutional layer 1

        # Max Pooling Layer 2
        x = tf.keras.layers.MaxPool2D((2, 2), name='max_pooling2d_1')(x)

        # Convolutional Layer 3
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=(kernel_size, kernel_size), padding=padding,
                                   activation='elu',
                                   name='conv2d_2')(x)
        x = tf.keras.layers.Dropout(dropout)(x)  # Add dropout layer after convolutional layer 1

        # Max Pooling Layer 3
        x = tf.keras.layers.MaxPool2D((2, 2), name='max_pooling2d_2')(x)

        # Flattening Layer
        x = tf.keras.layers.Flatten(name='flatten')(x)

        # Dense Layers for mu1 with dropout
        mu1 = tf.keras.layers.Dense(128, activation='elu', name='Dense_1_mu1')(x)
        mu1 = tf.keras.layers.Dropout(dropout)(mu1)  # Add dropout
        mu1 = tf.keras.layers.Dense(64, activation='elu', name='Dense_2_mu1')(mu1)
        mu1 = tf.keras.layers.Dropout(dropout)(mu1)  # Add dropout
        mu1 = tf.keras.layers.Dense(32, activation='elu', name='Dense_3_mu1')(mu1)
        mu1 = tf.keras.layers.Dropout(dropout)(mu1)  # Add dropout
        mu1 = tf.keras.layers.Dense(1, activation='linear', name='mu1')(mu1)

        # Dense Layers for mu2 with dropout
        mu2 = tf.keras.layers.Dense(128, activation='elu', name='Dense_1_mu2')(x)
        mu2 = tf.keras.layers.Dropout(dropout)(mu2)  # Add dropout
        mu2 = tf.keras.layers.Dense(64, activation='elu', name='Dense_2_mu2')(mu2)
        mu2 = tf.keras.layers.Dropout(dropout)(mu2)  # Add dropout
        mu2 = tf.keras.layers.Dense(32, activation='elu', name='Dense_3_mu2')(mu2)
        mu2 = tf.keras.layers.Dropout(dropout)(mu2)  # Add dropout
        mu2 = tf.keras.layers.Dense(1, activation='linear', name='mu2')(mu2)

        # Dense Layers for mu3 with dropout
        mu3 = tf.keras.layers.Dense(128, activation='elu', name='Dense_1_mu3')(x)
        mu3 = tf.keras.layers.Dropout(dropout)(mu3)  # Add dropout
        mu3 = tf.keras.layers.Dense(64, activation='elu', name='Dense_2_mu3')(mu3)
        mu3 = tf.keras.layers.Dropout(dropout)(mu3)  # Add dropout
        mu3 = tf.keras.layers.Dense(32, activation='elu', name='Dense_3_mu3')(mu3)
        mu3 = tf.keras.layers.Dropout(dropout)(mu3)  # Add dropout
        mu3 = tf.keras.layers.Dense(1, activation='linear', name='mu3')(mu3)

        # Concatenation Layer
        outputs = tf.keras.layers.concatenate([mu1, mu2, mu3], name='concatenate')

        # Model
        model = tf.keras.Model(inputs=inputs_img, outputs=outputs)

        return model

    def mse(y_true, y_pred):
        return K.mean(K.square(y_true - y_pred))

    def mae(y_true, y_pred):
        return K.mean(K.abs(y_true - y_pred))

    def mre(y_true, y_pred, eps=1e-6):
        return K.mean(K.abs(y_true - y_pred) / (K.abs(y_true) + eps))

    ###################################
    # Call Model
    ###################################

    FLUX = flux(inputs_img=inputs_img)
    FLUX.summary()

    #################################
    # Compile Model
    #################################

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=config['earlystopping_patience'], verbose=0, mode='auto',
        restore_best_weights=True
    )

    def lr_scheduler(epoch, lr):
        if epoch < 7:
            return lr
        else:
            return lr * tf.math.exp(-config['lr_decay_rate'])

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

    opt = tf.keras.optimizers.Adam(learning_rate=config['init_learning_rate'])

    FLUX.compile(optimizer=opt,
                  loss=config['loss_fn'],
                  metrics=[mse, mae, mre])

    # callbacks
    callbacks = [early_stop, lr_callback]

    FLUX.fit(x=train_dataset,
              epochs=config['epochs'],
              steps_per_epoch=total_train // config['batch_size'],
              validation_data=valid_dataset,
              validation_steps = total_val // config['batch_size'],
              callbacks=callbacks,
              verbose=1)
