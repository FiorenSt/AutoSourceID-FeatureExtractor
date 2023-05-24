import tensorflow as tf
from keras import backend as K
from utils import load_config_partII


def train_model(train_dataset, valid_dataset, total_train, total_val, config):

    config = load_config_partII()

    ###################################
    # Create Model
    ###################################

    target_shape = config['batch_size']

    IMG_WIDTH = 32
    IMG_HEIGHT = 32
    IMG_CHANNELS = 1

    # Inputs
    inputs_img = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    def flux(inputs_img, kernel_size=config['kernel_size'], dropout=config['dropout'], padding=config['padding'],
             reg=config['regularization']):

        # Input layer
        x = inputs_img

        # Convolutional Layer 1
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=(kernel_size, kernel_size), padding=padding,
                                   activation='elu',
                                   name='conv2d')(x)

        # Max Pooling Layer 1
        x = tf.keras.layers.MaxPool2D((2, 2), name='max_pooling2d')(x)

        # Convolutional Layer 2
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=(kernel_size, kernel_size), padding=padding,
                                   activation='elu',
                                   name='conv2d_1')(x)

        # Max Pooling Layer 2
        x = tf.keras.layers.MaxPool2D((2, 2), name='max_pooling2d_1')(x)

        # Convolutional Layer 3
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=(kernel_size, kernel_size), padding=padding,
                                   activation='elu',
                                   name='conv2d_2')(x)

        # Max Pooling Layer 3
        x = tf.keras.layers.MaxPool2D((2, 2), name='max_pooling2d_2')(x)

        # Flattening Layer
        x = tf.keras.layers.Flatten(name='flatten')(x)

        # Dense Layers for mu1 with dropout
        mu1 = tf.keras.layers.Dense(128, activation='elu', name='Dense_1_mu1')(x)
        mu1 = tf.keras.layers.Dense(64, activation='elu', name='Dense_2_mu1')(mu1)
        mu1 = tf.keras.layers.Dense(32, activation='elu', name='Dense_3_mu1')(mu1)
        mu1 = tf.keras.layers.Dense(1, activation='linear', name='mu1')(mu1)

        # Dense Layers for mu2 with dropout
        mu2 = tf.keras.layers.Dense(128, activation='elu', name='Dense_1_mu2')(x)
        mu2 = tf.keras.layers.Dense(64, activation='elu', name='Dense_2_mu2')(mu2)
        mu2 = tf.keras.layers.Dense(32, activation='elu', name='Dense_3_mu2')(mu2)
        mu2 = tf.keras.layers.Dense(1, activation='linear', name='mu2')(mu2)

        # Dense Layers for mu3 with dropout
        mu3 = tf.keras.layers.Dense(128, activation='elu', name='Dense_1_mu3')(x)
        mu3 = tf.keras.layers.Dense(64, activation='elu', name='Dense_2_mu3')(mu3)
        mu3 = tf.keras.layers.Dense(32, activation='elu', name='Dense_3_mu3')(mu3)
        mu3 = tf.keras.layers.Dense(1, activation='linear', name='mu3')(mu3)

        ############# SIGMA PREDICTION

        # Input layer
        x2 = inputs_img

        # Convolutional Layer 1
        x2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(kernel_size, kernel_size), padding=padding,
                                    activation='elu',
                                    name='conv2d_x2')(x2)

        # Max Pooling Layer 1
        x2 = tf.keras.layers.MaxPool2D((2, 2), name='max_pooling2d_x2')(x2)

        # Convolutional Layer 2
        x2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(kernel_size, kernel_size), padding=padding,
                                    activation='elu',
                                    name='conv2d_1_x2')(x2)

        # Max Pooling Layer 2
        x2 = tf.keras.layers.MaxPool2D((2, 2), name='max_pooling2d_1_x2')(x2)

        # Convolutional Layer 3
        x2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(kernel_size, kernel_size), padding=padding,
                                    activation='elu',
                                    name='conv2d_2_x2')(x2)

        # Max Pooling Layer 3
        x2 = tf.keras.layers.MaxPool2D((2, 2), name='max_pooling2d_2_x2')(x2)

        # Flattening Layer
        x2 = tf.keras.layers.Flatten(name='flatten_x2')(x2)

        # Sigma1 layers
        sigma1 = tf.keras.layers.Dense(128, activation='elu', name='Dense_1_sigma1',
                                       kernel_regularizer=tf.keras.regularizers.L2(reg))(x2)
        sigma1 = tf.keras.layers.Dense(64, activation='elu', name='Dense_2_sigma1',
                                       kernel_regularizer=tf.keras.regularizers.L2(reg))(sigma1)
        sigma1 = tf.keras.layers.Dense(32, activation='elu', name='Dense_3_sigma1',
                                       kernel_regularizer=tf.keras.regularizers.L2(reg))(sigma1)
        sigma1 = tf.keras.layers.Dense(1, activation='linear', name='sigma1',
                                       kernel_regularizer=tf.keras.regularizers.L2(reg))(sigma1)

        # Sigma2 layers
        sigma2 = tf.keras.layers.Dense(128, activation='elu', name='Dense_1_sigma2',
                                       kernel_regularizer=tf.keras.regularizers.L2(reg))(x2)
        sigma2 = tf.keras.layers.Dense(64, activation='elu', name='Dense_2_sigma2',
                                       kernel_regularizer=tf.keras.regularizers.L2(reg))(sigma2)
        sigma2 = tf.keras.layers.Dense(32, activation='elu', name='Dense_3_sigma2',
                                       kernel_regularizer=tf.keras.regularizers.L2(reg))(sigma2)
        sigma2 = tf.keras.layers.Dense(1, activation='linear', name='sigma2',
                                       kernel_regularizer=tf.keras.regularizers.L2(reg))(sigma2)

        # Sigma3 layers
        sigma3 = tf.keras.layers.Dense(128, activation='elu', name='Dense_1_sigma3',
                                       kernel_regularizer=tf.keras.regularizers.L2(reg))(x2)
        sigma3 = tf.keras.layers.Dense(64, activation='elu', name='Dense_2_sigma3',
                                       kernel_regularizer=tf.keras.regularizers.L2(reg))(sigma3)
        sigma3 = tf.keras.layers.Dense(32, activation='elu', name='Dense_3_sigma3',
                                       kernel_regularizer=tf.keras.regularizers.L2(reg))(sigma3)
        sigma3 = tf.keras.layers.Dense(1, activation='linear', name='sigma3',
                                       kernel_regularizer=tf.keras.regularizers.L2(reg))(sigma3)

        ######## CONCATENATE ALL RESULTS

        # Concatenation Layer
        outputs = tf.keras.layers.concatenate([sigma1, sigma2, sigma3, mu1, mu2, mu3], name='concatenate')

        # Model
        model = tf.keras.Model(inputs=inputs_img, outputs=outputs)

        return model

    def gauss_loss(y_true, y_pred):  ## PREDICTS MEAN AND LOG(VAR)
        pred_mean = y_pred[:, 3:6]
        pred_var = K.exp(y_pred[:, 0:3]) + 1e-6
        return tf.reduce_mean(K.log(pred_var) + (((y_true - pred_mean) ** 2) / pred_var))

    def mse(y_true, y_pred):
        pred_mean = y_pred[:, 3:6]
        return K.mean(K.square(y_true - pred_mean))

    def mae(y_true, y_pred):
        pred_mean = y_pred[:, 3:6]
        return K.mean(K.abs(y_true - pred_mean))

    def mre(y_true, y_pred, eps=1e-6):
        pred_mean = y_pred[:, 3:6]
        return K.mean(K.abs(y_true - pred_mean) / (K.abs(y_true) + eps))

    ###################################
    # Call Model
    ###################################

    FLUX = flux(inputs_img)

    # ### SET THE LAYER OF THE PREVIOUS NETWORK TO UNTRAINABLE
    # if config['freezing']=='True':
    #     FLUX.layers[32].trainable = False

    PartI.summary()
    FLUX.summary()

    # Get the list of layer names in both models
    old_layer_names = [layer.name for layer in PartI.layers][1:-1]
    new_layer_names = [layer.name for layer in FLUX.layers]

    # Iterate through each layer in the new model
    count = 0
    for i, new_layer in enumerate(FLUX.layers):
        new_layer_name = new_layer.name

        # Check if the new layer has a matching name in the old model
        if new_layer_name in old_layer_names:
            # Find the index of the matching layer in the old model
            old_layer_idx = old_layer_names.index(new_layer_name) + 1

            # Get the weights of the old layer
            old_layer_weights = PartI.layers[old_layer_idx].get_weights()

            # Set the weights of the new layer to the old weights
            new_layer.set_weights(old_layer_weights)
            count = count + 1
            print(count)

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
