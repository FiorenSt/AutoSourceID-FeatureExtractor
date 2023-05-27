import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
from keras import backend as K

def load_and_predict(model_name, patches):

    # Load the model
    def gauss_loss(y_true, y_pred):  ## PREDICTS MEAN AND LOG(VAR)
        pred_mean = y_pred[:, 3:6]
        pred_var = K.exp(y_pred[:, 0:3]) + 1e-6
        return tf.reduce_mean(K.log(pred_var) + (((y_true - pred_mean) ** 2) / pred_var))

    def mre(y_true, y_pred, eps=1e-6):
        return K.mean(K.abs(y_true - y_pred) / (K.abs(y_true) + eps))

    model = load_model(os.path.join('../Models', model_name),
                       custom_objects={'gauss_loss': gauss_loss, 'mre': mre})  ##FOCAL FOR NOW

    # Make predictions
    predictions = model.predict(patches)

    # TRANSFORM BACK
    mean = predictions[:, 3:6]
    sigmas = np.sqrt(np.exp(predictions[:, 0:3]))

    mean_flux = 0.5937121510505676
    sd_flux = 0.029028689488768578
    lambda_val = -1.425486982494622

    sigmas[:, 2] = (((lambda_val * sd_flux * mean[:, 2]) + (lambda_val * mean_flux) + 1) ** (
                (1 - lambda_val) / lambda_val)) * sd_flux * sigmas[:, 2]
    mean[:, 2] = (lambda_val * (sd_flux * mean[:, 2] + mean_flux) + 1) ** (1 / lambda_val)

    ##ERROR PROPAGATION

    sigmas[:, 0:2] = sigmas[:, 0:2] * 0.866
    mean[:, 0:2] = (mean[:, 0:2] * 0.866) + 16

    return np.hstack((mean,sigmas))