from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout
from tensorflow.python.ops import nn
from tensorflow.python.keras.utils import tf_utils
from tensorflow.keras.initializers import TruncatedNormal 
# from tensorflow.keras import backend as K

class MCDropout(Dropout):
    """
    Applies Dropout to the input regardless of the training phase
    """

    def __init__(self, rate, **kwargs):
        super(MCDropout, self).__init__(rate, **kwargs)

    def call(self, inputs, training=None):
        return nn.dropout(inputs, rate=self.rate,
                        noise_shape=self._get_noise_shape(inputs),
                        seed=self.seed)

import logging
from types import MethodType

def make_model_stochastic(model, custom_objects={}):
    logger = logging.getLogger(__name__)

    if not isinstance(model, Model):
        raise ValueError(f'the `model` parameter needs to be an instance of {Model.__name__}, not {type(model)}')

    def _make_stochastic_model(self):
        if not hasattr(self, '_stochastic_model'):

            def replace_dropout_layers(layer):
                if layer['class_name'] == 'Dropout':
                    layer['class_name'] = 'MCDropout'
                return layer

            model_config = self.get_config()
            model_config['layers'] = list(map(replace_dropout_layers, model_config['layers']))
            self._stochastic_model = super(self.__class__, self).from_config(model_config, custom_objects={**custom_objects, 'MCDropout': MCDropout, 'TruncatedNormal':TruncatedNormal})

            self._stochastic_model.compile(
                optimizer = self.optimizer,
                loss = self.compiled_loss._losses,
                metrics = self.compiled_metrics._user_metrics,
                loss_weights = self.compiled_loss._loss_weights,
                # sample_weight_mode = self.sample_weight_mode,
                weighted_metrics = self.compiled_metrics._user_weighted_metrics
            )

            self._stochastic_model._weights_initially_synced = False
            self._synchronize_weights()

    def _synchronize_weights(self):
        for layer_num in range(len(self._stochastic_model.layers)):
            source_layer = self.layers[layer_num]
            target_layer = self._stochastic_model.layers[layer_num]
            # only copy weights for trainable layers
            # after inital syncing because then the others are fixed
            if source_layer.trainable or not self._stochastic_model._weights_initially_synced:
                logger.info(f'syncing weights from main model layer {layer_num} to stochastic copy')
                target_layer._trainable_weights = source_layer._trainable_weights

            # non-trainable weight only need to be synced once
            if not self._stochastic_model._weights_initially_synced:
                target_layer._non_trainable_weights = source_layer._non_trainable_weights

        self._stochastic_model._weights_initially_synced = True



    def predict_stochastic(self, *args, **kwargs):
        self._make_stochastic_model()

        return self._stochastic_model.predict(*args, **kwargs)

    model._make_stochastic_model = MethodType(_make_stochastic_model, model)
    model._synchronize_weights = MethodType(_synchronize_weights, model)
    model.predict_stochastic = MethodType(predict_stochastic, model)
