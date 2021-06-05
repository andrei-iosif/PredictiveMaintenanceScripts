from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint


def create_mlp_model(input_dim, hidden_layer_sizes, activation='relu', dropout=0, output_weights_file=None):
    model = Sequential()
    model.add(Dense(hidden_layer_sizes[0],
                    input_dim=input_dim,
                    kernel_initializer='random_normal',
                    activation=activation))
    if dropout > 0:
        model.add(Dropout(dropout))

    for layer_size in hidden_layer_sizes[1:]:
        model.add(Dense(layer_size,
                        kernel_initializer='random_normal',
                        activation=activation))
        if dropout > 0:
            model.add(Dropout(dropout))

    model.add(Dense(1, kernel_initializer='random_normal'))

    model.compile(loss='mean_squared_error', optimizer='adam')

    if output_weights_file is not None:
        model.save_weights(output_weights_file)
    return model


def get_callbacks(model_path):
    """
    Create callback objects for model training.
    :param model_path: path to save trained model
    :return: list of callbacks
    """
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    mc = ModelCheckpoint(model_path, monitor='val_loss', mode='min', verbose=1,
                         save_best_only=True)
    return [es, mc]
