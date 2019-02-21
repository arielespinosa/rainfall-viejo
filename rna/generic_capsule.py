'''

'''
from __future__ import print_function

import os
import numpy as np
from utils.custom_losses import *
from keras_models import create_capsule, run_experiment
from utils.maps_utils import plot_performance, plt
from keras.optimizers import Adam, RMSprop, Adadelta
from utils.utils import StandardScaler, MinMaxScaler
from utils.utils import normalize, to_binary, to_categorical, get_categories
from utils.utils import loadmat, savemat, train_test_split
from utils.utils import DataSet

if __name__ == "__main__":

    # network arquitectures
    network = {
        'num_capsules': 20,
        'dim_capsules': 5,
        'routings': 3,
        'activation': 'squash',
        'kernel_initializer': 'glorot_uniform',
        'optimizer': Adadelta(), #RMSprop(), #Adam(lr=0.001),
        'loss': margin_loss, #'mse',#'categorical_crossentropy', #
        'share_weights': True,
        'for_regression': False
    }

    parameters = {
        'num_inputs': 1,
        'num_classes': 1,
        'num_features': 1,
        'batch_size': 128,
        'epochs': 25,
        'test_split': 0.1,
        'val_split': 0.1,
        'test_shuffle': False,
        'train_shuffle': True
    }


    str_format = [
        "{}, forecast rmse {:2.4f}, neural-net rmse: {:2.4f}",
        "{}, forecast corr {:2.4f}, neural-net corr: {:2.4f}"
    ]


    # # Create the dataset
    # path = 'test_data/train_data/'

    # print('Loading data...')
    # x_data = loadmat(path + 'x_samples', squeeze_me=True)
    # y_data = loadmat(path + 'y_samples', squeeze_me=True)

    # #
    # for key in ['__header__', '__version__', '__globals__']:
    #     _ = x_data.pop(key)
    #     _ = y_data.pop(key)

    # # join all data
    # X = []
    # Y = []
    # for (stn, y), x in zip(y_data.iteritems(), x_data.itervalues()):
    #     if not np.isnan(y).any():
    #         X.append(x)
    #         Y.append(y)

    # X = np.concatenate(X, axis=0)
    # y = np.concatenate(Y, axis=0)

    file_name = 't2m_rain_mslp_rh2m_d01_00.mat'

    print('Loading data...')
    data = loadmat(path + file_name, squeeze_me=True)

    #
    for key in ['__header__', '__version__', '__globals__']:
        _ = data.pop(key)

    # join all data
    dataset = []
    for stn, values in data.iteritems():
        dataset.append(values)
    if len(dataset) > 0:
        values = np.concatenate(dataset, axis=0)
    else:
        exit()

    data = DataSet(values)

    # normalize x data with mean and standard deviation
    x_scaler = MinMaxScaler(feature_range=(0.0001, 1))
    y_scaler = MinMaxScaler(feature_range=(0.0001, 1))

    data_shape = X.shape
    X = x_scaler.fit_transform(X.reshape( (data_shape[0], -1) )).reshape(data_shape)


    # convert y data to categorical
    parameters['num_inputs'] = data_shape[1]
    parameters['num_features'] = data_shape[-1]

    # y = y_scaler.fit_transform(y[:, 1:2])
    # y = to_binary(y[:, 1], threshold=0.1); print( 1e2 * len(y[y==1]) / len(y) )
    _, y = get_categories(y[:, 1], classes=20)
    y = to_categorical(y)
    parameters['num_classes'] = y.shape[-1]

    # split data for validation and testing
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=parameters['test_split'],
        shuffle=parameters['test_shuffle'])

    print(y_test, x_test[:,0].max())

    # convert data to categorical: (see treshholds later)
    print('Convert class vector to binary class matrix '
          '(for use with categorical_crossentropy)')

    # fitting model
    print('----------------------------------------------------------------')
    print('inputs: integer tensor of shape (samples, predictors, features) ')
    print('inputs_train shape:', x_train.shape)
    print('inputs_test shape:', x_test.shape)
    print('----------------------------------------------------------------')
    print('outputs: float, tensor of shape (samples, classes)')
    print('outputs_train shape:', y_train.shape)
    print('outputs_test shape:', y_test.shape)
    print('----------------------------------------------------------------')

    #Nework architecture:
    print('\nBuilding a capsule neural network ...')
    model = create_capsule(
        num_classes=parameters['num_classes'],
        num_inputs=parameters['num_inputs'],
        num_features=parameters['num_features'],
        **network)

    history_model = run_experiment(
        model, x_train, y_train, parameters, save_model=True)

    score_model = model.evaluate(
        x_test, y_test, batch_size=len(x_test), verbose=1)

    print('\nNetwork results')
    print('Test score:', score_model[0])
    print('Test accuracy:', score_model[1])

    fig = plot_performance(parameters['epochs'], history_model)
    fig.savefig('outputs/imgs/' + 'categorical_mpl_performance.png')
    plt.close(fig)

    print("Making some predictions")
    y_pred = model.predict(x_test)
    print('Done !')

    # get predicted categories
    y_test = y_test.argmax(axis=-1)
    y_pred = y_pred.argmax(axis=-1)

    print(y_test.max(), y_pred.max(), (y_pred == 0).all())

    #==========================================================================
    data_shape = x_test.shape
    x_test = x_scaler.inverse_transform(x_test.reshape((data_shape[0], -1))).reshape(data_shape)
    # y_test = y_scaler.inverse_transform(y_test).squeeze()
    # y_pred = y_scaler.inverse_transform(y_pred).squeeze()

    # get the closest points average as forecast value
    x_plot = x_test[:, 0].mean(axis=-1)

    print(x_plot.shape, y_test.shape)

    raw_rmse = np.sqrt(mean_squared_error(y_test, x_plot))
    ada_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    raw_corr = explained_variance_score(y_test, x_plot)
    ada_corr = explained_variance_score(y_test, y_pred)

    print(str_format[0].format(stn, raw_rmse, ada_rmse))
    print(str_format[1].format(stn, raw_corr, ada_corr), '\n')

    slope, intcp, r_value, _, _ = linregress(y_test, x_plot)

    # look at the results
    str_linr = 'Linear Regression ' + r'r2' + ': %.3f' % (r_value)
    str_fcst = 'SPNOA forecast (rmse: %.3f, corr: %.3f)' % (raw_rmse, raw_corr)
    str_pred = 'MLP Regression (rmse: %.3f, corr: %.3f)' % (ada_rmse, ada_corr)

    plt.figure()
    plt.scatter(y_test, x_plot, c='b', s=2.5, label=str_fcst, zorder=1)
    plt.scatter(y_test, y_pred, c='r', s=2.5, label=str_pred, zorder=1)
    plt.plot(y_test, intcp + slope * y_test, '-k', label=str_linr)
    plt.plot(y_test, y_test, '--k', linewidth=0.25)

    plt.xlabel('observations')
    plt.ylabel('predictions')
    plt.legend()
    plt.show()
    #==========================================================================
