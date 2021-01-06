import pandas as pd
import xarray as xr
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Input, Dropout, Flatten, Reshape
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras import metrics
from keras.models import load_model  #  To load model from disk
import uuid


def read_ibtracs(datapath):

    ibtracs = pd.read_csv(datapath + 'COUNTS/IBTrACs/IBTrACS_1979_2008/IBTrACS_NH_NATL.txt')
    ibtracs = ibtracs.transpose()
    ibtracs = ibtracs[:-1]
    ibtracs.index = pd.to_datetime(ibtracs.index, format='%Y')
    ibtracs = ibtracs.to_xarray()[0].rename(index='time')
    return ibtracs


def read_gpi(datapath):

    gpi_path = 'GPI/PRESENT/time_series/NATL/ERA5/1/ERA5/1/'
    filenames = os.listdir(datapath + gpi_path)
    ds_list = []
    for filename in filenames:
        ds = xr.open_dataset(datapath + gpi_path + filename)
        # ds = ds.data_vars.variables.keys()
        ds_list.append(ds)

    ds = xr.merge(ds_list)
    ds = ds.drop(['latitude_longitude', 'longitude_bnds', 'latitude_bnds'])
    da = ds.to_array()
    return da


def common_index(list1, list2):
    return [element for element in list1 if element in list2]


def NN_mlp(x_train, x_val, y_train, y_val, epochs, batch_size):
    """
    MLP Keras Neural Network model
    author: Thomas Martin

    :param x: numpy array, exploratory input data with the shape Samples/Time Steps/ Features
    :param x2: numpy array, auxilary input
    :param y: numpy array, response data
    :param model_path: model outpath
    :return: model history
    """
    print('NN_MLP')
    cwd = os.getcwd()
    cwd = cwd + '/.kerasmodels'
    if not os.path.exists(cwd):
        os.makedirs(cwd)
    callback_model_path = cwd + '/.model_{}.h5'.format(
    str(uuid.uuid4()))  # Internal use to save model by callbacks
    main_input = Input(shape=x_train.shape[1:], name="main_input")
    dense1 = Dense(50, activation="relu")(main_input)
    drop1 = Dropout(0.1)(dense1)
    dense2 = Dense(50, activation="relu")(drop1)
    drop2 = Dropout(0.1)(dense2)
    dense3 = Dense(50, activation="relu")(drop2)
    out_dense = Dense(y_train.shape[1], activation='sigmoid')(dense3)
    callbacks = [ModelCheckpoint(filepath=callback_model_path, monitor="val_mean_absolute_error",
                                 save_best_only=True)]
    model = Model(inputs=[main_input], outputs=out_dense)
    model.compile(loss='mean_absolute_error', metrics=[metrics.mae, metrics.mse], optimizer='adam')
    print('Start fitting')
    history = model.fit([x_train.values], y_train.values,
                        epochs=epochs,
                        validation_data=(x_val.values, y_val.values),
                        verbose=2,
                        shuffle=False,
                        batch_size=batch_size, callbacks=callbacks)
    print('End fitting')
    model = load_model(callback_model_path)
    os.remove(callback_model_path)
    return model, history


# ---- Reading data ---- #

thisfile_path = os.path.dirname(os.path.realpath(__file__))
datapath = os.path.dirname(thisfile_path) + '/data/'
da_y = read_ibtracs(datapath)
da_x = read_gpi(datapath)
x_variables = ['potential_intensity_component_of_gpi',
               'wind_shear_component_of_gpi',
               'rh_component_of_gpi',
               'vorticity_term_of_gpi',
               'genesis_potential_index__gpi__for_tc'
               ]

#  Arrays for ML usually have the shape [samples, features]
da_x = da_x.transpose('time', 'variable').sel(variable=x_variables).resample(time='Ys').mean('time')
da_x_original = da_x.copy()
x_variables.remove('genesis_potential_index__gpi__for_tc')
da_x = da_x.sel(variable=x_variables)
da_y = da_y.expand_dims('variable')
da_y = da_y.transpose('time', 'variable')
da_x = da_x

# ---- Preprocessing ---- #

#  Scaling 0 - 1
scaler_x = MinMaxScaler().fit(da_x.values)
scaler_y = MinMaxScaler().fit(da_y.values)
da_x_scaled = da_x.copy(data=scaler_x.transform(da_x.values))
da_y_scaled = da_y.copy(data=scaler_y.transform(da_y.values))


#  Aligning the two datasets along time
da_x_scaled = da_x_scaled.sel(variable=x_variables)
idxs = common_index(da_x_scaled.time.values, da_y_scaled.time.values)
da_x_scaled= da_x_scaled.sel(time=idxs)
da_y_scaled= da_y_scaled.sel(time=idxs)

da_x_scaled_train = da_x_scaled.sel(time=slice(None, '2000'))
da_y_scaled_train = da_y_scaled.sel(time=slice(None, '2000'))
da_x_scaled_val = da_x_scaled.sel(time=slice('2000', None))
da_y_scaled_val = da_y_scaled.sel(time=slice('2000', None))


#  Plotting to check that it is all as expected
da_x_scaled_train.plot.line(x='time')
plt.plot(da_y_scaled_train.time.values, da_y_scaled_train.values)
plt.legend(x_variables + ['ibtracs'])
plt.gca().set_prop_cycle(None)
da_x_scaled_val.plot.line(x='time', linestyle='--', add_legend=False)
plt.plot(da_y_scaled_val.time.values, da_y_scaled_val.values, linestyle='--')
plt.title('Obs. storms and env. scaled')
plt.show()

# ---- Model setup ---- #

model, history = NN_mlp(da_x_scaled_train, da_x_scaled_val, da_y_scaled_train,
                        da_y_scaled_val, epochs=100, batch_size=100)
#  Plot train error
plt.plot(history.history['val_mean_absolute_error'])
plt.xlabel('Training epoch')
plt.ylabel('Mean absolute error')
plt.show()

prediction_val = model.predict(da_x_scaled_val.values)
prediction_val = scaler_y.inverse_transform(prediction_val)
prediction_val = da_y_scaled_val.copy(data=prediction_val)

prediction_train = model.predict(da_x_scaled_train.values)
prediction_train = scaler_y.inverse_transform(prediction_train)
prediction_train = da_y_scaled_train.copy(data=prediction_train)

# Plotting precition x gpi

fig, ax = plt.subplots(1, 1)
ax1 = ax.twinx()
ax.plot(da_y.time.values, da_y.values, linewidth=2, color='k')
ax.plot(prediction_train.time.values, prediction_train.values, linestyle='--', color='blue')
ax.plot(prediction_val.time.values, prediction_val.values, linestyle='--', color='red')
ax1.plot(da_x_original.time.values, da_x_original.sel(variable='genesis_potential_index__gpi__for_tc').values,
         color='green', linestyle='--')
ax.legend(['Obs.', 'MLP_train', 'MLP_validation'])
ax1.legend(['GPI'])
ax.set_ylabel('Number of storms')
ax1.set_ylabel('GPI')
plt.title('Storms: Neural net. vs GPI')
plt.show()

xr.corr(prediction_val, da_y, dim='time')
xr.corr(da_x_original.sel(variable='genesis_potential_index__gpi__for_tc'), da_y, dim='time')
error_MLP_train = prediction_train - da_y

#  Interpreting
import numpy as np
da_x_test = da_x_scaled.copy(data=np.zeros(da_x_scaled.shape) + .5)
da_x_test.values[:, -2] = np.linspace(0, 1, da_x_test.shape[0])

prediction_test = model.predict(da_x_test.values)
da_x_test = da_x_test.copy(data=scaler_x.inverse_transform(da_x_test.values))
plt.scatter(x=da_x_test.values[:, -2], y=scaler_y.inverse_transform(prediction_test))
plt.show()

plt.plot(da_x_test.values[:, 0],da_x_test.values[:, 0]**3 )
