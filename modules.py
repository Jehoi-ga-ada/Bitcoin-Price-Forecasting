import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras as k
import json
import numpy as np
import pandas as pd
sns.set_theme()

def convert_np_to_native(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.generic):
        return data.item()
    elif isinstance(data, dict):
        return {key: convert_np_to_native(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_np_to_native(item) for item in data]
    else:
        return data

def save_dict_to_json(data, filename='data.json'):
    native_data = convert_np_to_native(data)
    with open(filename, 'w') as json_file:
        json.dump(native_data, json_file, indent=4)
        
def train_model(model, model_name, loss_fn, optimizer, metrics, X_train, y_train, batch_size=32, epochs=10, shuffle=True, verbose=2, callbacks=None ,validation_data=None):
    tf.random.set_seed(42)
    model.compile(
        loss=loss_fn,
        optimizer=optimizer,
        metrics = metrics
    )
    history = model.fit(
        x=X_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        shuffle=shuffle,
        verbose=verbose,
        callbacks=callbacks
    )
    
    plot_loss_and_prediction(model, X_train, y_train, history, validation_data, model_name)
    
    return history

def plot_loss_and_prediction(model, X_train, y_train, history=None, validation_data=None, model_name=''):
    y_train_pred = tf.squeeze(model.predict(X_train, verbose=0))
    for i in range(y_train.shape[1]):          
        plt.figure(figsize=(15, 3))
        if history!=None and validation_data!=None:
            plt.subplot(1, 3, 1)
        plt.title(f'{model_name} Predictions H+{i}')
        plot_time_series(y_train_pred[:, i], format='-', label='Prediction')
        plot_time_series(y_train[:, i], format='-', label='Actual Price')
        if validation_data!=None:
            y_val_pred = tf.squeeze(model.predict(validation_data[0], verbose=0))
            plt.subplot(1, 3, 2)
            plot_time_series(y_val_pred[:, i], format='-', label='Prediction')
            plot_time_series(validation_data[1][:, i], format='-', label='Actual Price')
        if history!=None:
            plt.subplot(1, 3, 3)
            plot_train_val_loss(history, model_name)

def plot_train_val_loss(history, model_name):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
def plot_time_series(values, format='.', start=0, end=None, label=None):
    plt.plot(np.squeeze(values)[start:end], format, label=label)
    plt.xlabel('Time')
    plt.ylabel('BTC Price')
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)
    
def mean_absolute_symmetric_error(y_true, y_pred):
    """
    Implement MASE (assuming no seasonality of data)
    """
    mae = tf.reduce_mean(tf.abs(y_true-y_pred))
    
    mae_naive_no_season = tf.reduce_mean(tf.abs(y_true[1:] - y_true[:-1]))
    
    return mae / mae_naive_no_season

def evaluate_preds(y_true, y_pred):
    y_true = tf.squeeze(tf.cast(y_true, dtype=tf.float32))
    y_pred = tf.squeeze(tf.cast(y_pred, dtype=tf.float32))
    
    mae = k.metrics.mean_absolute_error(y_true, y_pred)
    mse = k.metrics.mean_squared_error(y_true, y_pred)
    rmse = tf.sqrt(mse)
    mape = k.metrics.mean_absolute_percentage_error(y_true, y_pred)
    mase = mean_absolute_symmetric_error(y_true, y_pred)
    
    return {
        'mae': tf.reduce_mean(mae).numpy(),
        'mse': tf.reduce_mean(mse).numpy(),
        'rmse': tf.reduce_mean(rmse).numpy(),
        'mape': tf.reduce_mean(mape).numpy(),
        'mase': tf.reduce_mean(mase).numpy(),
    }
    
def get_results_table(dir):
    file_list = [os.path.join(dir, file).replace("\\", "/") for file in os.listdir(dir) if file.endswith('.json')]
    model_name = [file.replace('.json', '') for file in os.listdir(dir) if file.endswith('.json')]

    results = pd.DataFrame(columns=['mae', 'mse', 'rmse', 'mape', 'mase', 'model'])
    for i, path in enumerate(file_list):   
        with open(path) as f:
            data = json.load(f)
        df = pd.DataFrame.from_dict([data])
        df['model'] = model_name[i]
        if i == 0:
            results = df.copy()
        else:
            results = pd.concat([results, df], ignore_index=True)
    
    return results