# %%
#1. Import Necessary Libraries
from time_series_helper import WindowGenerator
import os
import datetime
import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import mean_absolute_error

# %%
#2. Loading Data
path = os.getcwd()
path_train = os.path.join(path, 'cases_malaysia_train.csv')
path_test = os.path.join(path, 'cases_malaysia_test.csv')
path_log = os.path.join(os.getcwd(), 'Log')

train_df = pd.read_csv(path_train)
test_df = pd.read_csv(path_test)

# %%
#3. Data Inspection
#(A) for Train Data
print("NaN Values")
print(train_df.isna().sum())
print("Duplicates") 
print(train_df.duplicated().sum())
print("Description")
print(train_df.describe().transpose())

#(B) For Test Data
print("NaN Values")
print(test_df.isna().sum())
print("Duplicates") 
print(test_df.duplicated().sum())
print("Description")
print(test_df.describe().transpose())

# %%
#4. Data Cleaning
test_df['cases_new'] = test_df['cases_new'].fillna(0)
test_df['cases_new'] = pd.to_numeric(test_df['cases_new'])
test_df['cases_new'] = test_df['cases_new'].astype(float)
test_df['cases_new'] = test_df['cases_new'].replace('?', '0')



# %%
train_df['cases_new'] = train_df['cases_new'].fillna(0)
train_df['cases_new'] = pd.to_numeric(train_df['cases_new'], errors='coerce').fillna(0)
train_df['cases_new'] = train_df['cases_new'].astype(float)



# %%
train_df.fillna(0, inplace=True)
train_df.drop(train_df[(train_df['cases_new'] == 0) & (train_df.index > 75)].index, inplace=True)
train_df.to_csv('train.csv', index=False)


test_df = test_df[test_df['cases_new'] != 0]
test_df.to_csv('ca.csv', index=False)


# %%
test_df.describe().transpose()

# %%
train_df.describe().transpose()

# %%
# Isolate out the Date Column
date_train = pd.to_datetime(train_df.pop('date'), format ='%d/%m/%Y')
date_test = pd.to_datetime(test_df.pop('date'), format ='%d/%m/%Y')

print(train_df.columns)

# %%
date_train.info()
date_train.head()

# %%
date_test.info()
date_test.head()

# %%
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_minmax = scaler.fit_transform(train_df)
train_df = pd.DataFrame(data_minmax,columns = train_df.columns, index = train_df.index)

data_minmax = scaler.fit_transform(test_df)
test_df = pd.DataFrame(data_minmax,columns = test_df.columns, index = test_df.index)

#5. Train validation test split
column_indices = {name: i for i, name in enumerate(train_df.columns)}

n = len(train_df)
train_df_a = train_df[0:int(n*0.8)]
val_df = train_df[int(n*0.8):]

num_features = train_df.shape[1]


# %%
#6. Create The Data Windows
# #(A) Single-Step Window
# single_window = WindowGenerator(input_width = 30, label_width= 30, shift = 1, train_df = train_df_a,val_df = val_df, test_df = test_df, label_columns = ["cases_new"])
#(B) Multi-Step Window
multi_window = WindowGenerator(input_width= 30, label_width= 30, shift = 30,train_df = train_df_a,val_df = val_df, test_df = test_df, label_columns = ["cases_new"])
# print(single_window)
print(multi_window)

# %%
# single_window.plot()

# %%
# 7. Model Development
#(A) Single-Step Model
single_model = tf.keras.Sequential()
single_model.add(tf.keras.layers.LSTM(32,return_sequences=True))
single_model.add(tf.keras.layers.Dense(1))

# %%
# Define a function to compile and train the model
MAX_EPOCHS = 20
def compile_and_fit(model, window, epochs1 = None, patience=20):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(learning_rate = 0.00001),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=epochs1,
                      validation_data=window.val,
                      callbacks=[early_stopping, tensorboard_callback])
  return history

# %%
#Train The Model
history = compile_and_fit(model = single_model, window = single_window, epochs1 = 30)

# %%
# Output Model for single prediction
single_window.plot(single_model)
plt.show()

# %%
multi_model = tf.keras.Sequential()
multi_model.add(tf.keras.layers.LSTM(64,return_sequences=True))
multi_model.add(tf.keras.layers.Dropout(0.1))
multi_model.add(tf.keras.layers.LSTM(32,return_sequences=False))
multi_model.add(tf.keras.layers.Dropout(0.1))
multi_model.add(tf.keras.layers.Dense(100))
multi_model.add(tf.keras.layers.Dense(30*1))
multi_model.add(tf.keras.layers.Reshape([30,1]))


log_dir = os.path.join(path_log, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

# %%
history_multi = compile_and_fit(model = multi_model, window=multi_window, epochs1 = 200, patience=20)


# %%
multi_window.plot(multi_model)
plt.show()

# %%
multi_model.summary()

# %%
# plot training history
plt.plot(history_multi.history['loss'], label='train')
plt.plot(history_multi.history['val_loss'], label='validation')
plt.legend()
plt.show()

# %%
# Specify the index of the step you want to compare
# step_index = 0

# # Generate predictions for the specified step
# predictions_step = predictions[:, step_index]
prediction = np.mean(predictions, axis=0)

# Plot the specified step's predictions against the actual values
plt.figure()
plt.plot(predictions, label='Prediction')
plt.plot(actual_values, label='Actual')
plt.legend()
plt.show()

mean_absolute_error(predictions, actual_values)

print('\n Mean absolute percentage error:', 
mean_absolute_error(predictions, actual_values)/sum(abs(predictions))*100)


