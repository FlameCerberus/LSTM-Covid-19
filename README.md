# LSTM-Covid-19

## Project Description
The LSTM-Covid-19 project aims to forecast future trends of COVID-19 cases using Long Short-Term Memory (LSTM) neural network architecture. This predictive model leverages historical data to generate forecasts, aiding in understanding and potentially mitigating the spread of the virus.

## Model Used
The model employs LSTM architecture, a type of recurrent neural network (RNN), known for its ability to learn long-term dependencies in sequential data. LSTM networks are particularly effective for time series forecasting tasks due to their capacity to capture patterns and trends over time.

![Model Architecture](https://github.com/FlameCerberus/LSTM-Covid-19/assets/96816249/b9558814-7dc6-40a7-b065-68775412d2ad)

## Steps Taken
1. Data Collection: Gathered COVID-19 data from reliable sources such as from the Ministry of Health - Malaysia from a github website (link in referrence)
2. Data Cleaning: Preprocessed the data to handle missing values, outliers, and inconsistencies. This step ensures the quality and reliability of the dataset for training. The data had some strings and null values in the new_cases where it required some data cleaning and restructure
3. Feature Engineering: Feature engineering was not required for this dataset as the dataset was straightforward and would not affect the training and performance of the model
4. Model Training: Utilized the LSTM architecture to train the forecasting model. The dataset was split into training, test and validation sets to evaluate the model's performance.
5. Hyperparameter Tuning: Fine-tuned the model's hyperparameters, such as the learning rate to be lower and number of LSTM layers and regularization techniques used, to optimize performance and generalization.
6. Model Evaluation: Evaluated the trained model using graphs of validation and accruacy loss to determine whether the model is overfitting or underfitting and generated the RMSE percentage value which had less than 5% error which shown good results

![Train_Val_loss](https://github.com/FlameCerberus/LSTM-Covid-19/assets/96816249/e0387364-62c1-42b6-b4b8-287dc1b91901)

8. Forecast Generation: Generated forecasts for future COVID-19 cases using the trained LSTM model. Visualizations such as time series plots and prediction intervals were created to visualize the forecasted trends against its own dataset to see how well and close it performs to the original dataset

![Graph Predicted vs Actual](https://github.com/FlameCerberus/LSTM-Covid-19/assets/96816249/83e1e7cf-5294-49f6-8504-94086bf30573)

## Conclusion
The LSTM-Covid-19 project demonstrates the potential of LSTM neural networks in forecasting future trends of COVID-19 cases. By leveraging historical data and advanced machine learning techniques, the model provides valuable insights for policymakers, healthcare professionals, and the general public to better understand and respond to the ongoing pandemic.

## Reference
- [Ministry of Health - Malaysia COVID-19 Data](https://github.com/MoH-Malaysia/covid19-public)
- [Tensorflow tutorial on time series](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [LSTM in Keras medium](https://medium.com/plain-simple-software/long-short-term-memory-in-keras-acdf61c056da)
