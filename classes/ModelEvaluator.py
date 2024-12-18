import gym
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler
from sklearn.metrics import mean_squared_error

from pgmpy.models import BayesianNetwork
from pgmpy.inference import BeliefPropagation
from pgmpy.estimators import MaximumLikelihoodEstimator

import os
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
 
tf.get_logger().setLevel('ERROR')

class ModelEvaluator:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def lstm_prediction(self, data, look_back=60):
        data_scaled = self.scaler.fit_transform(data.values.reshape(-1, 1))
        X, y = [], []
        for i in range(look_back, len(data_scaled)):
            X.append(data_scaled[i - look_back:i, 0])
            y.append(data_scaled[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.compile(optimizer="adam", loss="mean_squared_error")
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

        predictions = model.predict(X_test)
        predictions = self.scaler.inverse_transform(predictions)
        y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))

        plt.figure(figsize=(10, 5))
        plt.plot(y_test_actual, label="Actual")
        plt.plot(predictions, label="Predicted")
        plt.title("LSTM Prediction vs. Actual")
        plt.xlabel("Time Steps")
        plt.ylabel("Stock Price")
        plt.legend()
        plt.show()

        return model, predictions

    def build_bayesian_network(self, data):
        discretizer = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="uniform")
        data_discrete = data.copy()

        data_discrete.ffill(inplace=True)
        data_discrete.bfill(inplace=True)
        data_discrete.ffill(inplace=True)

        columns_to_discretize = ["rsi_14", "close", "macd"]
        for col in columns_to_discretize:
            if data_discrete[col].isnull().any():
                raise ValueError(f"NaN values found in column {col} after preprocessing.")
            data_discrete[col] = discretizer.fit_transform(data_discrete[[col]]).astype(int)

        model = BayesianNetwork([("rsi_14", "close"), ("macd", "close")])
        model.fit(data_discrete, estimator=MaximumLikelihoodEstimator)

        for cpd in model.get_cpds():
            cpd.normalize()
            if not np.isclose(cpd.values.sum(axis=0), 1, atol=0.01).all():
                logging.error(f"CPD for {cpd.variable} still not normalized properly.")
                return None

        inference = BeliefPropagation(model)
        try:
            query_result = inference.query(variables=["close"], evidence={"rsi_14": 2}, show_progress=False)
        except Exception as e:
            logging.error(f"Error during inference: {e}")
            return None
        logging.info(f"Bayesian Network Query Result: {query_result}")
        return query_result

    def reinforcement_learning(self, env_name="CartPole-v1", episodes=100):
        env = gym.make(env_name)
        rewards = []
        for episode in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = env.action_space.sample()
                next_state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
            rewards.append(total_reward)
        env.close()
        plt.plot(range(episodes), rewards)
        plt.title(f"Rewards Over {episodes} Episodes in {env_name}")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.show()
        return rewards

    def visualize_data(self, data, columns, title, xlabel, ylabel):
        data[columns].plot(figsize=(10, 5), title=title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def plot_correlation_matrix(self, data, title="Correlation Matrix"):
        correlation_matrix = data.corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title(title)
        plt.show()

    def backtest_lstm(self, predictions, actual_values):
        mse = mean_squared_error(actual_values, predictions)
        logging.info(f"LSTM Mean Squared Error: {mse}")

        actual_returns = actual_values[1:] - actual_values[:-1]
        predicted_returns = predictions[1:] - predictions[:-1]

        profitability = (predicted_returns * actual_returns).sum()
        logging.info(f"LSTM Profitability (Cumulative Returns): {profitability}")

        return mse, profitability

    def evaluate_models(self, stock_data):
        model, predictions = self.lstm_prediction(stock_data["close"])
        predictions = predictions.flatten()
        actual_values = stock_data["close"].values[-len(predictions):]
        lstm_mse, lstm_profit = self.backtest_lstm(predictions, actual_values)

        query_result = self.build_bayesian_network(stock_data)
        logging.info(f"Bayesian Network Query Result: {query_result}")

        rewards = self.reinforcement_learning(env_name="CartPole-v1", episodes=50)
        avg_reward = sum(rewards) / len(rewards)
        logging.info(f"Reinforcement Learning Average Reward: {avg_reward}")

        comparison_results = {
            "LSTM": {"MSE": lstm_mse, "Profitability": lstm_profit},
            "Bayesian Network": {"Example Query": query_result},
            "Reinforcement Learning": {"Average Reward": avg_reward},
        }

        return comparison_results

    def visualize_comparison(self, stock_results, crypto_results):
        # Metrics for stocks
        stock_lstm_metrics = stock_results["LSTM"]
        stock_mse = stock_lstm_metrics.get("MSE", 0)
        stock_profitability = stock_lstm_metrics.get("Profitability", 0)
        stock_rl_avg_reward = stock_results["Reinforcement Learning"].get("Average Reward", 0)

        # Metrics for crypto
        crypto_lstm_metrics = crypto_results["LSTM"]
        crypto_mse = crypto_lstm_metrics.get("MSE", 0)
        crypto_profitability = crypto_lstm_metrics.get("Profitability", 0)
        crypto_rl_avg_reward = crypto_results["Reinforcement Learning"].get("Average Reward", 0)

        # Stock Graph
        plt.figure(figsize=(12, 6))
        x = ["MSE", "Profitability", "Avg Reward"]
        stock_values = [stock_mse, stock_profitability, stock_rl_avg_reward]
        plt.bar(x, stock_values, color=["blue", "green", "orange"], alpha=0.7)
        plt.title("Stock Model Comparison")
        plt.ylabel("Values")
        plt.ylim([0, max(stock_values) * 1.1])  # Adjust y-axis for visibility
        plt.show()

        # Crypto Graph
        plt.figure(figsize=(12, 6))
        crypto_values = [crypto_mse, crypto_profitability, crypto_rl_avg_reward]
        plt.bar(x, crypto_values, color=["blue", "green", "orange"], alpha=0.7)
        plt.title("Crypto Model Comparison")
        plt.ylabel("Values")
        plt.ylim([0, max(crypto_values) * 1.1])  # Adjust y-axis for visibility
        plt.show()
