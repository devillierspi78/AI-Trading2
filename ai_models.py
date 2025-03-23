import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import mt5_interaction as mt5
from data_preparation import prepare_market_data, get_data
from quantum_model import quantum_portfolio_optimization
from utils import is_market_open
from tensorflow_probability.python.distributions import kullback_leibler

tfd = tfp.distributions

# Portfolio symbols for AI-Based Trading Strategies
TRADING_ASSETS = ["EURUSD", "XAUUSD", "USDJPY", "AUDUSD"]


class DenseVariational(tf.keras.layers.Layer):
    """Dense layer with random kernel and bias using variational inference."""

    def __init__(self, units, make_posterior_fn, make_prior_fn, kl_weight=None, kl_use_exact=False, activation=None, use_bias=True, activity_regularizer=None, **kwargs):
        super().__init__(activity_regularizer=tf.keras.regularizers.get(activity_regularizer), **kwargs)
        self.units = int(units)
        self._make_posterior_fn = make_posterior_fn
        self._make_prior_fn = make_prior_fn
        self._kl_divergence_fn = self._make_kl_divergence_penalty(kl_use_exact, weight=kl_weight)
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.supports_masking = False
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2)

    def build(self, input_shape):
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build DenseVariational layer with non-floating point dtype %s' % (dtype,))
        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError('The last dimension of the inputs to DenseVariational should be defined. Found None.')
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: last_dim})

        print("🔍 [DEBUG] Building DenseVariational Layer...")
        print(f"    - Input Shape: {input_shape}")
        print(f"    - Last Dimension: {last_dim}")
        print(f"    - Trainable: {self.trainable}")

        with tf.name_scope('posterior'):
            self._posterior = self._make_posterior_fn(last_dim * self.units, self.units if self.use_bias else 0, dtype, self.trainable, self.add_weight)

        with tf.name_scope('prior'):
            self._prior = self._make_prior_fn(last_dim * self.units, self.units if self.use_bias else 0, dtype, self.trainable, self.add_weight)

        self.built = True

    def call(self, inputs):
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        inputs = tf.cast(inputs, dtype, name='inputs')

        q = self._posterior(inputs)
        r = self._prior(inputs)
        self.add_loss(self._kl_divergence_fn(q, r))

        w = tf.convert_to_tensor(value=q.sample())
        prev_units = self.input_spec.axes[-1]

        if self.use_bias:
            split_sizes = [prev_units * self.units, self.units]
            kernel, bias = tf.split(w, split_sizes, axis=-1)
            kernel = tf.reshape(kernel, shape=[prev_units, self.units])
        else:
            kernel = tf.reshape(w, shape=[prev_units, self.units])
            bias = None

        outputs = tf.matmul(inputs, kernel)

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, bias)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if input_shape[-1] is None:
            raise ValueError(f'The innermost dimension of input_shape must be defined, but saw: {input_shape}')
        return input_shape[:-1].concatenate(self.units)

    def _make_kl_divergence_penalty(self, use_exact_kl=False, test_points_reduce_axis=(), test_points_fn=tf.convert_to_tensor, weight=None):
        if use_exact_kl:
            kl_divergence_fn = kullback_leibler.kl_divergence
        else:
            def kl_divergence_fn(distribution_a, distribution_b):
                z = test_points_fn(distribution_a.sample())
                return tf.reduce_mean(distribution_a.log_prob(z) - distribution_b.log_prob(z), axis=test_points_reduce_axis)

        def _fn(distribution_a, distribution_b):
            with tf.name_scope('kldivergence_loss'):
                kl = kl_divergence_fn(distribution_a, distribution_b)
                if weight is not None:
                    kl = tf.cast(weight, dtype=kl.dtype) * kl
                return tf.reduce_sum(kl, name='batch_total_kl_divergence')

        return _fn

def make_posterior_fn(kernel_size, bias_size=0, dtype=None):
    def posterior_fn(dtype, shape, name, trainable, add_variable_fn):
        print(f"shape: {shape}")  # Debugging statement
        if not isinstance(name, str):
            name = str(name)
        if not shape:
            raise ValueError("Shape must be a non-empty list or tuple of integers.")
        try:
            dtype = tf.as_dtype(dtype) if dtype else tf.float32  # Ensure dtype is valid
        except TypeError:
            print(f"Invalid dtype: {dtype}. Defaulting to tf.float32.")
            dtype = tf.float32
        shape = [shape] if isinstance(shape, int) else shape  # Ensure shape is a vector
        loc = add_variable_fn(
            name=name + '_loc',
            shape=shape,
            initializer=tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.1),
            dtype=dtype,
            trainable=trainable)
        rho = add_variable_fn(
            name=name + '_rho',
            shape=shape,
            initializer=tf.keras.initializers.Constant(np.log(np.expm1(1.0))),
            dtype=dtype,
            trainable=trainable)
        loc = tf.cast(loc, dtype)
        rho = tf.cast(rho, dtype)
        posterior = tfd.Normal(loc=loc, scale=tf.nn.softplus(rho))
        return lambda x: posterior  # Return a callable function that accepts input
    return posterior_fn

def make_prior_fn(kernel_size, bias_size=0, dtype=None):
    def prior_fn(dtype, shape, name, trainable, add_variable_fn):
        print(f"dtype: {dtype}")  # Debugging statement
        if not isinstance(name, str):
            name = str(name)
        try:
            dtype = tf.as_dtype(dtype) if dtype else tf.float32  # Ensure dtype is valid
        except TypeError:
            print(f"Invalid dtype: {dtype}. Defaulting to tf.float32.")
            dtype = tf.float32
        shape = [shape] if isinstance(shape, int) else shape  # Ensure shape is a vector
        prior = tfd.Independent(tfd.Normal(loc=tf.zeros(shape, dtype=dtype), scale=1.0), reinterpreted_batch_ndims=len(shape))
        return lambda x: prior  # Return a callable function that accepts input
    return prior_fn

def build_bayesian_model(input_dim):
    inputs = tf.keras.Input(shape=(input_dim,))
    
    # First Variational Dense Layer
    dense_variational1 = DenseVariational(
        units=64,
        make_posterior_fn=make_posterior_fn(64),
        make_prior_fn=make_prior_fn(64),
        kl_weight=1.0 / input_dim,
        activation='relu'
    )(inputs)
    
    # Second Variational Dense Layer
    dense_variational2 = DenseVariational(
        units=32,
        make_posterior_fn=make_posterior_fn(32),
        make_prior_fn=make_prior_fn(32),
        kl_weight=1.0 / input_dim,
        activation='relu'
    )(dense_variational1)
    
    # Output Layer
    outputs = tf.keras.layers.Dense(1)(dense_variational2)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.MeanSquaredError())
    
    return model

def train_and_forecast_bayesian_model():
    if not is_market_open():
        print("Market is closed. Skipping training and forecasting.")
        return
    for symbol in ["EURUSD", "XAUUSD", "USDJPY", "AUDUSD"]:
        X, y, scaler = prepare_market_data(symbol, 50)
        if X is None or y is None:
            continue
        input_dim = X.shape[1]
        model = build_bayesian_model(input_dim)
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)
        future_price_pred = model(X[-1:])
        future_price = future_price_pred.mean().numpy()[0]

        trade_direction = "BUY" if future_price > X[-1][-1][0] else "SELL"
        if not mt5.order_exists(symbol, trade_direction, 123456):
            order_result = mt5.place_order(order_type=trade_direction, symbol=symbol, volume=0.1, stop_loss=0.0, take_profit=0.0, comment="AI Trade BNN", magic_number=123456)
            if order_result:
                pass
            else:
                pass
        else:
            pass

def execute_ai_trade():
    optimized_weights = quantum_portfolio_optimization(["EURUSD", "XAUUSD", "USDJPY", "AUDUSD"])
    for symbol, weight in zip(["EURUSD", "XAUUSD", "USDJPY", "AUDUSD"], optimized_weights):
        trade_direction = "BUY" if weight > 0 else "SELL"
        if not mt5.order_exists(symbol, trade_direction, 654321):
            order_result = mt5.place_order(order_type=trade_direction, symbol=symbol, volume=0.1, stop_loss=0.0, take_profit=0.0, comment="AI Trade Quantum", magic_number=654321)
            if order_result:
                pass
            else:
                pass
        else:
            pass

def run_backtesting():
    print("Running backtesting...")
    # Backtest BNN
    all_bnn_trades = []
    for symbol in TRADING_ASSETS:
        bnn_trades = backtest_bnn(symbol)
        if bnn_trades:
            all_bnn_trades.extend(bnn_trades)
    
    bnn_results, bnn_total_profit_loss = evaluate_performance(all_bnn_trades)
    print("BNN Results:")
    print(bnn_results)
    print(f"BNN Total Profit/Loss: {bnn_total_profit_loss}")
    
    # Backtest Quantum
    quantum_trades = backtest_quantum()
    quantum_results, quantum_total_profit_loss = evaluate_performance(quantum_trades)
    print("Quantum Results:")
    print(quantum_results)
    print(f"Quantum Total Profit/Loss: {quantum_total_profit_loss}")


def backtest_bnn(symbol):
    X, y, scaler = prepare_market_data(symbol, 50)
    if X is None or y is None:
        return None
    
    input_dim = X.shape[1]
    model = build_bayesian_model(input_dim)
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)
    
    predicted_prices = model(X).mean().numpy().flatten()
    actual_prices = y.flatten()
    
    trades = []
    for i in range(len(predicted_prices)):
        trade_direction = "BUY" if predicted_prices[i] > X[i][-1][0] else "SELL"
        trades.append((symbol, trade_direction, actual_prices[i], predicted_prices[i]))
    
    return trades

def backtest_quantum():
    weights = quantum_portfolio_optimization(["EURUSD", "XAUUSD", "USDJPY", "AUDUSD"])
    
    trades = []
    for symbol, weight in zip(["EURUSD", "XAUUSD", "USDJPY", "AUDUSD"], weights):
        trade_direction = "BUY" if weight > 0 else "SELL"
        price = get_data(symbol, mt5.TIMEFRAME_M5, bars=1)['close'].iloc[-1]
        trades.append((symbol, trade_direction, price, weight))
    
    return trades

def evaluate_performance(trades):
    """Evaluate the performance of the trades."""
    results = []
    for trade in trades:
        symbol, direction, actual_price, predicted_value = trade
        profit_loss = actual_price - predicted_value if direction == "SELL" else predicted_value - actual_price
        results.append((symbol, direction, actual_price, predicted_value, profit_loss))
    
    df_results = pd.DataFrame(results, columns=["Symbol", "Direction", "Actual Price", "Predicted Value", "Profit/Loss"])
    total_profit_loss = df_results["Profit/Loss"].sum()
    return df_results, total_profit_loss