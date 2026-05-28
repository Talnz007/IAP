"""
Deep Learning models for AQI prediction using TensorFlow.
"""
from typing import Dict, Any, Tuple, List
import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not installed. Deep learning models will not be available.")


def check_tf_available():
    """Check if TensorFlow is available."""
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for deep learning models. Install with: pip install tensorflow")


def build_mlp_model(
    input_dim: int,
    hidden_layers: List[int] = None,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001
) -> 'Model':
    """
    Build a Multi-Layer Perceptron model.
    
    Args:
        input_dim: Number of input features
        hidden_layers: List of hidden layer sizes
        dropout_rate: Dropout rate for regularization
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled Keras model
    """
    check_tf_available()
    
    if hidden_layers is None:
        hidden_layers = [128, 64, 32]
    
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
    ])
    
    for units in hidden_layers:
        model.add(layers.Dense(units, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
    
    model.add(layers.Dense(1))  # Output layer for regression
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def build_lstm_model(
    sequence_length: int,
    n_features: int,
    lstm_units: List[int] = None,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001
) -> 'Model':
    """
    Build an LSTM model for time series prediction.
    
    Args:
        sequence_length: Length of input sequences
        n_features: Number of features per timestep
        lstm_units: List of LSTM layer sizes
        dropout_rate: Dropout rate for regularization
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled Keras model
    """
    check_tf_available()
    
    if lstm_units is None:
        lstm_units = [64, 32]
    
    model = keras.Sequential([
        layers.Input(shape=(sequence_length, n_features)),
    ])
    
    # Add LSTM layers
    for i, units in enumerate(lstm_units):
        return_sequences = i < len(lstm_units) - 1
        model.add(layers.LSTM(units, return_sequences=return_sequences))
        model.add(layers.Dropout(dropout_rate))
    
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))  # Output layer
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def build_gru_model(
    sequence_length: int,
    n_features: int,
    gru_units: List[int] = None,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001
) -> 'Model':
    """
    Build a GRU model for time series prediction.
    
    Args:
        sequence_length: Length of input sequences
        n_features: Number of features per timestep
        gru_units: List of GRU layer sizes
        dropout_rate: Dropout rate for regularization
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled Keras model
    """
    check_tf_available()
    
    if gru_units is None:
        gru_units = [64, 32]
    
    model = keras.Sequential([
        layers.Input(shape=(sequence_length, n_features)),
    ])
    
    # Add GRU layers
    for i, units in enumerate(gru_units):
        return_sequences = i < len(gru_units) - 1
        model.add(layers.GRU(units, return_sequences=return_sequences))
        model.add(layers.Dropout(dropout_rate))
    
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))  # Output layer
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def build_cnn_lstm_model(
    sequence_length: int,
    n_features: int,
    cnn_filters: int = 64,
    lstm_units: int = 50,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001
) -> 'Model':
    """
    Build a CNN-LSTM hybrid model for time series prediction.
    
    Args:
        sequence_length: Length of input sequences
        n_features: Number of features per timestep
        cnn_filters: Number of CNN filters
        lstm_units: Number of LSTM units
        dropout_rate: Dropout rate for regularization
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled Keras model
    """
    check_tf_available()
    
    model = keras.Sequential([
        layers.Input(shape=(sequence_length, n_features)),
        
        # CNN layers for feature extraction
        layers.Conv1D(filters=cnn_filters, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(dropout_rate),
        
        # LSTM layer for temporal dependencies
        layers.LSTM(lstm_units),
        layers.Dropout(dropout_rate),
        
        # Dense layers
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def prepare_sequences(
    X: np.ndarray, 
    y: np.ndarray, 
    sequence_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences for LSTM/GRU models.
    
    Args:
        X: Feature array
        y: Target array
        sequence_length: Length of sequences to create
        
    Returns:
        Tuple of (X_sequences, y_sequences)
    """
    X_seq, y_seq = [], []
    
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length])
    
    return np.array(X_seq), np.array(y_seq)


def get_callbacks(
    patience: int = 10,
    model_path: str = None
) -> List:
    """
    Get training callbacks.
    
    Args:
        patience: Patience for early stopping
        model_path: Path to save best model
        
    Returns:
        List of Keras callbacks
    """
    check_tf_available()
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    if model_path:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=True
            )
        )
    
    return callbacks
