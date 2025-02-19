import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt


class PhysicsInformedELM:
    def __init__(self, input_dim, hidden_dim, activation="tanh"):
        """
        Initialize the Physics-Informed ELM.

        Args:
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Number of hidden neurons.
            activation (str): Activation function, options are 'tanh' or 'sigmoid'.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = self.get_activation_function(activation)
        self.activation_prime2 = self.get_activation_second_derivative(activation)

        # Randomly initialize input weights and biases
        self.W_in = np.random.randn(input_dim, hidden_dim)
        self.b = np.random.randn(hidden_dim)
        self.beta = None  # Output weights (to be trained)

    def get_activation_function(self, activation):
        if activation == "tanh":
            return np.tanh
        elif activation == "sigmoid":
            return lambda x: 1 / (1 + np.exp(-x))
        else:
            raise ValueError("Unsupported activation function.")

    def get_activation_second_derivative(self, activation):
        if activation == "tanh":
            return lambda x: -2 * np.tanh(x) * (1 - np.tanh(x) ** 2)
        elif activation == "sigmoid":
            return (
                lambda x: self.get_activation_function("sigmoid")(x)
                * (1 - self.get_activation_function("sigmoid")(x))
                * (1 - 2 * self.get_activation_function("sigmoid")(x))
            )
        else:
            raise ValueError("Unsupported activation function.")

    def hidden_layer_output(self, X):
        """
        Compute the hidden layer output.

        Args:
            X (np.ndarray): Input features (N x input_dim).

        Returns:
            np.ndarray: Hidden layer output (N x hidden_dim).
        """
        return self.activation(X @ self.W_in + self.b)

    def compute_laplacian(self, X):
        """
        Compute the Laplacian of the hidden layer output.

        Args:
            X (np.ndarray): Input features (N x input_dim).

        Returns:
            np.ndarray: Laplacian of the hidden layer output (N x hidden_dim).
        """
        N, d = X.shape
        laplacian = np.zeros((N, self.hidden_dim))

        for k in range(self.hidden_dim):
            z_k = X @ self.W_in[:, k] + self.b[k]
            laplacian[:, k] = self.activation_prime2(z_k) * np.sum(self.W_in[:, k] ** 2)

        return laplacian

    def compute_gradient(self, X):
        """
        Compute the gradient (acceleration) of the output.

        Args:
            X (np.ndarray): Input features (N x input_dim).

        Returns:
            np.ndarray: Gradient of the output (N x input_dim).
        """
        N, d = X.shape
        gradient = np.zeros((N, d))

        for k in range(self.hidden_dim):
            z_k = X @ self.W_in[:, k] + self.b[k]  # Pre-activation of the k-th neuron
            activation_derivative = self.activation(
                z_k
            )  # Activation for the k-th neuron
            grad_k = (
                activation_derivative[:, None] * self.W_in[:, k]
            )  # Gradient contribution of neuron k
            gradient += grad_k * self.beta[k]  # Scale by output weight of neuron k

        return gradient

    def train(self, X_data, y_data, X_collocation, lambda_physics=1e-3):
        """
        Train the Physics-Informed ELM.

        Args:
            X_data (np.ndarray): Input features for data loss (N_data x input_dim).
            y_data (np.ndarray): Target outputs for data loss (N_data x 1).
            X_collocation (np.ndarray): Collocation points for enforcing physics (N_collocation x input_dim).
            lambda_physics (float): Regularization parameter for physics loss.
        """
        # Hidden layer output for data
        H_data = self.hidden_layer_output(X_data)

        # Hidden layer Laplacian for collocation points
        H_physics = self.compute_laplacian(X_collocation)

        # Augmented system
        H_aug = np.vstack([H_data, np.sqrt(lambda_physics) * H_physics])
        y_aug = np.vstack([y_data, np.zeros((H_physics.shape[0], 1))])

        # Solve for beta using least squares
        self.beta, _, _, _ = lstsq(H_aug, y_aug)

    def predict(self, X):
        """
        Make predictions using the trained ELM.

        Args:
            X (np.ndarray): Input features (N x input_dim).

        Returns:
            np.ndarray: Predicted outputs (N x 1).
        """
        H = self.hidden_layer_output(X)
        return H @ self.beta


# Load the dataset
data = np.load("cylindrical_gravity_dataset.npz")
X_data = data["points"]  # 3D points
y_data = data["potential"].reshape(-1, 1)  # Gravitational potential at each point

# Generate collocation points (same range as dataset points)
X_collocation = np.random.uniform(
    low=np.min(X_data, axis=0),
    high=np.max(X_data, axis=0),
    size=(1000, X_data.shape[1]),
)

# Train the Physics-Informed ELM
elm = PhysicsInformedELM(input_dim=3, hidden_dim=10000, activation="tanh")
elm.train(X_data, y_data, X_collocation, lambda_physics=1e-3)

# Test and evaluate the ELM
X_test = X_data  # Test on training data (you can use new points)
y_pred = elm.predict(X_test)

# Compute percentage errors
potential_error = np.abs((y_pred.flatten() - y_data.flatten()) / y_data.flatten()) * 100

# Compute predicted acceleration (gradient of potential)
y_pred_gradient = elm.compute_gradient(X_test)

# Load true acceleration data for comparison
true_acceleration = data["acceleration"]  # True acceleration at each point

# Compute percentage error in acceleration
acceleration_error_magnitude = (
    np.linalg.norm(y_pred_gradient - true_acceleration, axis=1)
    / np.linalg.norm(true_acceleration, axis=1)
    * 100
)

# Plot results
plt.figure(figsize=(12, 6))

# Subplot 1: Potential Error
plt.subplot(2, 1, 1)
plt.plot(potential_error, label="Potential Error (%)", color="blue")
plt.xlabel("Sample Index")
plt.ylabel("Error (%)")
plt.title("Percentage Error in Gravitational Potential")
plt.legend()

# Subplot 2: Acceleration Error
plt.subplot(2, 1, 2)
plt.plot(acceleration_error_magnitude, label="Acceleration Error (%)", color="orange")
plt.xlabel("Sample Index")
plt.ylabel("Error (%)")
plt.title("Percentage Error in Gravitational Acceleration")
plt.legend()

plt.tight_layout()
plt.show()
