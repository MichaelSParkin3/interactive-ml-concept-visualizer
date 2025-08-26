from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing(as_frame=True)
print(housing.data.shape, housing.target.shape)
print(housing.feature_names[0:6])
# select 1 feature ( Feature = MedianIncome )
X = housing.data[["MedInc"]].to_numpy()
# scale feature with standard scaler. Why?; because the range of feature is too large.
#X = (X - X.mean()) / X.std()
# select target ( Target = MedianHouseValue )
y = housing.target.to_numpy()
print(X.shape, y.shape)
# create a scatter plot
fig, ax = plt.subplots()
ax.scatter(X, y, s=1)
ax.set_xlabel("Median Income (scaled)")
ax.set_ylabel("Median House Value")
ax.set_title("California Housing Data")
plt.show()

---

### FIRST GRAPH

# lists for storing loss values by epoch
loss_list = []
epoch_list = []

# make the data into a smaller sample for faster training for learning purpose
sample_size = 1000
indices = np.random.choice(X.shape[0], sample_size, replace=False)
X = X[indices]
y = y[indices]
print(X.shape, y.shape)
# set up a model to train on this housing data with batch settings, learning rate, and initial parameters
np.random.seed(42)
# batch settings
batch_size = 250
n_batches = int(np.ceil(X.shape[0] / batch_size))
# epoch settings
n_epochs = 100
# learning rate
learning_rate = 0.03
# set weights (initial parameters) as set value of 1.5
theta = np.array([[1.2]])
print(theta)
# set bias (initial parameters)
bias = 0.1
print(bias)
# set up a figure for animation
fig, ax = plt.subplots()
ax.scatter(X, y, s=1)
line, = ax.plot(X, X.dot(theta) + bias, color='red')
ax.set_ylim(0, 5.5)
ax.set_xlabel("Median Income (scaled)")
ax.set_ylabel("Median House Value")
ax.set_title("California Housing Data with Linear Regression Fit")
plt.close()
# function to update the line for animation
def update_line(frame):
    global theta, bias, X_shuffled, y_shuffled
    # shuffle the data at the beginning of each epoch
    if frame % n_batches == 0:
        # Calculate loss over the entire dataset at the start of each epoch
        y_pred_full = X.dot(theta) + bias
        loss = np.mean((y_pred_full - y.reshape(-1, 1))**2)
        epoch = frame // n_batches
        print(f"Epoch: {epoch}, Loss: {loss:.4f}")
        
        epoch_list.append(epoch)
        loss_list.append(loss)
        
        indices = np.random.permutation(X.shape[0])
        X_shuffled = X[indices]
        y_shuffled = y[indices]
    # get the current batch
    batch_index = frame % n_batches
    start = batch_index * batch_size
    end = min(start + batch_size, X.shape[0])
    X_batch = X_shuffled[start:end]
    y_batch = y_shuffled[start:end]
    # add bias term to the input features
    X_batch_b = np.c_[np.ones((X_batch.shape[0], 1)), X_batch]
    # make predictions
    y_pred = X_batch_b.dot(np.array([bias, theta.item()]).reshape(-1, 1
     ))
    # compute gradients
    error = y_pred - y_batch.reshape(-1, 1)
    gradients = 2 / X_batch_b.shape[0] * X_batch_b.T.dot(error)
    # update parameters
    bias -= learning_rate * gradients[0, 0]
    theta -= learning_rate * gradients[1:, 0].reshape(-1, 1)
    # update the line in the plot
    line.set_ydata(X.dot(theta) + bias)
    return line,
# create the animation
n_frames = n_epochs * n_batches
ani = FuncAnimation(fig, update_line, frames=n_frames, blit=True, interval=100)
# display the animation
HTML(ani.to_jshtml())



---

# Create a new figure for the loss plot
fig_loss, ax_loss = plt.subplots()
ax_loss.plot(epoch_list, loss_list)
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Loss (MSE)")
ax_loss.set_title("Loss per Epoch")
ax_loss.grid(True)
plt.show()

---

# Full script for 3D visualization with two features

# --- Step 1: Imports and Data Preparation ---
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Necessary for 3D plots
from sklearn.datasets import fetch_california_housing

# Load data
housing = fetch_california_housing(as_frame=True)

# Select two features ("MedInc", "AveRooms") and the target
X_3d = housing.data[["MedInc", "AveRooms"]].to_numpy()
y_3d = housing.target.to_numpy()

# Create a smaller sample for faster training
sample_size = 1000
np.random.seed(42) # for reproducibility
indices = np.random.choice(X_3d.shape[0], sample_size, replace=False)
X_3d = X_3d[indices]
y_3d = y_3d[indices]

print("---> Data Loaded ---")
print(f"X shape: {X_3d.shape}")
print(f"y shape: {y_3d.shape}")


# --- Step 2: Model Training ---
print("\n---> Starting Model Training ---")

# Hyperparameters
learning_rate = 0.01
n_epochs = 30
batch_size = 100
n_batches = int(np.ceil(X_3d.shape[0] / batch_size))

# Initial parameters (2 weights for 2 features)
np.random.seed(42) # for reproducibility
theta_3d = np.random.randn(2, 1)
bias_3d = np.random.randn(1)

# --- Training Loop ---
for epoch in range(n_epochs):
    # Shuffle data at the start of each epoch
    shuffle_indices = np.random.permutation(X_3d.shape[0])
    X_shuffled = X_3d[shuffle_indices]
    y_shuffled = y_3d[shuffle_indices]

    for i in range(n_batches):
        # Get batch
        start = i * batch_size
        end = min(start + batch_size, X_3d.shape[0])
        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]

        # Add bias term to features
        X_batch_b = np.c_[np.ones((X_batch.shape[0], 1)), X_batch]

        # Combine all parameters into one vector [bias, theta1, theta2]
        params = np.vstack([bias_3d, theta_3d])

        # Make predictions and calculate error
        y_pred = X_batch_b.dot(params)
        error = y_pred - y_batch.reshape(-1, 1)

        # Compute gradients
        gradients = 2 / X_batch.shape[0] * X_batch_b.T.dot(error)

        # Update parameters
        bias_3d -= learning_rate * gradients[0, 0]
        theta_3d -= learning_rate * gradients[1:, 0].reshape(-1, 1)

    # Print loss at the end of each epoch (print every 5th epoch)
    if epoch % 5 == 0 or epoch == n_epochs - 1:
        y_pred_full = X_3d.dot(theta_3d) + bias_3d
        loss = np.mean((y_pred_full - y_3d.reshape(-1, 1))**2)
        print(f"Epoch: {epoch}, Loss: {loss:.4f}")

print("\n---> Training Complete ---")
print(f"Final Bias: {bias_3d[0]:.4f}")
print(f"Final Thetas: [{theta_3d[0,0]:.4f}, {theta_3d[1,0]:.4f}]")


# --- Step 3: Plot the Final 3D Result ---
print("\n---> Generating 3D Plot ---")

# Create a meshgrid to plot the regression plane
x_surf = np.linspace(X_3d[:, 0].min(), X_3d[:, 0].max(), 10)
y_surf = np.linspace(X_3d[:, 1].min(), X_3d[:, 1].max(), 10)
x_surf, y_surf = np.meshgrid(x_surf, y_surf)

# Calculate the z-values for the plane using the final trained parameters
z_surf = (x_surf * theta_3d[0, 0] + y_surf * theta_3d[1, 0] + bias_3d).squeeze()

# Create the 3D figure
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Plot the original data points
ax.scatter(X_3d[:, 0], X_3d[:, 1], y_3d, c='blue', s=5, label='Data Points', alpha=0.6)

# Plot the regression plane
ax.plot_surface(x_surf, y_surf, z_surf, color='red', alpha=0.5, label='Regression Plane')

# Set labels and title
ax.set_xlabel('Median Income', fontsize=10)
ax.set_ylabel('Average Rooms', fontsize=10)
ax.set_zlabel('Median House Value', fontsize=10)
ax.set_title('3D Linear Regression with Two Features', fontsize=14)

plt.show()