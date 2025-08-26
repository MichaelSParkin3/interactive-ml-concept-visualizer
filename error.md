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

# make the data into a smaller sample for faster training for learning purpose
sample_size = 1000
indices = np.random.choice(X.shape[0], sample_size, replace=False)
X = X[indices]
y = y[indices]
print(X.shape, y.shape)
# set up a model to train on this housing data with batch settings, learning rate, and initial parameters
np.random.seed(42)
# batch settings
batch_size = 500
n_batches = int(np.ceil(X.shape[0] / batch_size))
# learning rate
learning_rate = 0.015
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
n_frames = 200
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