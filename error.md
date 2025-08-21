# Set up figure with initial scatter (done once)
fig, ax = plt.subplots()
ax.scatter(x, y, color='blue', label='Data points')  # Persistent scatter
ax.set_title('Animated Trend Line Fitting')
ax.set_xlabel('Independent Variable (x)')
ax.set_ylabel('Dependent Variable (y)')
ax.legend()
ax.grid(True)

# Initialize empty line
line, = ax.plot([], [], color='red', label='Trend Line')
ax.legend()  # Update legend for line

def init():
    line.set_data([], [])
    return line,

def update(frame):
    # Dynamically change slope over frames (0 to 3)
    slope = frame * 0.03  # e.g., frame 0-100 -> slope 0 to 3
    y_pred = slope * x  # Simple line (add intercept if needed)
    line.set_data(x, y_pred)
    return line,

# Create animation: 100 frames, 100ms interval
ani = FuncAnimation(fig, update, frames=100, init_func=init, interval=100, blit=True)

# Display as HTML5 video in Jupyter
HTML(ani.to_html5_video())