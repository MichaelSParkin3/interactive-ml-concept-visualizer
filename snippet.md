# make a graph that shows a parabola
# Create a new figure for the parabola
fig_parabola, ax_parabola = plt.subplots()
# Generate x values for the parabola
x_parabola = np.linspace(-40, 40, 100)
# Generate y values for the parabola (y = ax^2 + bx + c)
y_parabola = 0.1 * x_parabola**2 - 2 * x_parabola + 5
# Plot the parabola
ax_parabola.plot(x_parabola, y_parabola, color='green', label='Parabola: y = 0.1x^2 - 2x + 5')
# Add labels and title
ax_parabola.set_title('Parabola Plot')
ax_parabola.set_xlabel('x')
ax_parabola.set_ylabel('y')
ax_parabola.legend()
# Add grid for better readability
ax_parabola.grid(True)
# zoom out to see the whole parabola twice
ax_parabola.set_xlim(-10, 30)
ax_parabola.set_ylim(-10, 20)
#set a red point on the parabola at x=25
ax_parabola.plot(25, 0.1 * 25**2 - 2 * 25 + 5, 'ro')  # Red point at x=25

#animate the red point moving fast down the parabola
def update_parabola(frame):
    # Calculate the new x position for the red point
    x_red = 25 - frame * 0.5  # Move left along the parabola
    if x_red < -10:  # Stop moving when x is less than -10
        x_red = -10
    y_red = 0.1 * x_red**2 - 2 * x_red + 5
    # Update the red point's position
    ax_parabola.lines[1].set_data(x_red, y_red)
    return ax_parabola.lines[1],

# Create animation for the red point
ani_parabola = FuncAnimation(fig_parabola, update_parabola, frames=np.arange(0, 50), interval=100, blit=True)

# Show the parabola plot
# plt.show()