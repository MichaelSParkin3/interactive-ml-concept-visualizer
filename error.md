---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Cell In[29], line 21
     19 y = 0.1 * x**2 - 2 * x + 5
     20 # Store the current x and y values
---> 21 x_history.append(x[0])
     22 y_history.append(y)
     23 #print(f"Iteration {i+1}: x = {x[0]:.2f}, y = {y:.2f}")
     24 # Calculate the slope (derivative) at the current x value

TypeError: 'int' object is not subscriptable

This error means that you are trying to access an element of an integer as if it were a list or an array.

In the line `x_history.append(x[0])`, the variable `x` is an integer. Integers are single numerical values and do not support indexing (like `[0]`).

**To fix this, you should change the line:**

`x_history.append(x[0])`

**to:**

`x_history.append(x)`

This assumes that `x` is intended to be a single numerical value representing the current x-coordinate in your gradient descent simulation. If `x` was meant to be an array, then you would need to ensure it's initialized as such (e.g., `x = np.array([initial_x_value])`). However, given the context of a single point moving along the parabola, `x` being a simple integer or float is more likely.
