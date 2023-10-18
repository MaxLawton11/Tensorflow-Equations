import numpy as np
import tensorflow as tf

# List 1 (X-values, 1000 data points)
x_values_list = [i for i in range(1, 100_000)]
X = np.array(x_values_list)

# List 2 (Y-values, corresponding to the function f(x), 1000 data points)
# f(x) = 2x + 3
y_values_list = [2*x + 3 for x in x_values_list]
y = np.array(y_values_list)

# Build and compile your model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X, y, epochs=100, batch_size=64, validation_split=0.2)

print("--Training Complete--")

# Make predictions

while True: 
	guess = float(input(" f(x)=2x+3 >> "))
	print( model.predict(np.array([guess])) )
