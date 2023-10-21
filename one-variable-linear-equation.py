import numpy as np
import tensorflow as tf

# input list (X-values)
x_values_list = [i for i in range(1, 100_000)]
X = np.array(x_values_list)

# output/answers list (Y-values, corresponding to the function f(x))
# f(x) = 2x + 3
y_values_list = [2*x + 3 for x in x_values_list]
Y = np.array(y_values_list)

# build and compile model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(loss='mean_squared_error', optimizer='adam')

# train the model
model.fit(X, Y, epochs=50, batch_size=64, validation_split=0.2)

print("--Training Complete--")

# make predictions
while True: 
	guess = float(input(" f(x)=2x+3 >> "))
	model_guess = model.predict(np.array([guess]))
	real_answer = (2*guess + 3)
	print("Model Guess:", model_guess )
	print("Real Answer:", real_answer, str((model_guess/real_answer)*10)+"%" )
