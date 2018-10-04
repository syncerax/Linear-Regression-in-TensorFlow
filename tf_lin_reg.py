import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Loading the dataset
dataset = np.loadtxt("data.txt", delimiter=',')
print(dataset.shape)

# Splitting it into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(dataset[:, :-1], dataset[:, -1], test_size=0.3, random_state=1)
Y_train = Y_train.reshape(Y_train.shape[0], 1)
Y_test = Y_test.reshape(Y_test.shape[0], 1)
# Some useful values
m, n = X_train.shape
print("Number of training examples:", m)
print("Number of features:", n)

# TensorFlow Time!

# Creating placeholders for the inputs
X = tf.placeholder(tf.float32, [None, n], name='X')
Y = tf.placeholder(tf.float32, [None, 1], name='Y')

# Creating variables to hold the parameters
W = tf.Variable(tf.zeros([n, 1]))
b = tf.Variable(tf.zeros([1]))

# Defining h(x), cost, and the optimizer
hypothesis = tf.add(tf.matmul(X, W), b, name='hypothesis')
cost = tf.reduce_mean(tf.square(hypothesis - Y), name='MSE')
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()
epochs = 100
history = []
test_history = []

with tf.Session() as sess:
    # Initializing all variables
    sess.run(init)
    # Training
    for epoch in range(epochs):
        dummy, c = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})
        history.append(c)
        #For plotting costVsIterations for testing
        ctest = sess.run(cost, feed_dict={X: X_test, Y: Y_test})
        test_history.append(ctest)

    # Making predictions on the training and testing sets    
    train_predictions = sess.run(hypothesis, feed_dict={X: X_train, Y: Y_train})
    test_predictions = sess.run(hypothesis, feed_dict={X: X_test, Y: Y_test})

    # Saving the model to disk
    tf.saved_model.simple_save(sess, 'lin-model-dir', 
        inputs={"X": X},
        outputs={"hypothesis": hypothesis})

print("First 5 predictions on the training set.")
print(train_predictions[:5])

print("Final cost:", history[-1])

plt.plot(history,label="training")
plt.plot(test_history,label="testing")
plt.legend(loc='upper right')
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()

# Plotting the predictions
min_max = [[X_train.min()], [X_train.max()]]
min_max_pred = train_predictions[[np.argmin(X_train), np.argmax(X_train)]]
plt.plot(min_max, min_max_pred)
plt.scatter(X_train[:, 0], Y_train, color="red")
plt.scatter(X_test[:, 0], Y_test, color="green")
plt.xlabel("Size")
plt.ylabel("Price")
plt.title("Predictions")
plt.show()

# Loading the saved model
print("\nAfter reloading the model:")
with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, ['serve'], 'lin-model-dir')
    graph = tf.get_default_graph()
    train_predictions = sess.run('hypothesis:0', feed_dict={'X:0': X_train})

print("First 5 predictions on the training set.")
print(train_predictions[:5])