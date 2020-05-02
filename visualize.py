import matplotlib.pyplot as plt 
import numpy as np

# Plot learning curves for best model 
train_loss = np.load("train_loss.npy")
test_loss = np.load("test_loss.npy")
epochs = list(range(len(train_loss)))

plt.plot(epochs, train_loss)
plt.plot(epochs, test_loss)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"])
plt.savefig("learningcurve.png")


# Plot the log log scale training set vs train and test accuracy 
training_size = np.array([3188, 6375, 12750, 25500, 60000])
training_size = np.log(training_size)

train_loss = [0.1980, 0.1084, 0.0714, 0.0477, 0.0383]
test_loss = [0.2209, 0.1312, 0.0913, 0.0603, 0.0530]

train_loss = np.log(train_loss)
test_loss = np.log(test_loss)

plt.plot(training_size, train_loss)
plt.plot(training_size, test_loss)
plt.xlabel("Log Training Size")
plt.ylabel("Log Loss")
plt.legend(["Train", "Validation"])
plt.savefig("size_vs_perf.png")
