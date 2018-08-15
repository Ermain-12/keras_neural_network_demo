import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Generate dummy data
X_train = np.random.random((1000, 20))
y_train = np.random.randint(2, size=(1000, 1))
X_test = np.random.random((100, 20))
y_test = np.random.randint(2, size=(100, 20))

model = Sequential()

model.add(Dense(64, input_dim=20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
)

# Fit the training data on the neural network
model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=128
)

# Score our neural network
score = model.evaluate(X_test, y_test, batch_size=128)
print("Accuracy: ", score)
