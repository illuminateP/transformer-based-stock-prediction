import numpy as np
import keras
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from sklearn.preprocessing import StandardScaler

traindata = np.loadtxt('result_train.csv', delimiter=',', dtype=np.float32)
train_images = traindata[:,0:-1]
train_labels = traindata[:,[-1]]

testdata = np.loadtxt('result_test.csv', delimiter=',', dtype=np.float32)
test_images = testdata[:,0:-1]
test_labels = testdata[:,[-1]]

scaler = StandardScaler()
train_images_scaled = scaler.fit_transform(train_images)
test_images_scaled = scaler.transform(test_images)

accuracies = []
train_accuracies = []
val_accuracies = []

for _ in range(5):
    network = models.Sequential()
    network.add(layers.Dense(64, activation='relu'))
    network.add(layers.Dense(64, activation='relu'))
    network.add(layers.Dense(1, activation='sigmoid'))
    network.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 5)
    
    history = network.fit(train_images_scaled, train_labels,
                          batch_size=32,
                          epochs=10,
                          verbose=1,
                          shuffle='batch',
                          callbacks=[early_stopping],
                          validation_data=(test_images_scaled, test_labels))

    predict = network.predict(test_images_scaled)

    predicted_labels = (predict > 0.5).astype(int)

    accuracy = accuracy_score(test_labels, predicted_labels)
    train_accuracies.append(history.history['accuracy'])
    val_accuracies.append(history.history['val_accuracy'])

    print(f"정확도: {accuracy:.4f}")
    
    accuracies.append(accuracy)

average_accuracy = sum(accuracies) / len(accuracies)
print(f"평균 정확도: {average_accuracy:.4f}")

max_length_train = max(len(acc) for acc in train_accuracies)
max_length_val = max(len(acc) for acc in val_accuracies)

train_accuracies_padded = [np.pad(acc, (0, max_length_train - len(acc)), 'constant') for acc in train_accuracies]
val_accuracies_padded = [np.pad(acc, (0, max_length_val - len(acc)), 'constant') for acc in val_accuracies]

train_accuracies_array = np.array(train_accuracies_padded)
val_accuracies_array = np.array(val_accuracies_padded)

mean_train_accuracy = train_accuracies_array.mean(axis=0)
mean_val_accuracy = val_accuracies_array.mean(axis=0)

plt.figure(figsize=(12, 4))
plt.plot(mean_train_accuracy, label='Average Training Accuracy', color='blue')
plt.plot(mean_val_accuracy, label='Average Validation Accuracy', color='orange')
plt.title('Average Model Accuracy Across Runs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()