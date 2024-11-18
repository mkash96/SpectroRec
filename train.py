import numpy as np
import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras import initializers, optimizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Conv2D, BatchNormalization, AveragePooling2D, Flatten, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from data_loader import load_dataset
import pandas as pd

# Load dataset
train_x, train_y, test_x, test_y, n_classes, genre = load_dataset(verbose=1, mode="Train", datasetSize=0.75)

# Expand the dimensions of the image to have a channel dimension.
train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)
test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2], 1)

# Normalize the matrices.
train_x = train_x / 255.0
test_x = test_x / 255.0

# Define CNN model
model = Sequential()

# Convolutional Layers with Batch Normalization and Average Pooling
model.add(Conv2D(filters=64, kernel_size=(7,7), kernel_initializer=initializers.he_normal(seed=1), activation="relu", input_shape=(128,128,1)))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(2,2), strides=2))

model.add(Conv2D(filters=128, kernel_size=(7,7), strides=2, kernel_initializer=initializers.he_normal(seed=1), activation="relu"))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(2,2), strides=2))

model.add(Conv2D(filters=256, kernel_size=(3,3), kernel_initializer=initializers.he_normal(seed=1), activation="relu"))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(2,2), strides=2))

model.add(Conv2D(filters=512, kernel_size=(3,3), kernel_initializer=initializers.he_normal(seed=1), activation="relu"))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(2,2), strides=2))

# Fully Connected Layers
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dropout(0.6))
model.add(Dense(1024, activation="relu", kernel_initializer=initializers.he_normal(seed=1)))
model.add(Dropout(0.5))
model.add(Dense(256, activation="relu", kernel_initializer=initializers.he_normal(seed=1)))
model.add(Dropout(0.25))
model.add(Dense(64, activation="relu", kernel_initializer=initializers.he_normal(seed=1)))
model.add(Dense(32, activation="relu", kernel_initializer=initializers.he_normal(seed=1)))
model.add(Dense(n_classes, activation="softmax", kernel_initializer=initializers.he_normal(seed=1)))

# Compile model
model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])

# Implement Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Print model summary
print(model.summary())

# Plot model architecture
plot_model(model, to_file="Saved_Model/CNN_Model_Architecture.jpg", show_shapes=True)

# Train model with early stopping
history = model.fit(
    train_x, train_y,
    epochs=30,
    verbose=1,
    validation_split=0.1,
    callbacks=[early_stopping]
)

# Save training history
pd.DataFrame(history.history).to_csv("Saved_Model/CNN_training_history.csv")

# Evaluate model
score = model.evaluate(test_x, test_y, verbose=1)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Save model
model.save("Saved_Model/CNN_Model.h5")

# Plot Accuracy and Loss Graphs
plt.figure(figsize=(12, 5))

# Accuracy Graph
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Loss Graph
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.savefig("Saved_Model/CNN_Training_Accuracy_Loss.png")
plt.show()

# Predictions on the test set
y_pred = model.predict(test_x)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(test_y, axis=1)

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=genre, yticklabels=genre)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig("Saved_Model/CNN_Confusion_Matrix.png")
plt.show()

# Classification Report
print("\nClassification Report:\n", classification_report(y_true, y_pred_classes, target_names=genre))
