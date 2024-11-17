import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

GESTURES = ['hello','iloveyou','bye','stop','yes','no','peace']

def train_model():
    X = np.load('landmarks.npy')
    y = np.load('labels.npy')

    y = np.eye(len(GESTURES))[y]  # One-hot encode labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential([
        Dense(128, activation='relu', input_shape=(X.shape[1],)),
        Dense(64, activation='relu'),
        Dense(len(GESTURES), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=32)

    model.save('gesture_model.h5')
    print("Model Training Completed!")

if __name__ == "__main__":
    train_model()
