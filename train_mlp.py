import os
import numpy as np
from PIL import Image
from sklearn.neural_network import MLPClassifier
import pickle

X = []
y = []

letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

for i, letter in enumerate(letters):

    img_path = f"{letter}.png"

    if not os.path.exists(img_path):
        print(f"{img_path} not found")
        continue

    image = Image.open(img_path).convert('L')
    image = image.resize((28, 28))

    image_array = np.array(image)

    # Normalize
    image_array = image_array / 255.0

    # Same preprocessing as app
    image_array = np.rot90(image_array, k=3)
    image_array = np.fliplr(image_array)

    image_array = image_array.reshape(-1)

    X.append(image_array)
    y.append(i)

X = np.array(X)
y = np.array(y)

print("Training model...")

model = MLPClassifier(
    hidden_layer_sizes=(256,128),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42,
    verbose=True
)

model.fit(X, y)

print("Training Accuracy:", model.score(X, y))

pickle.dump(model, open("mlp_emnist_model.pkl", "wb"))

print("✅ Model trained and saved successfully!")