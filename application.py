import pickle

import matplotlib.pyplot as plt
import numpy as np
from dnn_app_utils import L_layer_model, load_data, predict, print_mislabeled_images
from PIL import Image

plt.rcParams["figure.figsize"] = (5.0, 4.0)  # set default size of plots
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"

np.random.seed(1)

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# Explore your dataset
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print("Number of training examples: " + str(m_train))
print("Number of testing examples: " + str(m_test))
print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("train_x_orig shape: " + str(train_x_orig.shape))
print("train_y shape: " + str(train_y.shape))
print("test_x_orig shape: " + str(test_x_orig.shape))
print("test_y shape: " + str(test_y.shape))

# Reshape the training and test examples
# The "-1" makes reshape flatten the remaining dimensions
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten / 255
test_x = test_x_flatten / 255

print("train_x's shape: " + str(train_x.shape))
print("test_x's shape: " + str(test_x.shape))
### CONSTANTS ###
layers_dims = [12288, 20, 7, 5, 1]  # 4-layer model

parameters = L_layer_model(
    train_x, train_y, layers_dims, num_iterations=2500, print_cost=True
)
pred_train = predict(train_x, train_y, parameters)
pred_test = predict(test_x, test_y, parameters)
print_mislabeled_images(classes, test_x, test_y, pred_test)

fileImage = Image.open("test.png").convert("RGB").resize([num_px, num_px])
my_label_y = [1]  # the true class of your image (1 -> cat, 0 -> non-cat)

image = np.array(fileImage)
my_image = image.reshape(num_px * num_px * 3, 1)
my_image = my_image / 255.0
my_predicted_image = predict(my_image, my_label_y, parameters)

plt.imshow(image)
print(
    "y = "
    + str(np.squeeze(my_predicted_image))
    + ', your L-layer model predicts a "'
    + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8")
    + '" picture.'
)
my_content = [train_x_orig, train_y, test_x_orig, test_y, classes]

# Saving variables in an array
with open("trainingDatasetL-LayerNN.pickle", "wb") as fileToBeWritten:
    # For compatibility we use open(filename, 'wb') for non-text files and open(filename, 'w') for text files
    pickle.dump(my_content, fileToBeWritten)
    # Loading Variables
with open("trainingDatasetL-LayerNN.pickle", "rb") as fileToBeRead:
    ttrain_x_orig, train_y, test_x_orig, test_y, classes = pickle.load(fileToBeRead)
