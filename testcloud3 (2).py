import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet18
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dataset_dir = 'path_to_dataset' #set the path to your dataset

#mode architecture of FPN model
def create_model(input_shape, num_classes):
    base_model = ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')

    #freeze weights
    base_model.trainable = False

    C2 = base_model.get_layer('stage2_unit1_relu1').output
    C3 = base_model.get_layer('stage3_unit1_relu1').output
    C4 = base_model.get_layer('stage4_unit1_relu1').output
    C5 = base_model.get_layer('stage5_unit1_relu1').output

    P5 = layers.Conv2D(256, 1, strides=1, padding='same')(C5)
    P4 = layers.Add()([layers.UpSampling2D()(P5), layers.Conv2D(256, 1, strides=1, padding='same')(C4)])
    P3 = layers.Add()([layers.UpSampling2D()(P4), layers.Conv2D(256, 1, strides=1, padding='same')(C3)])
    P2 = layers.Add()([layers.UpSampling2D()(P3), layers.Conv2D(256, 1, strides=1, padding='same')(C2)])

    #smoothen the layers
    P3 = layers.Conv2D(256, 3, strides=1, padding='same')(P3)
    P4 = layers.Conv2D(256, 3, strides=1, padding='same')(P4)
    P5 = layers.Conv2D(256, 3, strides=1, padding='same')(P5)
    P2 = layers.Conv2D(256, 3, strides=1, padding='same')(P2)

    P3_upsampled = layers.UpSampling2D()(P3)
    P4_upsampled = layers.UpSampling2D()(P4)
    P5_upsampled = layers.UpSampling2D()(P5)

    feature_pyramid = layers.Concatenate()([P2, P3_upsampled, P4_upsampled, P5_upsampled])
    output = layers.Conv2D(num_classes, 3, activation='softmax', padding='same')(feature_pyramid)
    model = tf.keras.Model(inputs=base_model.input, outputs=output)
    return model

#loading the dataset
def load_dataset(dataset_dir):
    images = []
    labels = []

    image_dir = os.path.join(dataset_dir, 'images')
    label_dir = os.path.join(dataset_dir, 'labels')

    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(image_dir, filename)
            label_path = os.path.join(label_dir, filename[:-4] + '.png')

            img = load_img(img_path, target_size=(256, 256))
            img = img_to_array(img)
            images.append(img)

            label = load_img(label_path, target_size=(256, 256), color_mode='grayscale')
            label = img_to_array(label)
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels

#preprocessing
def preprocess_dataset(images, labels, num_classes):
    images = images.astype('float32') / 255.0
    labels = labels.astype('float32') / 255.0
    labels = np.expand_dims(labels, axis=-1)

    labels = np.where(labels > 0.5, 1, 0)  #threshold segmentation mask

    #splitting into training and testing 
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

    return train_images, test_images, train_labels, test_labels

#model params
input_shape = (256, 256, 3) #or change the input size to (256,256,1) according to your usage
num_classes = 2  #cloud and background are the two classes here

images, labels = load_dataset(dataset_dir) #load the dataset
train_images, test_images, train_labels, test_labels = preprocess_dataset(images, labels, num_classes) #preprocess the dataset

model = create_model(input_shape, num_classes) #creating the model

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) #model compilation

model.fit(train_images, train_labels, epochs=10, batch_size=32) #training rhe model

#model evaluation 
loss, accuracy = model.evaluate(test_images, test_labels, batch_size=32)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')

predictions = model.predict(test_images) #semantic segmantation on test images

predictions = np.argmax(predictions, axis=-1) #Convert probabilities to class indices


#visualization using matplotlib
n = 10
fig, axes = plt.subplots(n, 3, figsize=(10, 20))

for i in range(n):
    axes[i, 0].imshow(test_images[i])
    axes[i, 0].axis('off')
    axes[i, 0].set_title('Input')

    axes[i, 1].imshow(np.squeeze(test_labels[i]), cmap='gray')
    axes[i, 1].axis('off')
    axes[i, 1].set_title('Ground Truth')

    axes[i, 2].imshow(np.squeeze(predictions[i]), cmap='gray')
    axes[i, 2].axis('off')
    axes[i, 2].set_title('Predicted')

plt.tight_layout()
plt.show()