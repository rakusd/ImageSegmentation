import tensorflow as tf
from IPython.display import clear_output
import matplotlib.pyplot as plt
from tensorflow_examples.models.pix2pix import pix2pix
from os import listdir
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

SEED = 1234

IMG_PATH = 'Q:\Magisterka\SEM_III\Drogi\Dane_Przykładowe\Photos\\200MSDCF'
PROCESSED_IMG_PATH = 'Q:\Magisterka\SEM_III\Drogi\Results'
PREDICTION_RESULTS = 'Q:\Magisterka\SEM_III\Drogi\SegmentationResults'
IMG_WIDTH = 128
IMG_HEIGHT = 128

ASPHALT_PIX = 29
DIRT_PIX = 76
SAND_PIX = 150

ASPHALT_CLASS = 1
DIRT_CLASS = 2
SAND_CLASS = 3

OUTPUT_CHANNELS = 4

# LOAD IMAGES:
with open('./data/wrong_imgs/must.txt', 'r') as f:
    wrong_imgs = f.readlines()

wrong_img_names = [wrong_image.strip() for wrong_image in wrong_imgs if wrong_image.strip() != '']

image_names = [img.split('.')[0] for img in listdir(IMG_PATH)]


def read_imgs(image_names, wrong_images, verbose=False):
    raw_imgs = []
    final_imgs = []

    count = 0
    for img_name in image_names:
        count += 1
        if count > 5:
            break
        print(f'{img_name} {count}/{len(image_names)}')

        if img_name in wrong_images:
            continue

        img_path = IMG_PATH + '\\' + img_name + '.JPG'
        final_img_path = PROCESSED_IMG_PATH + '\\out_' + img_name + '.png'

        try:
            final_img = Image.open(final_img_path).convert('L')
        except:
            continue
        final_img = final_img.resize((IMG_WIDTH, IMG_HEIGHT), resample=Image.NEAREST)
        final_img = np.array(final_img)
        final_img[final_img == ASPHALT_PIX] = ASPHALT_CLASS
        final_img[final_img == DIRT_PIX] = DIRT_CLASS
        final_img[final_img == SAND_PIX] = SAND_CLASS
        final_imgs.append(final_img)

        raw_img = Image.open(img_path).convert('RGB')
        raw_img = raw_img.resize((IMG_WIDTH, IMG_HEIGHT))
        raw_img = np.array(raw_img)
        raw_img = raw_img / 255
        raw_imgs.append(raw_img)

    return np.array(raw_imgs), np.array(final_imgs)


raw_imgs, final_imgs = read_imgs(image_names, wrong_img_names, verbose=True)

# TRAIN TEST DS
train_X, test_X, train_Y, test_Y = train_test_split(raw_imgs, final_imgs, train_size=0.7, random_state=SEED)

print(train_X.shape)
print(train_X[0].shape)

class_nr, class_count = np.unique(train_Y, return_counts=True)
print(class_count)
print(class_nr)
train_class_weights = np.copy(train_Y)
for i in range(len(class_nr)):
    train_class_weights[train_Y == class_nr[i]] = sum(class_count) / class_count[i]

print(np.unique(train_class_weights, return_counts=True))

class_nr, class_count = np.unique(test_Y, return_counts=True)
print(class_count)
print(class_nr)
test_class_weights = np.copy(test_Y)
for i in range(len(class_nr)):
    test_class_weights[test_Y == class_nr[i]] = sum(class_count) / class_count[i]

print(np.unique(test_class_weights, return_counts=True))

# MODEL

base_model = tf.keras.applications.DenseNet121(input_shape=[IMG_WIDTH, IMG_HEIGHT, 3], include_top=False)
# Use the activations of these layers
layer_names = [
    'conv1/relu',  # 'block_1_expand_relu',   # 64x64
    'conv2_block1_0_relu',  # 'block_3_expand_relu',   # 32x32
    'conv3_block1_0_relu',  # 'block_6_expand_relu',   # 16x16
    'conv4_block1_0_relu',  # 'block_13_expand_relu',  # 8x8
    'conv5_block1_0_relu',  # 'block_16_project',      # 4x4
]

base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),  # 32x32 -> 64x64
]


def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_HEIGHT, 3])

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, strides=2,
        padding='same')  # 64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()


# Training

class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        print('\nSample Prediction after epoch {}\n'.format(epoch + 1))


train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_Y, train_class_weights))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(32)

test_dataset = tf.data.Dataset.from_tensor_slices((test_X, test_Y, test_class_weights))
test_dataset = test_dataset.shuffle(buffer_size=128).batch(16)

model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=5, batch_size=64, validation_data=test_dataset, callbacks=[DisplayCallback()])


# PREDictiONS

def display(display_list, index, save=False):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')
    plt.savefig(PREDICTION_RESULTS + f'\\pred_{index}.png')
    plt.show()


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    return pred_mask


def show_predictions(dataset_x, dataset_y, predictions, num=1, save=False):
    if dataset_x.any() and dataset_y.any():
        if num is None or num > dataset_x.shape[0]:
            num = dataset_x.shape[0]
        for i in range(num):
            display([dataset_x[i], dataset_y[i], create_mask(predictions[i])], i, save)


predictions = model.predict(train_X)

show_predictions(test_X, test_Y, predictions, 2, True)

# tf.keras.utils.plot_model(model, show_shapes=True)
