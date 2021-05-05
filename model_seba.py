import tensorflow as tf
from IPython.display import clear_output
import matplotlib.pyplot as plt

from os import listdir
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Conv2D, UpSampling2D, Lambda
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras import initializers, layers, models
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from skimage.morphology import label

tf.config.run_functions_eagerly(True)


SEED = 1234

IMG_PATH = 'F:\Download\WD_dane\Photos\\200MSDCF'
PROCESSED_IMG_PATH = 'F:\Download\WD_dane\Out_Results'
PREDICTION_RESULTS = 'F:\Download\WD_dane\Segmentation_Results'
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

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
        # if count > 100:
        #     break
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




train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_Y, train_class_weights))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(32)

test_dataset = tf.data.Dataset.from_tensor_slices((test_X, test_Y, test_class_weights))
test_dataset = test_dataset.shuffle(buffer_size=128).batch(16)


def plot_learning_curve(history):
    plt.figure(figsize=(8,8))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.clf()
    # summarize history for loss
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

def conv_block(inputs, filters, pool=True):
    x = Conv2D(filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    if pool == True:
        p = MaxPool2D((2, 2))(x)
        return x, p
    else:
        return x

def model1(train, test):
    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    """ Encoder """
    x1, p1 = conv_block(inputs, 16, pool=True)
    x2, p2 = conv_block(p1, 32, pool=True)
    x3, p3 = conv_block(p2, 48, pool=True)
    x4, p4 = conv_block(p3, 64, pool=True)

    """ Bridge """
    b1 = conv_block(p4, 128, pool=False)

    """ Decoder """
    u1 = UpSampling2D((2, 2), interpolation="bilinear")(b1)
    c1 = Concatenate()([u1, x4])
    x5 = conv_block(c1, 64, pool=False)

    u2 = UpSampling2D((2, 2), interpolation="bilinear")(x5)
    c2 = Concatenate()([u2, x3])
    x6 = conv_block(c2, 48, pool=False)

    u3 = UpSampling2D((2, 2), interpolation="bilinear")(x6)
    c3 = Concatenate()([u3, x2])
    x7 = conv_block(c3, 32, pool=False)

    u4 = UpSampling2D((2, 2), interpolation="bilinear")(x7)
    c4 = Concatenate()([u4, x1])
    x8 = conv_block(c4, 16, pool=False)

    """ Output layer """
    output = Conv2D(OUTPUT_CHANNELS, 1, padding="same", activation="softmax")(x8)

    model = Model(inputs, output)
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])

    model.summary()
    earlystopper = EarlyStopping(min_delta=0.0001, patience=3)

    results = model.fit(train,epochs=40, batch_size=16, validation_data=test, callbacks= [earlystopper])
    plot_learning_curve(results)
    plt.show()

    return model

def model2(train, test):

    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    s = Lambda(lambda x: x / 255) (inputs)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

    output = Conv2D(OUTPUT_CHANNELS, 1, padding="same", activation="softmax")(c9)


    model = Model(inputs, output)
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])
    model.summary()

    earlystopper = EarlyStopping(min_delta=0.0001, patience=2)


    results = model.fit(train,epochs=40, validation_data=test, callbacks= [earlystopper])
    plot_learning_curve(results)
    plt.show()

    return model



model = model2(train_dataset, test_dataset)



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

predictions = model.predict(test_X)
show_predictions(test_X, test_Y, predictions, 2)
