# baseline model for the dogs vs cats dataset
import os
import sys
import shutil
import numpy as np
import pandas as pd
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Dropout
from tensorflow.python.keras.models import Sequential

from flask_app import BASE_DIR
from keras_preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import load_model

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_train_generator():
    imagegen = ImageDataGenerator()
    # load train data
    train_it = imagegen.flow_from_directory("out/train", class_mode="categorical", shuffle=False, batch_size=128,
                                         target_size=(224, 224))
    # load val data
    val = imagegen.flow_from_directory("out/test", class_mode="categorical", shuffle=False, batch_size=128,
                                       target_size=(224, 224))
    return train_it

def get_test_generator():
    datagen = ImageDataGenerator(featurewise_center=True)
    # specify imagenet mean values for centering
    # prepare iterator
    test_it = datagen.flow_from_directory('out/test',
                                           class_mode='categorical', batch_size=128, target_size=(224, 224))
    return test_it

def define_model():
    # load model
    model = VGG16(include_top=False, input_shape=(224, 224, 3))
    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(2, activation='sigmoid')(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def define_sequencial_model():
    # building a linear stack of layers with the sequential model
    model = Sequential()

    # convolutional layer
    model.add(Conv2D(50, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=(224, 224, 3)))


    # convolutional layer
    model.add(Conv2D(75, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(125, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # flatten output of conv
    model.add(Flatten())

    # hidden layer
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(250, activation='relu'))
    model.add(Dropout(0.3))
    # output layer
    model.add(Dense(2, activation='softmax'))

    # compiling the sequential model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    # training the model for 10 epochs
    return model

def run_sequenctial():
    model = define_sequencial_model()
    train_it = get_train_generator()
    print([train_it.class_indices])
    model.fit(train_it, epochs=10)
    model.save(os.path.join(BASE_DIR, 'ml_app', 'covid19'), save_format='h5')
    model.summary()

# run the test harness for evaluating a model
def run_test_harness():
    # define model
    model = define_model()
    # create data generator
    datagen = ImageDataGenerator(featurewise_center=True)
    # specify imagenet mean values for centering
    datagen.mean = [123.68, 116.779, 103.939]
    # prepare iterator
    train_it = datagen.flow_from_directory('out/train',
                                           class_mode='categorical', batch_size=64, target_size=(224, 224))
    # fit model
    print([train_it.class_indices])
    model.fit(train_it, steps_per_epoch=len(train_it), epochs=10, verbose=1)
    model.save(os.path.join(BASE_DIR, 'ml_app', 'saved_models'), save_format='h5')
    loaded_model = load_model(os.path.join(BASE_DIR, 'ml_app', 'saved_models'))
    return loaded_model  # loaded_model.summary()



def icloud_api():
    from pyicloud import PyiCloudService
    api = PyiCloudService('divyesh.dabi@icloud.com', 'Div32190')
    if api.requires_2sa:
        import click
        print("Two-step authentication required. Your trusted devices are:")

        devices = api.trusted_devices
        for i, device in enumerate(devices):
            print("  %s: %s" % (i, device.get('deviceName',
                                        "SMS to %s" % device.get('phoneNumber'))))

        device = click.prompt('Which device would you like to use?', default=0)
        device = devices[device]
        if not api.send_verification_code(device):
            print("Failed to send verification code")
            sys.exit(1)

        code = click.prompt('Please enter validation code')
        if not api.validate_verification_code(device, code):
            print("Failed to verify verification code")
            sys.exit(1)
    print(api.iphone.location())
    print("Files =============>", api.files.dir())
    print("Archives ==========>", api.archives.dir())

if __name__ == '__main__':
    # run_test_harness()
    # run_sequenctial()
    icloud_api()

def load_and_predict(xray):
    # load the image
    img = load_img(xray, target_size=(224, 224))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 224, 224, 3)
    # center pixel data
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    model = load_model(os.path.join(BASE_DIR, 'ml_app', 'saved_models'))
    result = model.predict(img)
    return result.tolist()
    # print(result.argmax(axis=-1))


def get_covid_19_xrays():
    # Selecting all combination of 'COVID-19' patients with 'PA' X-Ray view
    virus = "COVID-19"  # Virus to look for
    x_ray_view = "PA"  # View of X-Ray

    metadata = "covid-dataset/metadata.csv"  # Meta info
    imageDir = "covid-dataset/images"  # Directory of images
    outputDir = 'out/'  # Output directory to store selected images

    df = pd.read_csv(metadata)

    # loop over the rows of the COVID-19 data frame

    df = df.loc[df['finding'].isin(['COVID-19', 'SARS']) & (df['view'] == 'PA')]
    test, train = np.split(df[["filename", "finding", "view"]], [int(.2 * len(df))])

    # copy train set to train directory
    for (i, row) in train.iterrows():
        if row["finding"] == 'COVID-19':
            filename = row["filename"].split(os.path.sep)[-1]
            filePath = os.path.sep.join([imageDir, filename])
            shutil.copy2(filePath, outputDir + 'train/COVID-19')
        if row["finding"] == 'SARS':
            filename = row["filename"].split(os.path.sep)[-1]
            filePath = os.path.sep.join([imageDir, filename])
            shutil.copy2(filePath, outputDir + 'train/SARS')

    # copy testing set to the test directory
    for (i, row) in test.iterrows():
        if row["finding"] == 'COVID-19':
            filename = row["filename"].split(os.path.sep)[-1]
            filePath = os.path.sep.join([imageDir, filename])
            shutil.copy2(filePath, outputDir + 'test/COVID-19')
        if row["finding"] == 'SARS':
            filename = row["filename"].split(os.path.sep)[-1]
            filePath = os.path.sep.join([imageDir, filename])
            shutil.copy2(filePath, outputDir + 'test/SARS')


