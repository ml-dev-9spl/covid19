# baseline model for the dogs vs cats dataset
import sys
import pandas as pd
import shutil
import os
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
from tensorflow.python.keras.preprocessing import image

from flask_app import BASE_DIR


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


def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()


# run the test harness for evaluating a model
def run_test_harness():
    # train data generator
    train_datagen = ImageDataGenerator(rescale=1.0 / 255.0, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

    # test data generator
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_it = train_datagen.flow_from_directory('ml_app/out/train',
                                                 class_mode='categorical', batch_size=64, target_size=(200, 200))
    test_it = test_datagen.flow_from_directory('ml_app/out/test',
                                               class_mode='categorical', batch_size=64, target_size=(200, 200))

    # define model
    model = define_model()

    history = model.fit(train_it, steps_per_epoch=len(train_it),
                                  validation_data=test_it, validation_steps=len(test_it), epochs=3, verbose=1)
    # evaluate model
    _, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
    print('> %.3f' % (acc * 100.0))

    # learning curves
    model.save(os.path.join(BASE_DIR, 'ml_app', 'saved_models'),  save_format='h5')
    loaded_model = load_model(os.path.join(BASE_DIR, 'ml_app', 'saved_models'))
    return loaded_model # loaded_model.summary()


if __name__ == '__main__':
    run_test_harness()


def load_and_predict(xray):
    img = image.load_img(xray,
                         target_size=(200, 200))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    # loaded_model = run_test_harness()
    loaded_model = load_model(os.path.join(BASE_DIR, 'ml_app', 'saved_models'))
    prediction = loaded_model.predict(img)[0][0] * 100
    return prediction
