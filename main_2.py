import os
import random
import shutil
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import VGG16, ResNet50
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def prepare_dataset(original_directory, base_directory):
    # If the folder already exists, remove everything
    if os.path.exists(base_directory):
        shutil.rmtree(base_directory)

    # Recreate the base folder
    os.mkdir(base_directory)

    # Create the training folder in the base directory
    train_directory = os.path.join(base_directory, 'train')
    os.mkdir(train_directory)

    # Create the validation folder in the base directory
    validation_directory = os.path.join(base_directory, 'validation')
    os.mkdir(validation_directory)

    # Create the test folder in the base directory
    test_directory = os.path.join(base_directory, 'test')
    os.mkdir(test_directory)

    # Create the positive/negative folders in training/validation directories
    train_positive_directory = os.path.join(train_directory, 'positive')
    os.mkdir(train_positive_directory)

    train_negative_directory = os.path.join(train_directory, 'negative')
    os.mkdir(train_negative_directory)

    validation_positive_directory = os.path.join(validation_directory, 'positive')
    os.mkdir(validation_positive_directory)

    validation_negative_directory = os.path.join(validation_directory, 'negative')
    os.mkdir(validation_negative_directory)

    test_directory = os.path.join(test_directory, 'test_folder')
    os.mkdir(test_directory)

    # Shuffle and split the positive files
    positive_files = os.listdir(os.path.join(original_directory, 'positive'))
    random.shuffle(positive_files)
    num_positive_files = len(positive_files)
    num_train_positive_files = int(0.6 * num_positive_files)
    num_validation_positive_files = int(0.2 * num_positive_files)

    for fname in positive_files[:num_train_positive_files]:
        src = os.path.join(original_directory, 'positive', fname)
        dst = os.path.join(train_positive_directory, fname)
        shutil.copyfile(src, dst)

    for fname in positive_files[num_train_positive_files:num_train_positive_files + num_validation_positive_files]:
        src = os.path.join(original_directory, 'positive', fname)
        dst = os.path.join(validation_positive_directory, fname)
        shutil.copyfile(src, dst)

    for fname in positive_files[num_train_positive_files + num_validation_positive_files:]:
        src = os.path.join(original_directory, 'positive', fname)
        dst = os.path.join(test_directory, fname)
        shutil.copyfile(src, dst)

    # Shuffle and split the negative files
    negative_files = os.listdir(os.path.join(original_directory, 'negative'))
    random.shuffle(negative_files)
    num_negative_files = len(negative_files)
    num_train_negative_files = int(0.6 * num_negative_files)
    num_validation_negative_files = int(0.2 * num_negative_files)

    for fname in negative_files[:num_train_negative_files]:
        src = os.path.join(original_directory, 'negative', fname)
        dst = os.path.join(train_negative_directory, fname)
        shutil.copyfile(src, dst)

    for fname in negative_files[num_train_negative_files:num_train_negative_files + num_validation_negative_files]:
        src = os.path.join(original_directory, 'negative', fname)
        dst = os.path.join(validation_negative_directory, fname)
        shutil.copyfile(src, dst)

    for fname in negative_files[num_train_negative_files + num_validation_negative_files:]:
        src = os.path.join(original_directory, 'negative', fname)
        dst = os.path.join(test_directory, fname)
        shutil.copyfile(src, dst)

    # Print the number of images in each directory as a sanity check
    print('Total training positive images:', len(os.listdir(train_positive_directory)))
    print('Total training negative images:', len(os.listdir(train_negative_directory)))
    print('Total validation positive images:', len(os.listdir(validation_positive_directory)))
    print('Total validation negative images:', len(os.listdir(validation_negative_directory)))
    print('Total test images:', len(os.listdir(test_directory)))

    return
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),  # L2 regularization
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=1e-4),
                  metrics=['accuracy'])
    return model
def repeat_generator(generator):
    while True:
        for data, labels in generator:
            yield data, labels
def train_model(model, train_dir, validation_dir):
    # Augmentation configuration for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Validation data should not be augmented
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        shuffle=True,
        seed=42,
        class_mode='binary')

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=32,
        shuffle=True,
        seed=42,
        class_mode='binary')

    train_generator = repeat_generator(train_generator)
    validation_generator = repeat_generator(validation_generator)

    steps_per_epoch = int(3547/32)
    validation_steps = int(1217/32)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=[early_stopping])

    return model, history

def evaluate_model(model, test_dir):
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        shuffle=False,
        seed=42,
        class_mode='binary')

    test_steps = int(test_generator.samples / test_generator.batch_size)

    test_loss, test_accuracy = model.evaluate(test_generator,
                                              steps=test_steps)
    print(f'Test accuracy: {test_accuracy:.4f}')
    print(f'Test loss: {test_loss:.4f}')

    return test_generator

def save_model(model, model_path):
    model.save(model_path)

def main():
    original_directory = 'faces_all'
    base_directory = 'faces_prepared'
    model_path = 'My_model.h5'

    prepare_dataset(original_directory, base_directory)

    model = create_model()

    train_dir = os.path.join(base_directory, 'train')
    validation_dir = os.path.join(base_directory, 'validation')
    test_dir = os.path.join(base_directory, 'test')

    model, history = train_model(model, train_dir, validation_dir)
    save_model(model, model_path)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))
    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

    # Evaluate the model on the test set
    evaluate_model(model, test_dir)

if __name__ == "__main__":
    main()
