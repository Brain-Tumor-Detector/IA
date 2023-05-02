from zipfile import ZipFile
from kaggle.api.kaggle_api_extended import KaggleApi
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def download_dataset(dataset, dst_path):
    # Descargamos el dataset
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset=dataset, path=dst_path)

def unzip(file, path):
    # Descomprimimos el dataset
    with ZipFile(file, 'r') as zObject:
        zObject.extractall(path=path)

def process_dataset(rescale, horizontal_flip, vertical_flip, validation_split, rotation_range):
    return ImageDataGenerator(
        rescale=rescale,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        validation_split=validation_split,
        rotation_range=rotation_range
    )

def generate_datasets(dataset, path, target_size, batch_size, color_mode, shuffle, class_mode):
    train_dataset = dataset.flow_from_directory(
        path,
        target_size = target_size,
        batch_size = batch_size,
        color_mode = color_mode,
        shuffle = shuffle,
        class_mode = class_mode,
        subset = 'training'
    )
    
    test_dataset = dataset.flow_from_directory(
        path,
        target_size = target_size,
        batch_size = batch_size,
        color_mode = color_mode,
        shuffle = not shuffle,
        class_mode = class_mode,
        subset = 'validation'
    )

    return train_dataset, test_dataset