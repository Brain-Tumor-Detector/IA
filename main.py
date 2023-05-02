from src.dataset import *
from src.model import *

import configparser

if __name__ == '__main__':
    # Obtenemos el fichero de configuración
    CONFIG_FILE_NAME = "./config.ini"
    config = configparser.RawConfigParser(allow_no_value=True)
    config.read(CONFIG_FILE_NAME)

    # Descargamos el dataset
    download_dataset(config['DATASET']['dataset_url'], '.')

    # Descomprimimos el dataset
    unzip(config['DATASET']['dataset_name_zip'], '.')

    # Generamos los datasets ya procesados
    dataset = process_dataset(
        rescale = config['DATA_AUGMENTATION']['RESCALE'],
        horizontal_flip = config['DATA_AUGMENTATION']['HORIZONTAL_FLIP'],
        vertical_flip = config['DATA_AUGMENTATION']['VERTICAL_FLIP'],
        validation_split = float(config['DATA_AUGMENTATION']['VALIDATION_SPLIT']),
        rotation_range = int(config['DATA_AUGMENTATION']['ROTATION_RANGE'])
    )

    # Generamos el dataset de entrenamiento y validación
    train, test = generate_datasets(
        dataset = dataset,
        path = config['PATHS']['IN_TRAIN_DATASET'],
        target_size = (config['DATASET_CONFIG']['PHOTO_HEIGHT'], config['DATASET_CONFIG']['PHOTO_WIDTH']), 
        batch_size = int(config['DATASET_CONFIG']['BATCH_SIZE']),
        color_mode = config['DATASET_CONFIG']['COLOR_MODE'],
        shuffle = config['DATASET_CONFIG']['SHUFFLE'],
        class_mode = config['DATASET_CONFIG']['CLASS_MODE'],
    )

    # Generamos el modelo
    model = generate_model(
        heigth = int(config['DATASET_CONFIG']['PHOTO_HEIGHT']),
        width = int(config['DATASET_CONFIG']['PHOTO_WIDTH']),
        optimizer = config['MODEL_CONFIG']['OPTIMIZER'],
        loss = config['MODEL_CONFIG']['LOSS']
    )
    print(model.summary())

    # Entrenamos el modelo
    result = train_model(
        model = model,
        path = config['PATHS']['MODEL'],
        name = config['MODEL_CONFIG']['NAME'],
        train_data = train,
        epochs = int(config['MODEL_CONFIG']['EPOCHS']),
        validation_data = test
    )

    # Guardamos las estadísticas del modelo