import tensorflow.keras as kr
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def generate_model(heigth, width, optimizer, loss):
    model = kr.Sequential([
    # Capa 1 (Entrada)
    kr.layers.Conv2D(32, activation='relu', kernel_size=(3, 3), input_shape=(heigth, width, 1)),
    kr.layers.MaxPooling2D(2,2),

    # Capa 2
    kr.layers.Conv2D(64, activation='relu', kernel_size=(3, 3)),
    kr.layers.MaxPooling2D(2,2),

    # Capa 3
    kr.layers.Conv2D(128, activation='relu', kernel_size=(3, 3)),
    kr.layers.MaxPooling2D(2,2),

    # Capa 4
    kr.layers.Flatten(),
    kr.layers.Dropout(0.5),

    # Capa 5
    kr.layers.Dense(512, activation='relu'),
    kr.layers.Dropout(0.2),

    # Capa 6 (salida)
    kr.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer = optimizer,
        metrics = ['accuracy'],
        loss = loss
    )
    return model

def train_model(model, path, name, train_data, epochs, validation_data):
    # Establecemos los checkpoints
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint('./model.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    
    return model.fit(
        train_data, 
        epochs=epochs, 
        callbacks=[early_stopping, checkpoint], 
        validation_data=validation_data
        )