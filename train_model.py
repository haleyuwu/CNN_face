# -*- coding: utf-8 -*-
"""
Train CNN 3 class: Thai Ha, Van Nam, My Trinh
Chỉ dùng CNN thuần, nhiều lớp + augmentation
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, models

# ==== CONFIG MẶC ĐỊNH ====
DATA_DIR = r"c:/New folder/lonai/faxe/data_root"   # sửa lại nếu cần
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 40
OUTPUT_DIR = "models"

def get_datasets(data_dir, img_size=128, batch_size=32, val_split=0.2, seed=42):
    # Tạo dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=(img_size, img_size),
        batch_size=batch_size
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=(img_size, img_size),
        batch_size=batch_size
    )
    # LƯU class_names TRƯỚC khi prefetch/shuffle
    class_names = train_ds.class_names

    AUTOTUNE = tf.data.AUTOTUNE
    # (tùy chọn) cache để nhanh hơn nếu RAM cho phép
    train_ds = train_ds.shuffle(1000).cache().prefetch(AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names

def build_cnn(input_shape, num_classes):
    model = models.Sequential(name="face_cnn")
    model.add(layers.Rescaling(1./255, input_shape=input_shape))

    # Data augmentation
    model.add(layers.RandomFlip("horizontal"))
    model.add(layers.RandomRotation(0.1))
    model.add(layers.RandomZoom(0.1))
    model.add(layers.RandomTranslation(0.1, 0.1))

    # CNN blocks
    model.add(layers.Conv2D(32, (3,3), activation='relu', padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3,3), activation='relu', padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3,3), activation='relu', padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (3,3), activation='relu', padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.4))

    # Dense head
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

def save_labels_txt(class_names, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, name in enumerate(class_names):
            f.write(f"{idx} {name}\n")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_ds, val_ds, class_names = get_datasets(DATA_DIR, IMG_SIZE, BATCH_SIZE)
    num_classes = len(class_names)
    input_shape = (IMG_SIZE, IMG_SIZE, 3)

    model = build_cnn(input_shape, num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(OUTPUT_DIR, "best.keras"),
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=6, restore_best_weights=True, monitor="val_accuracy"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=3, min_lr=1e-6, verbose=1
        )
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    # Save final model + labels
    model.save(os.path.join(OUTPUT_DIR, "keras_model.h5"))
    save_labels_txt(class_names, os.path.join(OUTPUT_DIR, "labels.txt"))

    print("\n✅ Training done. Model & labels saved in", OUTPUT_DIR)

if __name__ == "__main__":
    main()
