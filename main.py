import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras import layers, models, optimizers

# Path to your dataset directory
data_dir = r""

image_size = (224, 224)

num_classes = 4

batch_size = 32

num_total_samples = 4217 

data_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
files = os.listdir(data_dir)
train_files, validation_files = train_test_split(files, test_size=0.2, random_state=42)

train_generator = data_gen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=False  
)

validation_generator = data_gen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False  
)

base_model = MobileNetV2(input_shape=image_size + (3,), include_top=False, weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(num_classes, activation='softmax')
])
base_model.trainable = False
model.compile(
    optimizer=optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

print("Number of training files:", len(train_files))
print("Number of validation files:", len(validation_files))

print("Class indices (training):", train_generator.class_indices)
print("Class indices (validation):", validation_generator.class_indices)

print("Steps per epoch:", len(train_generator))
print("Validation steps:", len(validation_generator))

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=25,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)
model.save('model.h5')

val_loss, val_accuracy = model.evaluate(validation_generator)
print("Validation Accuracy: {:.2f}%".format(val_accuracy * 100))
