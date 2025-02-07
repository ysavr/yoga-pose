# references: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/image_classifier.ipynb

import os
import tensorflow as tf

assert tf.__version__.startswith("2")

from mediapipe_model_maker import image_classifier

import matplotlib.pyplot as plt

# image_path = tf.keras.utils.get_file(
#     "flower_photos.tgz",
#     "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
#     extract=True,
# )
image_path = os.path.join("data", "images_processed")
print(image_path)
labels = []
for i in os.listdir(image_path):
    if os.path.isdir(os.path.join(image_path, i)):
        labels.append(i)
print(labels)

NUM_EXAMPLES = 4

for label in labels:
    label_dir = os.path.join(image_path, label)
    example_filenames = os.listdir(label_dir)[:NUM_EXAMPLES]
    fig, axs = plt.subplots(1, NUM_EXAMPLES, figsize=(10, 2))
    for i in range(NUM_EXAMPLES):
        axs[i].imshow(plt.imread(os.path.join(label_dir, example_filenames[i])))
        axs[i].get_xaxis().set_visible(False)
        axs[i].get_yaxis().set_visible(False)
    fig.suptitle(f"Showing {NUM_EXAMPLES} examples for {label}")

plt.show()

data = image_classifier.Dataset.from_folder(image_path)
train_data, remaining_data = data.split(0.8)
test_data, validation_data = remaining_data.split(0.5)

spec = image_classifier.SupportedModels.EFFICIENTNET_LITE4
hparams = image_classifier.HParams(export_dir="exported_model_lite_4")
options = image_classifier.ImageClassifierOptions(supported_model=spec, hparams=hparams)

model = image_classifier.ImageClassifier.create(
    train_data=train_data,
    validation_data=validation_data,
    options=options,
)

loss, acc = model.evaluate(test_data)
print(f"Test loss:{loss}, Test accuracy:{acc}")

model.export_model(
    model_name="poseLite4.tflite",
)
