from gooey import Gooey, GooeyParser
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import sys

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5', custom_objects={'KerasLayer': hub.KerasLayer})

@Gooey(progress_regex=r"\d+/\d+ Complete")
def s():
    parser = GooeyParser()
    parser.add_argument('Image', help='Upload an image\nThe AI can only recognize: \n1. Cataracts\n2. Diabetic retinopathy\n3. Glaucoma\n4. Normal eyes', widget='FileChooser')
    args = parser.parse_args()

    result = classify(args.Image)


    print(f"Image Classification Result: {result}")

def classify(img_path):
    size = (224, 224)
    
    # Load the image using PIL
    image = Image.open(img_path)
    image = image.resize(size)
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = image_array / 255.0
    image_array = tf.expand_dims(image_array, axis=0)

    # Make the prediction
    predictions = model.predict(image_array)
    predicted_class_idx = tf.argmax(predictions, axis=1)[0]

    categories = {0: 'Cataract', 1: 'Diabetic Retinopathy', 2: 'Glaucoma', 3: 'Normal'}
    category = categories.get(int(predicted_class_idx), 'Unknown')

    return category

if __name__ == "__main__":
    if len(sys.argv) > 1:
        s()
    else:
        s()
