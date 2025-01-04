import numpy as np
import PIL.Image as Image

def load_image(image_path):
    image = Image.open(image_path)
    image = image.resize((28, 28))
    image = np.array(image)
    return image

def infer(model, image_path):
    image = load_image(image_path)
    image = image.reshape(1, 28, 28, 1)
    output = model.forward(image)
    return np.argmax(output)


