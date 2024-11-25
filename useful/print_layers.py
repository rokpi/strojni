import tensorflow as tf
#can see from directories which models apply for exceptions

def print_layers(model_path):
    model = tf.keras.models.load_model(model_path, compile = False)
    for layer in model.layers:
        print(f"Layer name: {layer.name}, Input shape: {layer.input_shape}")
    print("Konec...")

if __name__ == '__main__':
  print_layers('/home/rokp/test/models/mtcnn/vgg-vgg/vgg-vgg.mtcnn.conv4_3.20230124-204737.hdf5')

  