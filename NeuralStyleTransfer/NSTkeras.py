from keras.applications.vgg19 import VGG19
from keras.layers import Dense, AveragePooling2D, MaxPooling2D
from keras.models import Model


IMG_HEIGHT = 512
IMG_WIDTH = 512

def prepare_vgg_model(img_shape=(IMG_HEIGHT, IMG_WIDTH, 3), pool_type='avg'):
  def replace_max_to_ave_pooling(model):
    x = model.layers[0].output
    avgpooli = 1;
    for layer in model.layers[1:]:
      if type(layer) == MaxPooling2D and pool_type='avg':
        x = AveragePooling2D(pool_size=(2, 2),
                             padding='SAME',
                             name='block{}_pool'.format(avgpooli))(x)
        avgpooli += 1
      else:
        x = layer(x)
    return Model(inputs=model.layers[0].input, outputs=x)

  base_model = VGG19(weights='imagenet',
                     include_top=False,
                     input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

  return replace_max_to_ave_pooling(base_model)

def main():
  vgg_model = prepare_vgg_model()
  vgg_model.summary()


if __name__ == '__main__':
	main()
