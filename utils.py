import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from keras_vggface import utils
from keras_vggface.vggface import VGGFace
from mtcnn import MTCNN

#import adaptive

detector = MTCNN()

def detect_face(img):
  # Detect faces in the input image
  faces = detector.detect_faces(img)
  # If there are multiple faces, keep only the face with the largest detection box
  if len(faces) > 0:
    face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
    x, y, w, h = face['box']
    x, y, w, h = int(x), int(y), int(w), int(h)
    side = max(w, h)
    x = x - (side - w) // 2
    y = y - (side - h) // 2
    detection_box = [x, y, side]
    #img_cropped = img[y:y+side, x:x+side]
  else:
    detection_box = False
  return detection_box

def change_model(model, new_input_shape=(None, 50, 50, 3),custom_objects=None):
  # replace input shape of first layer
  config = model.layers[0].get_config()
  config['batch_input_shape']=new_input_shape
  model._layers[0]=model.layers[0].from_config(config)
  # rebuild model architecture by exporting and importing via json
  new_model = tf.keras.models.model_from_json(model.to_json(),custom_objects=custom_objects)
  # copy weights from old model to new one
  for layer in new_model._layers:
    try:
      layer.set_weights(model.get_layer(name=layer.name).get_weights())
      print("Loaded layer {}".format(layer.name))
    except:
      print("Could not transfer weights for layer {}".format(layer.name))
  return new_model

def preproc_img(img_path, encoder_type, img_size):
  # Resize to target size
  img = image.load_img(img_path, target_size=img_size) #for Python < 3.10: tf.keras.utils. -> image.

  # Pad image to square size
  #img = image.load_img(img_path)
  #img = image.img_to_array(img)
  #old_size = img.shape[:2]  # old_size is in (height, width) format
  #ratio = float(img_size[0]) / max(old_size)
  #new_size = tuple([int(x * ratio) for x in old_size])
  #img = cv2.resize(img, (new_size[1], new_size[0]))
  #delta_w = img_size[0] - new_size[1]
  #delta_h = img_size[0] - new_size[0]
  #top, bottom = delta_h // 2, delta_h - (delta_h // 2)
  #left, right = delta_w // 2, delta_w - (delta_w // 2)
  ##import ipdb; ipdb.set_trace()
  ##img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE)
  #img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, value = 0)
  
  img_arr = image.img_to_array(img)
  img_arr = np.expand_dims(img_arr, axis=0)
  #import ipdb; ipdb.set_trace()
  if encoder_type == 'vgg' or encoder_type == 'facenet':
    #img_arr = preprocess_input(img_arr)
    img_arr = utils.preprocess_input(img_arr, version=1)
  elif encoder_type == 'resnet' or encoder_type == 'WeidiXie':
    img_arr = utils.preprocess_input(img_arr, version=2)
  elif encoder_type == 'mobilenet':
    img_arr = (img_arr - 127.5) * 0.0078125 #https://github.com/leondgarse/Keras_insightface/issues/54
  #elif encoder_type == 'facenet':
  #  mean, std = img_arr.mean(), img_arr.std()
  #  img_arr = (img_arr - mean) / std
  elif encoder_type == 'arcface':
      # ArcFace normalizes images to [-1, 1]
      img_arr = (img_arr / 127.5) - 1.0
  else:
    raise NameError('Unknown encoder_type argumet in image preprocessing routine.')
  return img_arr

def restore_original_image_from_array(x, encoder_type):
  if encoder_type == 'mobilenet':
    x = x/0.0078125
  # https://github.com/rcmalli/keras-vggface/blob/master/keras_vggface/utils.py
  if encoder_type == 'vgg' or encoder_type == 'facenet':
    pix_mean = [93.5940, 104.7624, 129.1863] #tf.keras.applications.VGG16: [103.939, 116.779, 123.68]
  elif encoder_type == 'resnet' or encoder_type == 'WeidiXie':
    pix_mean = [91.4953, 103.8827, 131.0912]
  elif encoder_type == 'mobilenet':
    pix_mean = [127.5, 127.5, 127.5]
  elif encoder_type == 'arcface':
    # Reverse ArcFace normalization
    x = (x + 1.0) * 127.5
    return x  # ArcFace uses RGB format, no need to reverse channels
  else:
    raise NameError('Unknown encoder_type argumet in image restoration routine.')
  x[..., 0] += pix_mean[0]
  x[..., 1] += pix_mean[1]
  x[..., 2] += pix_mean[2]
  if encoder_type != 'mobilenet':
    x = x[..., ::-1]
  return x

def get_loss(perceptual_model, layers, loss_type=['pixel', 'gradient', 'perceptual'], loss_weights=[1,1,1]):
  # Model definition for perceptual loss
  if 'perceptual' in loss_type:
    # Model definition for local perceptual loss
    if perceptual_model == 'vgg':
      #vgg = tf.keras.applications.VGG16(
      #  weights="imagenet",
      #  input_shape=(224, 224, 3),
      #  include_top=False,
      #  pooling='max')
      vgg = VGGFace(model='vgg16', weights='vggface', include_top=False)
      #with open('vgg16_summary.txt', 'w') as f:
      #  vgg.summary(print_fn=lambda x: f.write(x + '\n'))
      outputs = [vgg.get_layer(layer).output for layer in layers]
      m = Model([vgg.input], outputs)
    elif perceptual_model == 'resnet':
      #import ipdb; ipdb.set_trace()
      resnet = VGGFace(model='resnet50', weights='vggface', include_top=False)
      outputs = [resnet.get_layer(layer).output for layer in layers]
      m = Model([resnet.input], outputs)
    elif perceptual_model == 'mobilenet':
      mobilenet = tf.keras.models.load_model('data/models/keras_mobilenet_emore_adamw_5e5_soft_baseline_before_arc_E80_BTO_E2_arc_sgdw_basic_agedb_30_epoch_119_0.959333.h5', compile=False)
      outputs = [mobilenet.get_layer(layer).output for layer in layers]
      m = Model([mobilenet.input], outputs)
    if 'local' in loss_type:
      ndim = 100
      m_loc = tf.keras.Sequential()
      m_loc.add(tf.keras.layers.ZeroPadding2D((int((224-ndim)/2),int((224-ndim)/2)), input_shape=(ndim, ndim, 3)))
      m_loc.add(m)
      left_top = int(np.ceil((224 - ndim) / 2))
      right_bottom = 224 - int(np.floor((224 - ndim) / 2))
  #adaptive_lossfun = adaptive.AdaptiveImageLossFunction(
  #  image_size=[224,224,3], float_dtype=np.float32)
  def loss(true_img, pred_img):
    #true_img = true_img/0.0078125 + 127.5#[-1, 1] -> [0, 255]
    #pred_img = pred_img/0.0078125 + 127.5 #[-1, 1] -> [0, 255]
    #true_img = true_img[..., ::-1]
    #pred_img = pred_img[..., ::-1]
    #true_img = true_img - [93.5940, 104.7624, 129.1863]
    #pred_img = pred_img - [93.5940, 104.7624, 129.1863]
    total_loss = 0
    # Pixel loss
    if 'pixel' in loss_type:
      pixel_mse = tf.reduce_mean(tf.square(true_img - pred_img))
      #pixel_mse = tf.reduce_mean(adaptive_lossfun(tf.square(true_img - pred_img)))
      total_loss += pixel_mse * float(loss_weights[loss_type.index('pixel')])
      #tf.keras.backend.print_tensor(pixel_mse * float(loss_weights[loss_type.index('pixel')]), message='pixel_loss = ')
    # Gradient loss
    if 'gradient' in loss_type:
      dy_true, dx_true = tf.image.image_gradients(true_img)
      dy_pred, dx_pred = tf.image.image_gradients(pred_img)
      grad_mse = tf.reduce_mean(tf.square(dy_pred - dy_true)) + tf.reduce_mean(tf.square(dx_pred - dx_true))
      total_loss += grad_mse * float(loss_weights[loss_type.index('gradient')])
      #tf.keras.backend.print_tensor(grad_mse * float(loss_weights[loss_type.index('gradient')]), message='gradient_loss = ')
    # Perceptual loss
    if 'perceptual' in loss_type:
      true_feats = m(true_img)
      pred_feats = m(pred_img)
      if isinstance(true_feats, list):
        feat_mses = [tf.reduce_mean(tf.square(true - pred)) for true,pred in zip(true_feats, pred_feats)]
      else:
        feat_mses = [tf.reduce_mean(tf.square(true_feats - pred_feats))]
      for feat_loss in feat_mses:
        total_loss += feat_loss * float(loss_weights[loss_type.index('perceptual')])
        #tf.keras.backend.print_tensor(feat_loss * float(loss_weights[loss_type.index('perceptual')]), message='perceptual_loss = ')
    # Local losses
    if 'local' in loss_type:
      # Local pixel loss
      true_img_loc = true_img[:,left_top:right_bottom,left_top:right_bottom,:]
      pred_img_loc = pred_img[:,left_top:right_bottom,left_top:right_bottom,:]
      if 'pixel' in loss_type:
        pixel_mse_loc = tf.reduce_mean(tf.square(true_img_loc - pred_img_loc))
        total_loss += pixel_mse_loc * float(loss_weights[loss_type.index('pixel')])
      # Local gradient loss
      if 'gradient' in loss_type:
        dy_true_loc, dx_true_loc = tf.image.image_gradients(true_img_loc)
        dy_pred_loc, dx_pred_loc = tf.image.image_gradients(pred_img_loc)
        grad_mse_loc = tf.reduce_mean(tf.square(dy_pred_loc - dy_true_loc)) + tf.reduce_mean(tf.square(dx_pred_loc - dx_true_loc))
        total_loss += grad_mse_loc * float(loss_weights[loss_type.index('gradient')])
      # Local perceptual loss
      if 'perceptual' in loss_type:
        true_feats_loc = m_loc(true_img_loc)
        pred_feats_loc = m_loc(pred_img_loc)
        if isinstance(true_feats_loc, list):
          feat_mses_loc = [tf.reduce_mean(tf.square(true - pred)) for true,pred in zip(true_feats_loc, pred_feats_loc)]
        else:
          feat_mses_loc = [tf.reduce_mean(tf.square(true_feats_loc - pred_feats_loc))]
        for feat_loss_loc in feat_mses_loc:
          total_loss += feat_loss_loc * float(loss_weights[loss_type.index('perceptual')])
    return total_loss

  return loss
