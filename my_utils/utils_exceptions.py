import tensorflow as tf
#can see from directories which models apply for exceptions
def exception_layer_encoder(model_path):
    #print_layers(model_path)
    layer_name = None
    if model_path in ['/home/rokp/test/models/uncroped/resnet-resnet/resnet-resnet.hdf5', '/home/rokp/test/models/uncroped/resnet-vgg/resnet-vgg.hdf5']:
        layer_name = 'lyr_174'
    return layer_name

def exception_layer_decoder(model_path):
    #print_layers(model_path)
    layer_name = None
    if model_path in ['/home/rokp/test/models/uncroped/resnet-resnet/resnet-resnet.hdf5', '/home/rokp/test/models/uncroped/resnet-vgg/resnet-vgg.hdf5']:
        layer_name = 'lyr_174'
    return layer_name

def exception_transform(model_path):
    transform = None
    if model_path in ['/home/rokp/test/models/uncroped/resnet-resnet/resnet-resnet.hdf5', '/home/rokp/test/models/uncroped/resnet-vgg/resnet-vgg.hdf5']:
        transform = (-1, 2048)
    return transform

def encoder_layer(model, encoder_type, model_path):
    layer_name = exception_layer_encoder(model_path)

    if layer_name:
        embedding_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    elif encoder_type == 'vgg':      
        embedding_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer('fc6').output)
    elif encoder_type == 'arcface':       
        #embedding_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer('dense_1').output)
        outputs=model.get_layer('tf.image.resize').output
        resnet_model = model.get_layer('ResNet34')

        resnet_input = resnet_model.input  
        resnet_output = resnet_model.output  

        embedding_model1 = tf.keras.models.Model(inputs=model.input, outputs=outputs)
        embedding_model2 = tf.keras.models.Model(inputs=resnet_input, outputs=resnet_output)
        x = embedding_model1.output
        x = embedding_model2(x)
        embedding_model = tf.keras.models.Model(inputs=embedding_model1.input, outputs = x)

    elif encoder_type == 'resnet':
        embedding_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer('avg_pool').output)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    return embedding_model

def decoder_layer(model, model_path, encoder_type):
    layer_name = exception_layer_decoder(model_path)
    if layer_name:
        decoder_input = model.get_layer(layer_name).output
        decoder_output = model.output
    elif encoder_type == 'vgg':
        decoder_input = model.get_layer('fc6').output
        decoder_output = model.output
    elif encoder_type == 'arcface':
        resnet_model = model.get_layer('ResNet34')
        decoder_layers = model.layers[model.layers.index(resnet_model) + 1:]
        decoder_input = decoder_layers[0].input
        decoder_output = decoder_layers[-1].output
    elif encoder_type == 'resnet':
        decoder_input = model.get_layer('avg_pool').output    
        decoder_output = model.output   
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    

    decoder_model = tf.keras.models.Model(inputs = decoder_input, outputs = decoder_output)
    return decoder_model

def define_tranform(trans_embedd, encoder_type, model_path):
    transform = exception_transform(model_path)
    if transform:
        trans_embedd = trans_embedd.reshape(transform)                
    elif encoder_type == 'vgg':
        trans_embedd = trans_embedd.reshape(1, -1)                              
    elif encoder_type == 'resnet':
        trans_embedd = trans_embedd.reshape(-1, 1, 1, 2048) 
    elif encoder_type == 'arcface':
        trans_embedd = trans_embedd.reshape(1, -1)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    return trans_embedd