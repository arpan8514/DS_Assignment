#------------------------------------------------------------------------------#
import os
import sys

import keras.backend as Kb
from keras.utils import to_categorical, get_file
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import initializers, optimizers
from keras.applications.densenet import DenseNet201
from keras.applications.vgg19 import VGG19
from keras.engine import Layer, InputSpec, get_source_inputs

from keras.layers import Input, Dense, Activation, Flatten, Reshape, Lambda
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, ZeroPadding2D
from keras.layers import AveragePooling2D, GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import add

import sys
sys.setrecursionlimit(3000)
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
WEIGHTS_PATH = 'https://github.com/adamcasson/resnet152/releases/download/v0.1/resnet152_weights_tf.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/adamcasson/resnet152/releases/download/v0.1/resnet152_weights_tf_notop.h5'
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
################################################################################
## Custom Layer for ResNet used for BatchNormalization.
## Learns a set of weights and biases used for scaling the input data.
## The output consists simply in an element-wise multiplication of the input
## and a sum of a set of constants: {out = in * gamma + beta},
## where 'gamma' and 'beta' are the weights and biases larned.
## Arguments:-
## 1.axis -- integer, axis along which to normalize in mode 0. For instance,
##           if your input tensor has shape (samples, channels, rows, cols),
##           set axis to 1 to normalize per feature map (channels axis).
## 2.momentum -- momentum in the computation of the exponential average 
##               of the mean and standard deviation of the data, for 
##               feature-wise normalization.
## 3.weights -- Initialization weights. List of 2 Numpy arrays, with shapes:
##              `[(input_shape,), (input_shape,)]`
## 4.beta_init -- name of initialization function for shift parameter. This 
##                parameter is only relevant when passing a `weights` argument.
## 5.gamma_init -- name of initialization function for scale parameter. This 
##                 parameter is only relevant when not passing a `weights` 
##                 argument.
################################################################################
class Scale(Layer):
    def __init__(self, weights=None, axis=-1, momentum = 0.9, beta_init='zero', 
                 gamma_init='one', **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializers.get(beta_init)
        self.gamma_init = initializers.get(gamma_init)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)

        self.gamma = Kb.variable(self.gamma_init(shape), 
                                name='%s_gamma'%self.name)
        self.beta = Kb.variable(self.beta_init(shape), name='%s_beta'%self.name)
        self.trainable_weights = [self.gamma, self.beta]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = Kb.reshape(self.gamma, broadcast_shape) * x + \
              Kb.reshape(self.beta, broadcast_shape)
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
################################################################################
## The identity_block is the block that has no conv layer at shortcut
## Arguments:-
## 1. input_tensor -- input tensor
## 2. kernel_size -- defualt 3, kernel size of middle conv layer at main path
## 3. filters -- list of integers, the nb_filters of 3 conv layer at main path
## 4. stage -- integer, current stage label, used for generating layer names
## 5. block -- 'a','b'..., current block label, used for generating layer names
################################################################################
def identity_block(input_tensor, kernel_size, filters, stage, block):
    eps = 1.1e-5
    
    if Kb.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', 
               use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, 
                           name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size), 
               name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, 
                           name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', 
               use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, 
                           name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    x = add([x, input_tensor], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
################################################################################
## conv_block is the block that has a conv layer at shortcut
## Arguments:-
## 1. input_tensor -- input tensor
## 2. kernel_size -- defualt 3, kernel size of middle conv layer at main path
## 3. filters -- list of integers, the nb_filters of 3 conv layer at main path
## 4. stage -- integer, current stage label, used for generating layer names
## 5. block -- 'a','b'..., current block label, used for generating layer names
## Note: From stage 3, the first conv layer at main path is with subsample=(2,2)
##       and the shortcut should have subsample=(2,2) as well
################################################################################
def conv_block(input_tensor, kernel_size, filters, stage, block, 
               strides=(2, 2)):
    eps = 1.1e-5
    
    if Kb.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a', 
               use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, 
                           name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size),
                      name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, 
                           name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', 
               use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, 
                           name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides, 
                      name=conv_name_base + '1', use_bias=False)(input_tensor)
    shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, 
                      name=bn_name_base + '1')(shortcut)
    shortcut = Scale(axis=bn_axis, name=scale_name_base + '1')(shortcut)

    x = add([x, shortcut], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def ResNet152(include_top=True, 
              weights=None,
              input_tensor=None, 
              input_shape=None,
              large_input=False, 
              pooling=None,
              classes=1000):

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
    
    eps = 1.1e-5
    
    if large_input:
        img_size = 448
    else:
        img_size = 224
    
    # Determine proper input shape
    input_shape = (224, 224, 3)
    print (input_shape)

    
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # handle dimension ordering for different backends
    if Kb.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
            
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
    x = Scale(axis=bn_axis, name='scale_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    for i in range(1,8):
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b'+str(i))

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    for i in range(1,36):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b'+str(i))

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    if large_input:
        x = AveragePooling2D((14, 14), name='avg_pool')(x)
    else:
        x = AveragePooling2D((7, 7), name='avg_pool')(x)
    
    # include classification layer by default, not included for feature extraction 
    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
    
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, x, name='resnet152')
    
    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('resnet152_weights_tf.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models',
                                    md5_hash='cdb18a2158b88e392c0905d47dcef965')
        else:
            weights_path = get_file('resnet152_weights_tf_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='4a90dcdafacbd17d772af1fb44fc2660')

        model.load_weights(weights_path, by_name=True)

    return model
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# dimensions of our images.
WIDTH = 224
HEIGHT = 224

if (Kb.image_data_format() == 'channels_first'):
    print ("CHANNELS_FIRST")
    input_shape = (3, WIDTH, HEIGHT)
else:
    print ("CHANNELS_LAST")
    input_shape = (WIDTH, HEIGHT, 3)
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
train_data_dir = "damage_loc_data_high_contrast_4/training"
validation_data_dir = "damage_loc_data_high_contrast_4/validation"

nb_train_samples = sum([len(files) for r, d, files in os.walk(train_data_dir)])
print (nb_train_samples)

nb_validation_samples = sum([len(files) for r, d, files in os.walk(
    validation_data_dir)])
print (nb_validation_samples)

NUM_CLASSES = sum([len(dirs) for r, dirs, files in os.walk(validation_data_dir)])
print (NUM_CLASSES)
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
input_Tensor = Input(shape=input_shape, name='Arpan_Input')

##-- Model ResNet 152 --------------------------------------------------------##
model_resnet152 = ResNet152(weights='imagenet', include_top=False, 
                            pooling='max')
model_resnet152_Tensor = model_resnet152(input_Tensor)

# Adding the fully-connected layers
resnet_fc1_Tensor = Dense(2048, activation='relu', 
                          name='resnet_FC-1')(model_resnet152_Tensor)

resnet_fc2_Tensor = Dense(1624, activation='relu', 
                          name='resnet_FC-2')(resnet_fc1_Tensor)
#------------------------------------------------------------------------------#

##-- Model DenseNet 201 ------------------------------------------------------##
model_densenet201 = DenseNet201(weights='imagenet', include_top=False, 
                                pooling='max')
model_densenet201_Tensor = model_densenet201(input_Tensor)

# Adding the fully-connected layers
densenet_fc1_Tensor = Dense(1624, activation='relu', 
                            name='densenet_FC-1')(model_densenet201_Tensor)
#------------------------------------------------------------------------------#

##-- Model VGG-19 ------------------------------------------------------------##
model_vgg19 = VGG19(include_top=False, weights='imagenet')

#Use the generated model
model_vgg19_output = model_vgg19(input_Tensor)

# Flattening the output, as the last layer of model_vgg19 is a MAXPOOL layer
vgg19_output_Tensor = Flatten(name='FLATTEN')(model_vgg19_output)

# Adding the fully-connected layers
vgg19_fc1_Tensor = Dense(4096, activation='relu', 
                         name='vgg19_FC-1')(vgg19_output_Tensor)
vgg19_fc2_Tensor = Dense(4096, activation='relu', 
                         name='vgg19_FC-2')(vgg19_fc1_Tensor)
vgg19_fc3_Tensor = Dense(2048, activation='relu', 
                         name='vgg19_FC-3')(vgg19_fc2_Tensor)
vgg19_final_Tensor  = Dense(1624, activation='relu', 
                         name='vgg19_final')(vgg19_fc3_Tensor)
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
merged_Tensor = add([resnet_fc2_Tensor, densenet_fc1_Tensor, 
                     vgg19_final_Tensor])

out_Tensor = Dense(1024, activation='relu', name='merged-FC-1')(merged_Tensor)
out_Tensor = Dense(512, activation='relu', name='merged-FC-2')(out_Tensor)
out_Tensor = Dense(256, activation='relu', name='merged-FC-3')(out_Tensor)
final_out_Tensor = Dense(NUM_CLASSES, activation='softmax',
                              name='final_output')(out_Tensor)
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
training_model = Model(inputs=input_Tensor, outputs=final_out_Tensor)
print (training_model.summary())
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
LEARNING_RATE = 1e-4

# compile the model
training_model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.SGD(lr=LEARNING_RATE, momentum=0.9),
        metrics=['accuracy'])
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
NUM_EPOCHS = 300
BATCH_SIZE = 16

#### Augmentation configuration for training
train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(WIDTH, HEIGHT),
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        class_mode='categorical')

#### Augmentation configuration for testing: only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(WIDTH, HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical')
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# Training
checkpoint = ModelCheckpoint(
        'resnet152_densenet201_vgg19_2_{epoch:03d}_{val_acc:.4f}.hdf5',
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        mode='auto')

print ("Model Training starts now:\n")
training_model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // BATCH_SIZE,
        epochs=NUM_EPOCHS,
        callbacks=[checkpoint],
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // BATCH_SIZE)
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
Kb.clear_session()
#------------------------------------------------------------------------------#

