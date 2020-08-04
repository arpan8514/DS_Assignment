import os
import numpy

from keras.optimizers import SGD
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as Kb
from keras import initializers
from keras.engine import Layer, InputSpec
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
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
#test_dir = "Test_Data_All_4_10Classes"
#test_dir = "Test_Data_All_5_10Classes"
#test_dir = "Test_Data_All_sample_10Classes"
test_dir = "Test_Data_All_6_10Classes"

text_labels = ["Bumper_Dent", "Door_Dent", "Fender_Dent", "Glass_crack", 
               "Glass_shatter", "HeadLampDamage", "No_Damage", "Scratch", 
               "Smashes", "TailLampDamage"]
#print (text_labels)

SKIP_FILES = {'cmds'}
WIDTH = 224
HEIGHT = 224
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
#### Prediction:
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(WIDTH, HEIGHT),
        color_mode="rgb",
        class_mode='categorical',
        batch_size=1,
        shuffle=False)

filenames = test_generator.filenames
nb_samples = len(filenames)
#print (nb_samples)


# loading the weight file
test_model = load_model("resnet152_densenet201_vgg19_2_055_0.8937.hdf5",
                        custom_objects={'Scale': Scale})

test_model.compile(loss='categorical_crossentropy',
                   optimizer=SGD(lr=1e-4, momentum=0.9),
                   metrics=['accuracy'])

#predict = test_model.predict_generator(test_generator, steps=nb_samples)
prediction_array = test_model.predict_generator(test_generator)
#print (len(prediction_array))

for i in range (0, len(prediction_array)):
    result = prediction_array[i]
    #print (result)
    answer = numpy.argmax(result)
    #print (answer)
    
    print ("Test Sample: ", int(i+1))
    print ("Original File: ", test_generator.filenames[i])

    predicted_label = text_labels[answer]
    print("Predicted label: " + predicted_label + "\n")
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# It may happen: model.fit did not end the session cleanly. So, clearing the
# session manually might be required.
# example of session object: which can raise error
# <tensorflow.python.client.session.Session object at 0x7f0896aabe48>
Kb.clear_session()
#------------------------------------------------------------------------------#


