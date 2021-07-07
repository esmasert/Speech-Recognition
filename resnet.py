
import tensorflow as tf
import tensorflow.python.keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.python.keras.layers import AveragePooling2D, Input, Flatten, add, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, Callback
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers

######################### TEST MODEL #####################################
def testmodel(input, numclasses):

    x =Conv2D(32,(3, 3),padding ='same')(input)
    x =Conv2D(32,(3, 3),padding ='same')(x)
    x =Conv2D(32,(3, 3),padding ='same')(x)
    x = BatchNormalization()(x, training=True)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x =Conv2D(64,(3, 3))(x)
    x =Conv2D(64,(3, 3))(x)
    x =Conv2D(64,(3, 3))(x)
    x = BatchNormalization()(x, training=True)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x =Conv2D(128,(3, 3))(x)
    x =Conv2D(128,(3, 3))(x)
    x =Conv2D(128,(3, 3))(x)
    x = BatchNormalization()(x, training=True)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x =Conv2D(256,(3, 3))(x)
    x =Conv2D(256,(3, 3))(x)
    x =Conv2D(256,(3, 3))(x)
    x = BatchNormalization()(x, training=True)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x =Conv2D(512,(3, 3))(x)
    x =Conv2D(512,(3, 3))(x)
    x =Conv2D(512,(3, 3))(x)
    x = BatchNormalization()(x, training=True)

    x = Flatten()(x)

    x = Dense(1024,activation='relu')(x)
    x = Dense(numclasses,activation='softmax')(x)

    return x




############################### Better implementation ##########################

#
# image dimensions
#

img_height = 256
img_width = 256
img_channels = 3

#
# network params
#

cardinality = 32


def residual_network(x):
    """
    ResNeXt by default. For ResNet set `cardinality` = 1 above.

    """
    def add_common_layers(y):
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)

        return y

    def grouped_convolution(y, nb_channels, _strides):
        # when `cardinality` == 1 this is just a standard convolution
        if cardinality == 1:
            return layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)

        assert not nb_channels % cardinality
        _d = nb_channels // cardinality

        # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
        # and convolutions are separately performed within each group
        groups = []
        for j in range(cardinality):
            group = layers.Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
            groups.append(layers.Conv2D(_d, kernel_size=(3, 3), strides=_strides, padding='same')(group))

        # the grouped convolutional layer concatenates them as the outputs of the layer
        y = layers.concatenate(groups)

        return y

    def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
        """
        Our network consists of a stack of residual blocks. These blocks have the same topology,
        and are subject to two simple rules:
        - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
        - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
        """
        shortcut = y

        # we modify the residual building block as a bottleneck design to make the network more economical
        y = layers.Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        y = add_common_layers(y)

        # ResNeXt (identical to ResNet when `cardinality` == 1)
        y = grouped_convolution(y, nb_channels_in, _strides=_strides)
        y = add_common_layers(y)

        y = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        # batch normalization is employed after aggregating the transformations and before adding to the shortcut
        y = layers.BatchNormalization()(y)

        # identity shortcuts used directly when the input and output are of the same dimensions
        if _project_shortcut or _strides != (1, 1):
            shortcut = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        y = layers.add([shortcut, y])

        # relu is performed right after each batch normalization,
        # expect for the output of the block where relu is performed after the adding to the shortcut
        y = layers.LeakyReLU()(y)

        return y

    # conv1
    x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
    x = add_common_layers(x)

    # conv2
    x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    for i in range(3):
        project_shortcut = True if i == 0 else False
        x = residual_block(x, 128, 256, _project_shortcut=project_shortcut)

    # conv3
    for i in range(4):
        # down-sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 256, 512, _strides=strides)

    # conv4
    for i in range(6):
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 512, 1024, _strides=strides)

    # conv5
    for i in range(3):
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 1024, 2048, _strides=strides)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(3,activation='softmax')(x)

    return x


#############################################################################################
