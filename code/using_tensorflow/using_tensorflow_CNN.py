#%% Environment
# package
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from os.path import dirname

# root path
script_path = os.path.abspath(__file__)
root_path = dirname(dirname(dirname(script_path)))
os.chdir(root_path)

# data path
train_path = os.path.join(root_path, 'data', 'training')
val_path = os.path.join(root_path, 'data', 'validation')

#%% Data
# DataLoader with tensorflow


def make_data_generator(train_path, val_path):
    """
    使用tensorflow的ImageDataGenerator, 快捷进行数据读取和初始化
    """
    tr_data_gen = ImageDataGenerator(
        rescale=1/255,  # 标准化
        # rotation_range=30,  # 随机旋转
        # width_shift_range=0.2,  # 随机纵向移动
        height_shift_range=0.2,  # 随机横向移动
        shear_range=0.12,  # 侧旋
        zoom_range=0.12,  # 放大缩小
        horizontal_flip=True,  # 水平翻转
        fill_mode='nearest',  # 增强后的数据填充
    )
    
    tr_data = tr_data_gen.flow_from_directory(
        train_path,  # 路径
        class_mode='binary',  # 模型类型-分类
        batch_size = 32,  # 每组数据量
        target_size = (150, 150),  # 固定图像像素大小
    )
    
    val_data_gen = ImageDataGenerator(rescale=1/255)
    val_data = val_data_gen.flow_from_directory(
        train_path,
        class_mode='binary',
        batch_size=32,
        target_size=(150, 150),
    )
    return tr_data, val_data


tr_data, val_data = make_data_generator(train_path, val_path)


#%% Model
def create_model():
    model = keras.models.Sequential([
        # Conv层
        # keras.layers.Conv2D(256, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        keras.layers.MaxPooling2D(2, 2),
        # keras.layers.Conv2D(128, (3, 3), activation='relu', ),
        keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        keras.layers.MaxPooling2D(2, 2),
        # Flatten
        keras.layers.Flatten(),
        # Dense
        # keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid'),
        ]
    )
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


model = create_model()
#%% Model Describe
model.summary()
#%% Make callback


class MyCallback(tf.keras.callbacks.Callback):
    """
    继承tf.keras.callbacks.Callback来实现停止训练功能
    """
    def on_epoch_end(self, epoch, logs={}):
        """
        在每次迭代后检测logs中的内容
        """
        if logs.get('accuracy') > 0.99 and logs.get('val_accuracy'):
            self.model.stop_training = True

mycallback = MyCallback()
#%% Train and validate
def model_train(model, epoch=5):
    model.fit(
        tr_data,
        epochs=epoch,
        validation_data=val_data,
        callbacks=mycallback
    )
    return model

model = model_train(model)
#%% Model Analyse
"""
TODO LIST:
    1. 模型每层效果展示
    2. history分析
"""

#%% Transform Learning

from tensorflow.keras.applications.inception_v3 import InceptionV3

local_weights_file = os.path.join(root_path, 'pre_trained_weights', 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

def create_pre_trained_model(local_weights_file):
    pre_trained_model = InceptionV3(input_shape=(150, 150, 3),  # 定义待训练数据大小
                                    include_top=False, 
                                    weights=None)
    
    pre_trained_model.load_weights(local_weights_file)
    
    # 初始化禁止预训练模型weights进行更新
    for layer in pre_trained_model.layers:
        layer.trainable = False
    
    return pre_trained_model

pre_trained_model = create_pre_trained_model(local_weights_file)

pre_trained_model.summary()

