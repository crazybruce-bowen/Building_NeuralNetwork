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

# Data 生成器，可生成
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
    class_model='binary',
    batch_size=32,
    target_size=(150, 150),
)

#%% Model
model = keras.models.Sequential([
    # Conv层
    keras.layers.Conv2D(256, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(128, (3, 3), activation='relu', ),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(256, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    keras.layers.MaxPooling2D(2, 2),
    # Flatten
    keras.layers.Flatten(),
    # Dense
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
    ]
)

model.complie(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#%% Train and validate
model.fit(
    tr_data
)
#%% Pre