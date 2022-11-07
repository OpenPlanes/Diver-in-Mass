
import tensorflow as tf
from tensorflow.python.ops.linalg_ops import norm
print("TensorFlow version: ", tf.__version__)

import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image, ImageDraw
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, BatchNormalization, MaxPool2D

# Create emojis dictionary
emojis = {
    0: {'name': 'none'},
    1: {'name': 'diver', 'file': '1F642.png'},
    2: {'name': 'fish', 'file': '1F602.png'},
    3: {'name': 'rock', 'file': '1F928.png'}
}

# Add the images to the emoji dictionary
for class_id, values in emojis.items():
    if (values['name'] != 'none'):
        png_file = Image.open(os.path.join('emojis', values['file'])).convert('RGBA')
        png_file.load()
        new_file = Image.new("RGB", png_file.size, (255, 255, 255))
        new_file.paste(png_file, mask=png_file.split()[3])
        emojis[class_id]['image'] = new_file

# Show all the emojis
def plot_emojis():
    plt.figure(figsize=(9,9))
    for i, (j, e) in enumerate(emojis.items()):
        plt.subplot(3,3,i+1)
        plt.imshow(plt.imread(os.path.join('emojis', e['file'])))
        plt.xlabel(e['name'])
        plt.xticks([])
        plt.yticks([])
    plt.show()

# Create example (place a random emoji to a random position in the image)
def create_example():
    class_id = np.random.randint(0, len(emojis))
    image = np.ones((144,144,3)) * 255

    # If class is not none, then include the graphic (with some rotation and stretch)
    if (class_id > 0):
        row = np.random.randint(0, 72)
        col = np.random.randint(0, 72)
        image[row: row+72, col: col+72, :] = np.array(emojis[class_id]['image'])

    return image.astype('uint8'), class_id, (row + 10) / 144, (col + 10) / 144

# Plot bounding boxes
def plot_bounding_box(image, gt_coords, pred_coords=[],norm=False):
    if norm:
        image *= 255.
        image = image.astype('uint8')
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    row, col = gt_coords
    row *= 144
    col *= 144
    draw.rectangle((col, row, col+52, row+52), outline='green', width=3)

    if len(pred_coords) > 0:
        row,col = pred_coords
        row *= 144
        col *= 144
        draw.rectangle((col, row, col+52, row+52), outline='red', width=3)
    
    return image

# Create a data generator
def data_generator(batch_size = 6):
    while True:
        x_batch = np.zeros((batch_size, 144, 144, 3))
        y_batch = np.zeros((batch_size, 9))
        bbox_batch = np.zeros((batch_size,2))

        for i in range(0, batch_size):
            image, class_id, row, col = create_example()
            x_batch[i] = image/255.
            y_batch[i, class_id] = 1.0
            bbox_batch[i] = np.array([row,col])

        yield {'image': x_batch}, {'class_out' : y_batch, 'box_out': bbox_batch}

# Create the CNN
input_ = Input(shape=(144,144,3),name='image')

# 5 convolution blocks (=layers?)
x = input_
for i in range(0,5):
    n_filters = 2**(4+i)
    x = Conv2D(n_filters, 3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(2)(x)

x = Flatten()(x)
x = Dense(256, activation='relu')(x)

class_out = Dense(9, activation='softmax', name='class_out')(x)
box_out = Dense(2, name='box_out')(x)

model = tf.keras.models.Model(input_, [class_out, box_out])
model.summary()

# Custom Metric: IoU (intersection over union)
class IoU(tf.keras.metrics.Metric):
    def get_initial_states(self):
            iou = self.add_weight(name='iou', initializer='zeros')
            total_iou = self.add_weight(name='total_iou', initializer='zeros')
            num_ex = self.add_weight(name='num_ex', initializer='zeros')
            return iou, total_iou, num_ex

    def __init__(self, **kwargs):
        super(IoU, self).__init__(**kwargs)
        self.iou, self.total_iou, self.num_ex = self.get_initial_states()

    def update_state(self, y_true, y_pred, sample_weight=None):
        def get_box(y):
            rows, cols = y[:,0], y[:,1]
            rows, cols = rows*144, cols*144
            y1, y2 = rows, rows+52
            x1, x2 = cols, cols+52
            return x1,y1,x2,y2

        gt_x1,gt_y1,gt_x2,gt_y2 = get_box(y_true)
        pred_x1,pred_y1,pred_x2,pred_y2 = get_box(y_pred)

        # Compute intersection
        i_x1 = tf.maximum(gt_x1, pred_x1)
        i_y1 = tf.maximum(gt_y1, pred_y1)
        i_x2 = tf.minimum(gt_x2, pred_x2)
        i_y2 = tf.minimum(gt_y2, pred_y2)

        def get_area(x1,y1,x2,y2):
            return (x2-x1) * (y2-y1)

        i_area = get_area(i_x1, i_y1, i_x2, i_y2)
        u_area = get_area(gt_x1, gt_y1, gt_x2, gt_y2) + get_area(pred_x1, pred_y1, pred_x2, pred_y2) - i_area

        iou = tf.math.divide(i_area, u_area)
        self.num_ex.assign_add(1)
        self.total_iou.assign_add(tf.reduce_mean(iou))
        self.iou = tf.math.divide(self.total_iou, self.num_ex)

    def result(self):
        return self.iou

    # Called at end of each epoch
    def reset_state(self):
         self.iou, self.total_iou, self.num_ex = self.get_initial_states()


# Compile the model
model.compile(
    loss={
        'class_out': 'categorical_crossentropy',
        'box_out': 'mse'
    },
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    metrics={
        'class_out': 'accuracy',
        'box_out': IoU(name='iou')
    }
)

# Custom Callback: Model Testing
def test_model(model, test_datagen):
    example, label = next(test_datagen)
    x = example['image']
    y = label['class_out']
    box = label['box_out']

    pred_y, pred_box = model.predict(x)

    pred_coords = pred_box[0]
    gt_coords = box[0]
    pred_class = np.argmax(pred_y[0])
    image = x[0]

    gt = emojis[np.argmax(y[0])]['name']
    pred_class_name = emojis[pred_class]['name']

    image = plot_bounding_box(image, gt_coords, pred_coords, norm=True)
    color = 'green' if gt == pred_class_name else 'red'

    plt.imshow(image)
    plt.xlabel(f'pred: {pred_class_name}', color=color)
    plt.ylabel(f'GT: {gt}', color=color)
    plt.xticks([])
    plt.yticks([])

# Create a figure with 6 examples testing the model
def test(model, epoch = -1):
    test_datagen = data_generator(1)

    plt.figure(figsize=(16,4))

    for i in range(0,6):
        plt.subplot(1,6, i+1)
        test_model(model, test_datagen)

    if (epoch >= 0):
        plt.savefig(os.path.join('training', 'epoch%d.png' % epoch))
    else:
        plt.show()

def lr_schedule(epoch, lr):
    if (epoch + 1) % 5 == 0:
        lr *= 0.2
    return max(lr, 3e-7)

class ShowTestImages(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        test(self.model, epoch)

_ = model.fit(
    data_generator(),
    epochs=50,
    steps_per_epoch=500,
    callbacks=[
        ShowTestImages(),
        tf.keras.callbacks.EarlyStopping(monitor='box_out_iou', patience=3, mode='max'),
        tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    ]
)
