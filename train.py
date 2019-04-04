import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import Models, LoadBatches
from keras import optimizers
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

import os
import tensorflow as tf
from keras.callbacks import TensorBoard

class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()


parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str, default="weights")
parser.add_argument("--train_images", type = str)
parser.add_argument("--train_annotations", type = str)
parser.add_argument("--n_classes", type=int)
parser.add_argument("--input_height", type=int, default = 224)
parser.add_argument("--input_width", type=int, default = 224)

parser.add_argument('--validate',action='store_false')
parser.add_argument("--val_images", type = str, default = "")
parser.add_argument("--val_annotations", type = str, default = "")

parser.add_argument("--epochs", type = int, default = 5)
parser.add_argument("--batch_size", type = int, default = 2)
parser.add_argument("--val_batch_size", type = int, default = 2)
parser.add_argument("--load_weights", type = str, default = "")

parser.add_argument("--model_name", type = str, default = "")
parser.add_argument("--optimizer_name", type = str, default = "adadelta")

args = parser.parse_args()

train_images_path = args.train_images
train_segs_path = args.train_annotations
train_batch_size = args.batch_size
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
validate = args.validate
save_weights_path = args.save_weights_path
epochs = args.epochs
load_weights = args.load_weights

optimizer_name = args.optimizer_name
model_name = args.model_name

if validate:
	val_images_path = args.val_images
	val_segs_path = args.val_annotations
	val_batch_size = args.val_batch_size

modelFns = {'vgg_segnet':Models.VGGSegnet.VGGSegnet, 'vgg_unet':Models.VGGUnet.VGGUnet,
			'vgg_unet2':Models.VGGUnet.VGGUnet2, 'fcn8':Models.FCN8.FCN8, 'fcn32':Models.FCN32.FCN32}
modelFN = modelFns[model_name]

m = modelFN(n_classes, input_height=input_height, input_width=input_width)

if optimizer_name == 'sgd':
	optimizer = optimizers.SGD(lr=0.001, momentum=0.9)
elif optimizer_name == "rmsprop":
	optimizer = optimizers.RMSProp(lr=0.001)
elif optimizer_name == 'adagrad':
	optimizer = optimizers.Adagrad()
elif optimizer_name == 'adadelta':
	optimizer = optimizers.Adadelta()
else:
	print('Not recognized, using ADAM optimizer')
	optimizer = optimizers.Adam(lr=0.001)


m.compile(loss='categorical_crossentropy',
      optimizer= optimizer,
      metrics=['accuracy'])

if len(load_weights) > 0:
	m.load_weights(load_weights)

m.summary()
print("Model output shape",  m.output_shape)

output_height = m.outputHeight
output_width = m.outputWidth

if not os.path.exists("logs_{}".format(model_name)):
	os.mkdir("logs_{}".format(model_name))

log_dir = "logs_{}/{}".format(model_name, optimizer_name)

if not os.path.exists(log_dir):
	os.mkdir(log_dir)


if not os.path.exists("weights_{}".format(model_name)):
	os.mkdir("weights_{}".format(model_name))

weights_dir = "weights_{}/{}".format(model_name, optimizer_name)

if not os.path.exists(weights_dir):
	os.mkdir(weights_dir)

logging = TrainValTensorBoard(log_dir=log_dir)
checkpoint = ModelCheckpoint(weights_dir + '/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='loss', save_weights_only=True, save_best_only=True, period=3)
earlystopper = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=5,
                              verbose=1, mode='auto')

G = LoadBatches.imageSegmentationGenerator(train_images_path, train_segs_path, train_batch_size, n_classes, input_height, input_width, output_height, output_width)

if validate:
	G2 = LoadBatches.imageSegmentationGenerator(val_images_path, val_segs_path, val_batch_size, n_classes, input_height, input_width, output_height, output_width)

m.fit_generator(G, int(len(os.listdir(train_images_path))/train_batch_size), validation_data=G2,
                validation_steps=int(len(os.listdir(val_images_path))/val_batch_size), epochs=epochs, callbacks=[logging, checkpoint, earlystopper])

m.save_weights(os.path.join(weights_dir, "final_weights.model"))
m.save(os.path.join(weights_dir, "final.model"))
