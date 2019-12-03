import colorsys
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from timeit import default_timer as timer
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization,\
    Flatten, Dense, Lambda, LeakyReLU, Conv2DTranspose, concatenate
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageFont, ImageDraw
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model
from yolo import YOLO
from PIL import Image
import cv2
import argparse
import matplotlib.pyplot as plt


LABELS = ['PICKUP', 'SUV', 'BTR70', 'BRDM2',\
           'BMP2', 'T72', 'ZSU23', '2S3', 'D20', 'MTLB', 'MAN']
FLAGS = None

def customCNN():
    input_image = Input(shape=(64,64,3))

    # Layer 1
    x = Conv2D(32, (5,5), strides=(1,1), padding='same',activation='relu')(input_image)
    # x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(3,3), strides=2)(x)

    # Layer 2
    x = Conv2D(32, (5,5), strides=(1,1), padding='same',activation='relu')(x)
    # x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(3,3), strides=2)(x)

    # Layer 3
    x = Conv2D(64, (5,5), strides=(1,1), padding='same',activation='relu')(x)
    # x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(3,3), strides=2)(x)

    # Layer 4
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)

    # Layer 5
    x = Dense(len(LABELS), activation='softmax')(x)

    model = Model(inputs=input_image, outputs=x)

    model.load_weights("trackingClass_model.h5")

    return model

def evaluateTestingData(yolo):
    cnn = customCNN()
    testSetDir = "../../noPad/images/test/"
    testSet = os.listdir(testSetDir)
    total = len(testSet)
    i = 0
    k_labels = ['2S3', 'BMP2', 'BRDM2', 'BTR70', 'D20', 'MAN', 'MTLB', 'PICKUP', 'SUV', 'T72', 'ZSU23']

    done = os.listdir("/Users/adam/PycharmProjects/ObjectDetectionTesting/resized512/mAP/input/detection-results/")

    testSet = [img for img in testSet if img.replace(".jpg", ".txt") not in done]
    print(len(testSet))

    for img in testSet:
        i +=1
        image = Image.open(testSetDir + img)
        boxes = yolo.detect_image_atc(image)
        newBoxes = []
        imageCV2 = cv2.imread(testSetDir + img)

        # get image chip, box contains [predicted_class, score, left, top, right, bottom]
        for box in boxes:
            newBox = []
            xmin = box[2] - 5
            ymin = box[3] - 5
            xmax = box[4] + 5
            ymax = box[5] + 5
            imageChip = imageCV2[ymin:ymax,xmin:xmax,:]
            imageChip = cv2.resize(imageChip, (64, 64), interpolation=cv2.INTER_CUBIC)
            imageChip = imageChip/255.
            imageChip = np.expand_dims(imageChip, 0)
            y_pred = cnn.predict(imageChip)
            pred_class = k_labels[np.argmax(y_pred)]

            # confidence score reduced if yolo and CNN disagree
            if pred_class != box[0]:
                newConf = np.max(y_pred) - box[1]
            else:
                newConf = (np.max(y_pred) + box[1])/2

            newBox = [pred_class, newConf, box[2], box[3], box[4], box[5]]
            newBoxes.append(newBox)

        # free up some memory
        del imageCV2
        del image
        del imageChip

        txtFile = "/Users/adam/PycharmProjects/ObjectDetectionTesting/resized512/mAP/input/detection-results/" + img.replace(".jpg", ".txt")

        with open(txtFile, 'w+') as f:
            for new in newBoxes:
                label = '{} {:.2f}'.format(new[0], new[1])
                line = str(label) + " " + str(new[2]) + " " + str(new[3]) + " " + \
                       str(new[4]) + " " + str(new[5]) + "\n"
                f.write(line)

        print(str(i) + "/" + str(total))

    return

# draw bounding boxes on image
def draw_boxes(image, boxes):
    image_h, image_w, _ = image.shape

    for box in boxes:
        xmin = box[2]
        ymin = box[3]
        xmax = box[4]
        ymax = box[5]
        conf = '%.5f'%box[1]

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
        cv2.putText(image,
                    box[0] + ' ' + conf,
                    (xmin, ymin - 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1e-3 * image_h,
                    (0, 0, 255), 1)

    return image

# get bounding box for image
def evaluateImage(yolo, input, show, output):
    cnn = customCNN()
    image = Image.open(input)
    boxes = yolo.detect_image_atc(image)
    newBoxes = []
    imageCV2 = cv2.imread(input)
    k_labels = ['2S3', 'BMP2', 'BRDM2', 'BTR70', 'D20', 'MAN', 'MTLB', 'PICKUP', 'SUV', 'T72', 'ZSU23']

    # get image chip, box contains [predicted_class, score, left, top, right, bottom]
    for box in boxes:
        newBox = []
        xmin = box[2] - 7
        ymin = box[3] - 7
        xmax = box[4] + 7
        ymax = box[5] + 7
        imageChip = imageCV2[ymin:ymax, xmin:xmax, :]
        imageChip = cv2.resize(imageChip, (64, 64), interpolation=cv2.INTER_CUBIC)
        imageChip = np.expand_dims(imageChip, 0)
        y_pred = cnn.predict(imageChip)
        pred_class = k_labels[np.argmax(y_pred)]

        # confidence score reduced greatly if yolo and CNN disagree
        if pred_class != box[0]:
            newConf = np.max(y_pred) - box[1]*0.7
        else:
            newConf = (np.max(y_pred) + box[1]) / 2

        newBox = [pred_class, newConf, box[2], box[3], box[4], box[5]]
        newBoxes.append(newBox)

    # draw the boxes
    pred_image = draw_boxes(imageCV2, newBoxes)
    plt.imshow(pred_image[:, :, ::-1])

    if show:
        plt.show()

    if output:
        cv2.imwrite(output + "image_prediction.jpg", pred_image)


def main():
    global FLAGS
    # instantiate yolo model
    yolo = YOLO()

    if FLAGS.input:
        if FLAGS.show:
            show = True
        else:
            show = False

        if FLAGS.output or show:
            evaluateImage(yolo, FLAGS.input, show, FLAGS.output)
        else:
            print("Must specify at least output, show, or both. See usage with --help.")

    else:
        # evaluateTestingData(yolo)
        print("Must specify at least image path.  See usage with --help.")


    return

if __name__ == "__main__":
    '''
        Command line options
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input', type=str,
        help='Add absolute path of image.'
    )

    parser.add_argument(
        '--output', type=str,
        help='Add absolute path for saving image. Prediction is saved as image_prediction.jpg'
    )

    parser.add_argument(
        '--show', type=bool,
        help='Show Result. True or False'
    )

    FLAGS = parser.parse_args()

    main()


