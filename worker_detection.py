import json
import cv2
import os
import imutils
import tensorflow as tf
import glob
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pika
import time

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

connection1 = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel1 = connection1.channel()
channel1.queue_declare(queue='workers')

connection2 = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel2 = connection2.channel()
channel2.queue_declare(queue='peoples')
while True:
    get_peoples, b_property_peoples, body_peoples = channel.basic_get(queue='from_people_extraction_to_face_extraction')
    #ack = channel.basic_ack(delivery_tag=0, multiple=True)
    #body_peoples_json = json.dumps(body_peoples.decode('utf-8'))
    if body_peoples == None:
        continue
    # print(body_peoples)
    # print(type(body_peoples))
    loads_json = json.loads(body_peoples.decode('utf-8'))
    path = loads_json['RAM_path']
    str_path = str(path)
    #print(loads_json['RAM_path'])
    my_model = "detection_2.h5"
    model = tf.keras.models.load_model(my_model)

    detected_files = []

    detected_files.append(str_path)
    #print(detected_files)
    for file in str_path:
        detected_files.append(file)

    dataf_photos = (
        pd.DataFrame(
            {
                "filepath": detected_files,
                "label": np.zeros(len(detected_files))
            }
        )
    )


    def process_image(filepath):
        return np.asarray(Image.open(filepath).resize((128, 128))) / 255.0


    def model_prediction(model_, img):
        predictions = model_.predict(np.array([img]))
        if predictions[0][0] > predictions[0][1]:
            # return f"People: {round(100 * predictions[0][0], 4)}%"
            return "People"


        else:
            # return f"Workers: {round(100 * predictions[0][1], 4)}%"
            return "Worker"

    i = 0
    img_n = len(detected_files)

    for file in detected_files:
        #print(detected_files[i])
        my_img = process_image(detected_files[i])
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        ax.imshow(my_img, cmap='gray')
    if model_prediction(model, my_img) == "People":
        date_people = datetime.now().timestamp()
        path = f'/Volumes/RAMDisk/peoples/person_{date_people}.png'
        plt.savefig(str(path))
        person_data = {
            "RAM_Path": str(path),
            "timestamp": str(date_people),
        }
        # plt.savefig('ppl/person_{}'.format(path_to_peoples))
        # plt.savefig("ppl/person_{}.png".format((datetime.now().timestamp())))
        channel2.basic_publish(exchange='', routing_key='peoples',
                           body=json.dumps(person_data))
    else:
        date_workers = datetime.now().timestamp()
        path2 = f'/Volumes/RAMDisk/workers/person_{date_workers}.png'
        plt.savefig(str(path2))
        worker_data = {
            "RAM_Path": str(path),
            "timestamp": str(date_workers),
        }
    # plt.savefig('work/work_{}'.format(path_to_peoples))
    # plt.savefig("work/person_{}.png".format((datetime.now().timestamp())))
        channel1.basic_publish(exchange='', routing_key='workers',
                           body=json.dumps(worker_data))
        i += 1

