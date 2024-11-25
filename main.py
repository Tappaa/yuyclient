from time import sleep

import os
import requests

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tf_keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog
import cv2

def file_select():
    app = QApplication([])
    file_path, _ = QFileDialog.getOpenFileName(
        None,
        "파일 선택",
        "",
        "이미지 파일 (*.jpg *.jpeg *.png *.bmp *.gif *.webp)"
    )
    return file_path

def take_picture():
    cap = cv2.VideoCapture(1)
    ret, frame = cap.read()
    if not ret:
        print("카메라를 사용할 수 없습니다. 나중에 다시 시도해주세요.")
        exit(1)
    cv2.imwrite("taken_picture.jpeg", frame)
    cap.release()
    return "taken_picture.jpeg"

def predict(file_path): # From teachablemachine.withgoogle.com
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("h5/keras_model.h5", compile=False)

    # Load the labels
    class_names = open("h5/labels.txt", "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(file_path).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)

    return class_names[index].replace("\n", ""), prediction[0][index]

backend_url = "https://yuybackend.simpleproject.workers.dev"

if __name__ == "__main__":
    print("시스템: 안녕하세요 육은영 상담사 챗봇입니다.")
    sleep(3)
    print("시스템: 지금부터 상담을 진행하기 전 아이의 표정을 확인하여 아이의 기분에 맞춰 상담을 진행하겠습니다.")
    sleep(3)
    choice = int(input("시스템: 아이의 표정을 인식하기 위해 사진을 업로드 하려면 1번 지금 사진을 찍으려면 2번을 입력해 주세요 : "))

    file = None
    if choice == 1:
        file = file_select()
    elif choice == 2:
        file = take_picture()
    else:
        print("잘못된 입력입니다. 처음부터 다시 시작해주세요.")
        exit(1)
    print(f"시스템: 사진이 업로드 되었습니다. 지금부터 아이의 표정을 인식하겠습니다. 잠시만 기다려주세요.")

    class_name, confidence_score = predict(file)

    print(f"시스템: 인식된 결과 아이의 표정은 {class_name}이며, 가능성은 {confidence_score * 100:.2f}% 입니다.")
    sleep(3)
    print("육은영: 지금부터 상담을 시작하겠습니다. (상담을 바로 종료하려면 '/종료'를 입력해주세요.)")

    conversation_messages = dict()
    conversation_messages["prompts"] = [
        { "role": "user", "content": f"안녕하세요 육은영 상담사님. 저는 현재 {class_name} 상태 입니다." }
    ]

    sleep(3)
    print(f"나: {conversation_messages['prompts'][0]['content']}")

    request = requests.post(backend_url, json=conversation_messages)
    while True:
        prompts = request.json()
        responses = prompts["prompts"] # List of responses
        latest_response = responses[-1]

        if str(latest_response["content"]).find("|종료|") != -1:
            print(f"시스템: {str(latest_response['content']).replace('|종료|', '')}")
            sleep(1)
            print("육은영: 상담을 종료히겠습니다. 행복한 하루 되세요.")
            break

        print(f"육은영: {latest_response["content"]}")
        send = input("금쪽이: ")
        if send == "/종료":
            print("육은영: 상담을 종료히겠습니다. 행복한 하루 되세요.")
            break
        conversation_messages = prompts
        conversation_messages["prompts"].append({ "role": "user", "content": send })

        request = requests.post(backend_url, json=conversation_messages)

        if request.status_code == 429:
            print("시스템: 서버 전송 제한에 도달하여 10초 후 다시 전송합니다.")
            sleep(10)
            request = requests.post(backend_url, json=conversation_messages)
