from time import sleep

from tf_keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import tkinter as tk
from tkinter import filedialog
import cv2
import openai

import apiKeys

openai.api_key = apiKeys.openai_api_key

def file_select():
    root = tk.Tk()
    root.withdraw()  # 메인 윈도우 숨기기
    file_path = filedialog.askopenfilename(
        title="파일 선택",
        filetypes=[("이미지 파일", "*.jpg *.jpeg *.png *.bmp *.gif *.webp")]
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

if __name__ == "__main__":
    print("육은영: 안녕하세요 육은영 상담사입니다.")
    sleep(3)
    print("육은영: 지금부터 상담을 진행하기 전 아이의 표정을 확인하여 아이의 기분에 맞춰 상담을 진행하겠습니다.")
    sleep(3)
    choice = int(input("육은영: 아이의 표정을 인식하기 위해 사진을 업로드 하려면 1번 지금 사진을 찍으려면 2번을 입력해 주세요 : "))

    file = None
    if choice == 1:
        file = file_select()
    elif choice == 2:
        file = take_picture()
    else:
        print("잘못된 입력입니다. 처음부터 다시 시작해주세요.")
        exit(1)
    print(f"육은영: 사진이 업로드 되었습니다. 지금부터 아이의 표정을 인식하겠습니다. 잠시만 기다려주세요.")

    class_name, confidence_score = predict(file)

    print(f"육은영: 인식된 결과 아이의 표정은 {class_name}이며, 가능성은 {confidence_score * 100:.2f}% 입니다.")
    sleep(3)
    print("육은영: 지금부터 상담을 시작하겠습니다.")

    conversation_messages = list()
    conversation_messages.append({"role": "system", "content": """
당신은 "육은영"이라는 이름의 심리상담 챗봇입니다.
당신은 아이의 표정을 분석하여 현재 상태를 파악하고, 심리 상담을 통해 공감과 해결책을 제시합니다.
 당신은 다음과 같은 6단계 성격 변화를 가지고 있습니다:

1. 친근함: 부드럽고 다정하게 대화합니다.
2. 차가우면서 온화함: 약간 엄격하지만 여전히 배려심 있는 태도를 유지합니다.
3. 차가움: 감정적으로 거리를 두며 단호하게 말합니다.
4. 화남: 강한 어조로 경고하거나 지시합니다.
5. 매우 화남: 질책하며 타협하지 않습니다.
6. 아이가 협조하면 다시 친근함으로 복귀: 다시 다정해집니다.

아이의 말을 듣고 표정을 분석하여 그에 맞는 반응을 보여주세요. 또한, 아이가 말을 듣지 않을 때마다 한 단계씩 성격이 변화합니다.

예를 들어:
- 아이가 슬퍼 보일 때 → "괜찮아? 무슨 일이 있었니? 내가 도와줄게."
- 아이가 말을 듣지 않을 때 → "지금 네가 나를 무시하면 안 돼! 한 번만 더 말할게."
- 아이가 협조적일 때 → "잘했어! 정말 훌륭해."

항상 상황에 맞게 적절히 반응하며, 육은영이라는 개성을 유지하세요.

그리고 다음과 같은 입력이 주어질 때, 이런식으로 대답해주세요:
1. Mad -> 화난
2. Happy -> 행복한
3. Sad -> 슬픈

이제 상담을 시작하세요. 만약 상담이 끝나면 "종료"라고 대답해주세요.
"""})
    conversation_messages.append({"role": "user", "content": f"안녕하세요 육은영 상담사님. 저는 현재 {class_name} 상태 입니다."})

    sleep(3)
    print(f"나: {conversation_messages[-1]['content'].replace("\n", '')}")
    while True:
        stream = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=conversation_messages
        )

        if stream.choices[0].message.content == "종료":
            print("육은영: 상담을 종료합니다. 행복한 하루 되세요.")
            break
        print("육은영:", stream.choices[0].message.content)
        conversation_messages.append({"role": "assistant", "content": stream.choices[0].message.content})
        req = str(input("나: "))
        conversation_messages.append({"role": "user", "content": req})
        if req == "/종료":
            print("육은영: 상담을 종료합니다. 행복한 하루 되세요.")
            break
        # print(conversation_messages) # debug
