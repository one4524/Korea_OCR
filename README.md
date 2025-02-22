
# AI OCR을 이용한 스마트 가계부

**프로젝트 기간** : 2022.10 ~ 2022.12 (2달)

**기술** : Python, PyTorch, OpenCV, Flask, Firebase, WebSocket, Android

**링크** : [시연영상](https://drive.google.com/file/d/1txzjUW8pCZym7Q4UCjoakirp4-c7dPW8/view?usp=sharing), [결과보고서](https://drive.google.com/file/d/1htJ0AzzRBGsNAqrwEuxwGhcz0kHa-Ta6/view?usp=sharing), [GitHub](https://github.com/one4524/Korea_OCR)


## 프로젝트 내용

### 1. 프로젝트 소개

AI OCR을 이용한 스마트 가계부는 텍스트 인식 모델을 활용하여 영수증 이미지 내의 텍스트를 인식하고, 필요한 정보만을 추출하여 자동으로 지출을 관리하는 서비스입니다. 본 프로젝트에서는 기존 오픈소스 한글 인식 모델의 부족한 성능을 개선하는 데 초점을 맞췄습니다.

### 2. 개발 배경

AI 기술이 발전하면서 다양한 분야에서 한글 인식 AI 기술이 필요해졌습니다. 하지만 한글은 영어보다 훨씬 많은 조합을 가지며, 이로 인해 인식 모델 개발이 어렵습니다. 따라서 본 프로젝트는 한글 인식 기술의 수요를 충족하고자 시작되었으며, 비교적 쉽게 접근할 수 있는 영수증 인식을 통해 스마트 가계부를 구현하였습니다.

### 3. 프로젝트 서비스 과정

![image](https://github.com/user-attachments/assets/acf124ef-ad20-4cde-b998-c3023dbf87e9)

<details>
<summary>전처리 과정 [펼치기]</summary>
<div markdown="1">

#### **🔹 전처리 과정**
- **이미지 왜곡 보정**: OpenCV를 활용하여 GaussianBlur, Canny, findContours 등을 이용해 영수증 외곽을 검출하고 투시 변환 적용

![image](https://github.com/user-attachments/assets/663dfb4f-898c-42bf-aee5-98086cff13bb)


- **텍스트 영역 검출 오류 해결**: 잘못된 텍스트 그룹화를 방지하기 위한 추가 보정 작업 수행

![image](https://github.com/user-attachments/assets/655e9989-876a-4315-898e-c788a2b4dfae)


- **텍스트 영역 분할 및 Crop**: 이진화 후 텍스트 영역을 식별하여 필요한 정보만 추출

![image](https://github.com/user-attachments/assets/b099b92f-d0e5-46d3-a773-5dd443e02fa5)


</div>
</details>

<details>
<summary>AI 모델 - 텍스트 인식 [펼치기]</summary>
<div markdown="1">

#### **🔹 AI 모델**
- **CRAFT 모델 활용**: 텍스트 영역 검출
 
![image](https://github.com/user-attachments/assets/94624549-f51b-4601-82aa-b0a618c1220e)



- **OCR 모델 활용**: 한글 텍스트 인식
 
![image](https://github.com/user-attachments/assets/5d79a7dd-60b6-4aa0-90fb-26c752f5cdd4)




- **후처리 과정**: 정규표현식을 사용하여 금액 및 날짜 정보 추출
 
![image](https://github.com/user-attachments/assets/cdc1e071-6fad-498c-a52b-ae87f9523f3d)


</div>
</details>


  

### 4. 핵심 AI 기술

<details>
<summary>CRAFT (Character Region Awareness for Text Detection) 모델 [펼치기]</summary>
<div markdown="1">

### CRAFT 모델 설명

![image](https://github.com/user-attachments/assets/e4a02649-39c8-453d-abb0-f96119c90de1)



<그림> CRAFT 모델 입력과 출력

![image](https://github.com/user-attachments/assets/c2aa4326-6142-4e26-b4e8-1d523040644f)




<그림> CRAFT 모델 결과 예시

 - CRAFT는 Text Detection을 수행하며, 문자 단위로 영역을 인식한다.
 - Region score는 각 픽셀이 문자의 중심에 가까울수록 확률이 1에 가깝고, 멀수록 확률이 0에 가깝도록 모델을 학습한다.
 - 또한 각 픽셀이 인접한 문자의 중심에 가까울 확률인 Affinity score도 예측한다.
 - 학습된 모델을 통해 입력 이미지의 Region score와 Affinity Score를 예측하면 Text Detection 결과를 얻을 수 있다.

 ### CRAFT 모델 수행 과정

![image](https://github.com/user-attachments/assets/9a3d7da7-51e9-434b-819d-3b55afd8a8dc)



<그림> CRAFT 모델 수행 과정

 1. 이미지를 CRAFT 모델에 입력한다.
 2. CRAFT 모델은 이미지에서 텍스트로 예상되는 영역을 예측한다.
 3. 가장 인접한 두 문자와의 거리를 계산하여 연결성을 확인한다.
 4. 연결성이 높으면 하나의 단어로 인식한다.

</div>
</details>

<details>
<summary>OCR (Optical Character Recognition) 모델 [펼치기]</summary>
<div markdown="1">

![image](https://github.com/user-attachments/assets/708997c0-33e2-439e-8584-b0f2f5e7a871)



<그림> 채택한 OCR 모델 조합

 - 네이버 CLOVA AI 팀이 발표한 deep-text-recognition-benchmark 논문에서 발표한 최고 정확도의 모델 조합은 TPS-ResNet-BiLSTM-Attn (영문 단어기준 84%) 이었다.
 - 여러 조합으로 테스트한 결과, 가장 높은 정확도를 보인 TPS-VGGNet-BiLSTM-Attn 구조를 채택하였다.


![image](https://github.com/user-attachments/assets/fb3ce6c3-c05c-46df-adb6-62f3a551d418)


<그림> OCR 모델 수행 과정


1. Transformation
- Spatial Transformer Network(STN)를 사용하여 입력으로 들어온 텍스트 이미지를 정규화하는 단계이다.
2. Feature extraction
- input 이미지를 Feature Map으로 변환하여 이미지의 특징을 추출하는 단계이다.
- 글꼴, 색상, 크기 및 배경 등 글자 인식과 관련이 없는 특성들은 억제시킨다.
- 이 단계에서 생성한 Feature Map은 문맥 정보가 없다.
3. Sequence modeling
- 이전 단계의 Feature Map에서의 column들을 이 단계에서 sequence로 가져와 문맥 정보를 추가한 Feature Map으로 만든다.
4. Prediction
- 문맥 정보를 추가한 Feature Map으로 단어(문자들)를 예측한다.


</div>
</details>

### 5. OCR 모델 학습

<details>
<summary>1) 학습 데이터 준비 및 테스트 [펼치기]</summary>
<div markdown="1">

![image](https://github.com/user-attachments/assets/4b0b3b77-3ffe-4870-8fbc-b2db3f88cbfd)
![1231332](https://github.com/user-attachments/assets/92730330-0205-415d-9b85-47fa8ba9b20b)
![image](https://github.com/user-attachments/assets/3e50de07-8699-4cbc-bfa6-3bde6fe7784a)
![image](https://github.com/user-attachments/assets/e0759877-75d4-486c-8c9e-55470e4086db)
![image](https://github.com/user-attachments/assets/d730e34f-de6f-49a3-87c6-85e17aec1738)
![image](https://github.com/user-attachments/assets/fdf70415-559c-4726-aeef-c8d0546f3e74)
![image](https://github.com/user-attachments/assets/ead48e7e-5a39-4268-80c2-2816e3718a6b)
![image](https://github.com/user-attachments/assets/fb3d8b32-57fb-420a-b698-1e48b8df6b9c)
![image](https://github.com/user-attachments/assets/4a7c6d23-92b7-46b7-850d-477c3acd5f5e)

- OCR 모델의 한글 인식 성능 개선을 위해 배치 사이즈를 64로 증가, 반복횟수 20000번으로 증가, 입력 이미지의 크기를 64x200으로 변경하였다.
- 입력이미지를 32x100에서 64x200으로 변경하였을 때 성능에 큰 상승폭을 보였다.
  
</div>
</details>


### 6. 프로젝트 의의
- 한글 OCR 인식 성능 개선을 통해 다양한 프로젝트에 적용 가능
- 실시간 번역, TTS 음성 안내, 자율주행, 시각장애인 지원 서비스 등 응용 가능성 확보

### 7. 참고자료
1. [CRAFT 모델 GitHub](https://github.com/clovaai/CRAFT-pytorch)
2. [딥러닝 기반 OCR 모델 성능 비교](https://github.com/clovaai/deep-text-recognition-benchmark)
3. [한글 OCR 프로젝트 사례](https://medium.com/cashify-engineering/improve-accuracy-of-ocr-using-image-preprocessing-8df29ec3a033)
