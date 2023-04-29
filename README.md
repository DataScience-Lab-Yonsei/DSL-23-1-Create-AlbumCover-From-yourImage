# DSL-23-1-Create-AlbumCover-From-yourImage
[2023-1학기 DSL 모델링 프로젝트 생성모델/CV] prove that not everything can be an album cover


## DSL 23-1 Modelling Project D조
**팀명** : tired 피카Dju  
**팀원** : 김남훈, 김영현, 이원준, 유선재, 장현빈


# Overview
---

**1. 개요**
- 틱톡이라는 어플에서 유명한 해당 릴즈영상(https://www.youtube.com/watch?v=37eX8xgPJls&pp=ygUvcHJvb2YgdGhhdCBub3QgZXZlcnl0aGluZyBjYW4gYmUgYW4gYWxidW0gY292ZXI%3D)을 보고, 우리들의 인물사진을 사지고 앨범커버로 변경하는 프로그램을 개발하고자 함
- 우선적으로, 앨범사진의 장르를 선택하면 주어진 우리들의 인물사진을 해당 장르의 앨범커버로 바꾸고자 한다.(앨범커버사진 수집시에 두 개의 웹사이트 모두 장르별로 나눠져있기에, 이런 생각을 했으나 이게 수많은 Trial&Error를 가져왔다.)
 
**2. 데이터 확보**
- kaggle이라는 웹사이트에서 "Album Covers Dataset"(https://www.kaggle.com/datasets/anastasiapetrunia/album-covers-dataset) 해외음원의 앨범커버사진 확보
- 멜론이라는 음원사이트(https://www.melon.com/genre/album_list.htm?gnrCode=GN0100&dtlGnrCode=)에서 국내음원의 장르별 앨범커버사진 확보
  
**3. 데이터 전처리**
- Yolo를 통해 사람얼굴이 들어가 있는 얼범커버사진을 선택


# Trial And Error
---

**1. Neural Style transfer with AdaIN**
- AdaIN개념과 NN을 사용해서 사용할 우리들의 인물사진(a)을 Contents(A)로, 앨범커버사진(b)을 Style(B)로 Decoder를 학습시킨다.
- 여기서 VGG Encoder의 경우에는 사전에 학습된 모델을 사용하며, 통계학적으로 VGG로 manifold-learning을 수행하면 그 이미지의 content와 style이 남이있다는 것을 밝혀냈다. 따라서 AdaIN이라는 개념이 등장해서 content와 style을 뽑아낼 수 있는 것이다.
- x를 정규화하는 과정에서 우리들의 인물사진(a)의 스타일이 사라지고, y의 모수로 맞춰주는 과정에서 앨범커버사진(b)의 스타일을 씌워준다.
- 데이터의 경우에는 앨범커버사진(b) 여러장을 Style(B)에 계속 투입하고, 이에 반해 우리들의 인물사진(a)은 한 장을 Contents(A)에 계속 넣고 학습시킨다.
- 이후에 원하는 데이터를 얻기 위해서는 하나의 앨범커버사진(b)과 하나의 우리인물사진(a)를 각각 Style(B)와 Content(A)에 넣고 학습시킨 모델을 사용해 결과값(Deocder부분의 값)을 얻는다.
- 문제 : 필터를 입힌 것처럼 보인다. 우리가 원하는 앨범커버사진이 아니다. 따라서 이후에는 사람 얼굴에 대한 정보를 도메인으로 앨범사진을 만들어보고자 한다.

**2. StarGAN**
- 여러가지 Domain(2개이상)을 넣어 학습시킬 수 있는 GAN모델인 "StarGAN"을 사용한다.
- 사람의 피부색깔(1), 머리색깔(2), 성별(3), 장르(4)를 Domain으로 하여 장르뿐 아니라 얼굴도 변하는 사진을 생성할 수 있도록 한다.
- 문제 : 데이터의 수가 많아 학습이 오래걸리고, 성능도 좋지 않다.
  
**3. CycleGAN**
- 2개의 Domain을 사용하는 "CycleGAN"을 사용한다.
- CycleGAN의 경우, 일반적인 GAN에 Domain을 하나를 추가한다. noise로 시작되는 GAN과 달리, 새로운 Domain의 사진을 noise대신 넣고 댜신에 Decoder 앞부분에 Encoder를 붙여준다. 또한, 생성된 사진을 실제 사진과 비교하는 일반적인 GAN의 Discriminator뿐 아니라, 이전의 noise대신에 넣은 Domain사진을 비교하는 L1 Loss를 추가한다.
- 문제 : 마찬가지로 왠지 성능이 좋지 않다. 아마 앨범장르를 하나의 domain으로 넣었는데, 앨범장르별로 앨범커버가 구분되지 않기 때문일 것이다. 즉, 이 앨범커버사진을 보고 이게 어떤 장르인지 우리도 알 수 없는데, 그 장르의 앨범으로 우리의 인물사진을 바꿀 수 있냐는 의문이 나왔다.

**4. one to one Style Transfer & DCGAN**
- 이전의 1번의 Style Transfer를 사용할 때, 앨범커버사진 여러장에 우리인물사진 1장을 대응시켜 학습시켰는데 이 과정에서 DCGAN을 넣어 학습시켜본다.
- 문제 : 마찬가지로 성능이 좋지 않았고, 앨범장르별로 앨범커버가 구분되지 않는 것 같은 확신이 들었다.


# Model
---

**1. 개요**
- 이전의 장르라는 정보를 제거하고, 여러장의 사진을 사용하는 것이 아닌 우리들의 인물사진(a)과 가장 비슷한 앨범커버사진(b) 한 장을 선택하여 두 사진을 합치기로 하자.
- 얼굴에 대한 합성은 계속 사용했던 개념인 Style Transfer의 AdaIN 개념을 사용하였다.
- 앨범의 커버사진분위기는 Segementation의 누끼따기를 사용하였다.

**2. 유사한 이미지 찾기**
- ResNet50을 이요한 Manifold-learning으로 차원축소하여 하나의 벡터로 만든다.
- 이 벡터들간의 코사인유사도를 지표로 하여, 가장 큰 값을 비슷한 앨범으로 선정한다.
  
**3. 얼굴합성**
- AdaIN의 개념이 들어간 Style-GAN인 PSP(Pixel2Style2Pixel) 사용한다.

**4. 앨범커버합성**
- Mask-RCNN을 통해 누끼따기를 진행한다.
