[![Video Label](http://img.youtube.com/vi/yD82k54bnMs/0.jpg)](https://www.youtube.com/watch?v=yD82k54bnMs)

### 1.필요 모듈 설치	
(Anaconda 환경에서 설치를 권장합니다)
pip install numpy
pip install tensorflow
pip install opencv-python

### 2.실행방법 

##### 모델 학습 하기 
1. 학습 할 사진이 data 폴더에 폴더별로 분류됐고 있는지 확인
		./data ── class 1
       	     	├ class 2
 	          	├ class 3
	           	     ...

2. pretrain 된 mobilenet 모델frozen_graph.pb 이 ./model/mobilenet_v1_1.0_224에 있는지 확인
3. retrain_mod.py 실행 
（training 시간이 오래 걸림으로 학습된 결과를 제공 합니다,skip 가능)
 (기존 python에서는 log 기록이 보이지 않을 수 있습니다. 따라서 ipython 으로 실행 권장)	

### 모델 테스트
1. 현재 디렉터리에 hand.jpg 사진과, char 폴더가 있는지 확인	
2. output_graph.pb 와 output_graph.txt 파일이 있는지 확인
3. 웹캠이 켜져 있는지 확인 ( droidcam 프로그램이 실행되고 있는지 확인)
4. test_mod.py 을 실행 (droidcam 실행이 어려운 경우 test_video_mod.py 실행)
5. 오른쪽 위 검정 부분에 오른 손가락을 위치시키고 수화 진행 시 그에 맞는 결과가 중앙상단에 표시 됨	
6. q를 누르면 프로그램 종료

