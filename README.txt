1.�ʿ� ��� ��ġ	
(Anaconda ȯ�濡�� ��ġ�� �����մϴ�)
pip install numpy
pip install tensorflow
pip install opencv-python

2.������

�� �н� �ϱ� 
1. �н� �� ������ data ������ �������� �з��ư� �ִ��� Ȯ��
		./data ���� class 1
       	           	�� class 2
 	          	�� class 3
	               	     ...

2. pretrain �� mobilenet ��frozen_graph.pb �� ./model/mobilenet_v1_1.0_224�� �ִ��� Ȯ��
3. retrain_mod.py ���� 
��training �ð��� ���� �ɸ����� �н��� ����� ���� �մϴ�,skip ����)
 (���� python������ log ����� ������ ���� �� �ֽ��ϴ�. ���� ipython ���� ���� ����)	

�� �׽�Ʈ �ϱ�
1. ���� ���͸��� hand.jpg ������, char ������ �ִ��� Ȯ��	
2. output_graph.pb �� output_graph.txt ������ �ִ��� Ȯ��
3. ��ķ�� ���� �ִ��� Ȯ�� ( droidcam ���α׷��� ����ǰ� �ִ��� Ȯ��)
4. test_mod.py �� ���� (droidcam ������ ����� ��� test_video_mod.py ����)
5. ������ �� ���� �κп� ���� �հ����� ��ġ��Ű�� ��ȭ ���� �� �׿� �´� ����� �߾ӻ�ܿ� ǥ�� ��	
6. q�� ������ ���α׷� ����