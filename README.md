# AI
2019S


인공지능 강의 후 이론에 기반하여 구현한 코드입니다. ML 라이브러리를 사용하지 않고 기초적인 인공지능을 구현합니다.

### ex1. Non-regularized regression

data_lab1_iis.txt 에서 데이터를 읽어 그래프에 표시합니다. linear regression parameter를 batch gradient descent(BGD), stochastic gradient
descent(SGD), closed-form method 세 가지 방식을 이용하여 각각 구한 후 결과에서 이를 비교할 수 있습니다.

![Figure_5](https://user-images.githubusercontent.com/39822788/70224168-231c2d00-1790-11ea-8a67-d178eb49b2a3.png)

blue: BGD green: SGD cyan: CFS

### ex2. Regularized regression

data_lab2_iis.txt 에서 데이터를 불러옵니다. 70%를  training data 로 사용하고 나머지 30% 를 test set 으로 사용합니다.  
optimal regression parameters을 구하기 위한 hypothesis function은 다음과 같습니다.

a) unregularized linear : red line

b) unregularized parabolic : green line

c) unregularized 5th-order polynomial : blue line

d) regularized 5th-order polynomial (RIDGE) : cyan line

![Figure_5](https://user-images.githubusercontent.com/39822788/70264115-28529980-17db-11ea-8758-c191912be354.png)


붉은 점은 traning set, 녹색 점은 test set입니다.


### ex3. Feedforward Neural Network (FFNN)  

forward propagation, back propagation을 반복하며 G(output)과 v, w를 업데이트합니다.

### ex4. Feedback Neural Network  
a.k.a Recurrent Neural Network(RNN)

Elman RNN을사용해 sequential adder를 구현하는 예제입니다. 데이터는 0,1로 구성된 길이 8인 배열로 랜덤 함수로 직접 만듭니다.

![캡처](https://user-images.githubusercontent.com/39822788/70310844-edd91300-1853-11ea-9fb6-d72166042722.PNG)

a) backpropagation

b) resilient propagation

두 방법으로 구현했습니다.

### ex5. K-means  

#### a) ex5. Clustering some synthetic data  
ex5.py

data_kmeans.txt 에서 데이터를 읽어와 clustering합니다.

![Figure_1](https://user-images.githubusercontent.com/39822788/70317184-9beaba00-1860-11ea-8b6a-33a17e7c583b.png)

십자가 표시는 test 결과입니다.

#### b) Clustering some real data  
ex5_2.py

The source
of this dataset is the The Student/Teacher Achievement Ratio (STAR) Project
organized by the Tennessee State Department of Education in the USA. The reference is the
following:
https://dataverse.harvard.edu/dataset.xhtml?persistentId=hdl:1902.1/10766

grade_students.csv 에서 데이터를 읽어와 clustering 합니다. 데이터는 학생들의 freelunch, absent, score(reading, Math, listening) 으로 총 6개의 feature를 가집니다.  
주어진 데이터로 학생을 3개의 cluster로 분류합니다.(gifted, average, weak) 이를 통해 부유한 정도와 성적의 상관관계를 확인할 수 있습니다.
