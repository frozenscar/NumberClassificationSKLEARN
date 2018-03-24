import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
digits = datasets.load_digits()

clf = svm.SVC(gamma=0.001,C=100)
x,y =digits.data[:-15],digits.target[:-15]
clf.fit(x,y)
print("prediction : ",clf.predict(digits.data[[-13]]))
plt.imshow(digits.images[-13],cmap=plt.cm.gray_r,interpolation ="nearest")
plt.show()
