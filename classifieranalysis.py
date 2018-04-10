import pickle
import matplotlib.pyplot as plt

with open('MLPClassifier.pkl', 'rb') as f:
    clf = pickle.load(f)
for coef in clf.coefs_:
    print(coef.shape)