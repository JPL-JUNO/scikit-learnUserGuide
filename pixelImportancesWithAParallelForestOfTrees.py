from sklearn.datasets import fetch_olivetti_faces
from sklearn.ensemble import ExtraTreesClassifier
from time import time
import matplotlib.pyplot as plt

nJobs = 1
data = fetch_olivetti_faces()
X, y = data.data, data.target

mask = y < 5
