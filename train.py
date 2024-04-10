import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set random seed
seed = 42

# DATA PREPARATION
df = pd.read_csv("wine_quality.csv")
y = df.pop("quality")
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=seed)

# TRAINING
clf = RandomForestClassifier(max_depth=10, random_state=seed)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# CALCULATING AND SAVING METRICS
with open("metrics.txt", 'w') as outfile:
    outfile.write(f"Accuracy: {accuracy_score(y_pred, y_test)}")

cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()
plt.savefig("confmat.png", dpi=120)
