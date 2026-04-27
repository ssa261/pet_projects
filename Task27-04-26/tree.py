import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import Colormap, ListedColormap
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style='whitegrid')

import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import make_moons, make_circles, make_classification
datasets = [
    make_circles(noise=0.2, factor=0.5, random_state=42),
    make_moons(noise=0.2, random_state=42),
    make_classification(n_classes=3, n_clusters_per_class=1, n_features=2, class_sep=.8, random_state=3,
                        n_redundant=0, )
]

palette = sns.color_palette(n_colors=3)
cmap = ListedColormap(palette)

plt.figure(figsize=(15, 4))
for i, (x, y) in enumerate(datasets):
    plt.subplot(1, 3, i + 1)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap, alpha=.8)
# plt.show()

# задание 1

def plot_surface(clf, X, y):
    plot_step = 0.01
    palette = sns.color_palette(n_colors=len(np.unique(y)))
    cmap = ListedColormap(palette)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.3)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, alpha=.7,
                edgecolors=np.array(palette)[y], linewidths=2)

plt.figure(figsize=(15, 4))

for i, (X, y) in enumerate(datasets):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123) 
    # не r/s 42 так как 2-я выборка на нём не репрезентативна

    tree = DecisionTreeClassifier() # в 1 задании нужно обучить без параметров
    tree.fit(X_train, y_train)

    acc_train = tree.score(X_train, y_train)
    acc_test = tree.score(X_test, y_test)

    plt.subplot(1, 3, i + 1)
    plot_surface(tree, X, y) 
    plt.title(f"Выборка {i+1}\nTrain accuracy: {acc_train:.2f}, Test accuracy: {acc_test:.2f}")
plt.show()

# задание 2

# бля, я умоляю, нормально перенеси в юпитер

plt.figure(figsize=(15, 4))

for i, (X, y) in enumerate(datasets):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) 
    # не r/s 42 так как 2-я выборка на нём не репрезентативна

    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5) # вообще, я бы взял cpp_alpfa. но, подозреваю, авторы хотят другого
    tree.fit(X_train, y_train)

    acc_train = tree.score(X_train, y_train)
    acc_test = tree.score(X_test, y_test)

    plt.subplot(1, 3, i + 1)
    plot_surface(tree, X, y) 
    plt.title(f"Выборка {i+1}\nTrain accuracy: {acc_train:.2f}, Test accuracy: {acc_test:.2f}")
plt.show()

# бонус

import plotly.graph_objects as go
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# 1. Подготовка данных (возьмем "Луны")
X, y = datasets[1]
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), # шаг 0.05, чтобы файл не весил 100Мб
                     np.arange(y_min, y_max, 0.05))

fig = go.Figure()

# 2. Генерируем "слои" для разной глубины
depths = range(1, 16)
for depth in depths:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(X, y)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    # Добавляем поверхность (изначально видима только первая)
    fig.add_trace(go.Contour(
        x=np.arange(x_min, x_max, 0.05),
        y=np.arange(y_min, y_max, 0.05),
        z=Z,
        opacity=0.3,
        showscale=False,
        visible=(depth == 1), # Только первый слой True
        name=f"Depth {depth}"
    ))

# 3. Добавляем сами точки (они видны всегда)
fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', 
                         marker=dict(color=y, size=10, line=dict(width=1, color='Black'))))

# 4. Создаем логику слайдера
steps = []
for i, depth in enumerate(depths):
    step = dict(
        method="update",
        label=str(depth),
        args=[{"visible": [False] * len(depths) + [True]}, # Управление видимостью слоев
              {"title": f"Decision Boundary (max_depth={depth})"}]
    )
    # Магия видимости: все False, кроме текущего индекса i
    step["args"][0]["visible"][i] = True 
    steps.append(step)

sliders = [dict(active=0, currentvalue={"prefix": "Max Depth: "}, pad={"t": 50}, steps=steps)]

fig.update_layout(sliders=sliders, title="Decision Boundary (max_depth=1)")
fig.show()