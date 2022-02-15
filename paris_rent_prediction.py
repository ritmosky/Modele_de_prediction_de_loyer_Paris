import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split  # pour l'apprentissage
from sklearn import linear_model  # pour la prédiction

from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import r2_score
from matplotlib.ticker import AutoMinorLocator


#################### Modèle avec 1 feature (surface) ####################

########## Exploration-Nettoyage des données ##########

house_data = pd.read_csv('/Users/taoufiq/Documents/machine learning/house.csv')


house_data.describe()  # un résumé des données

house_data.info()  # des infos sur les colonnes de la table

house_data.columns  # afficher les colonnes du tableau

house_data.shape  # (827, 3) = taille du tableau

house_data.isna().sum()  # trouver le nombre de valeurs manquantes

house_data = house_data.dropna()  # supprime les lignes avec des valeurs manquantes

########## Affichage des données et manip ##########

# On affiche le nuage de points dont on dispose
plt.xlabel("surface")
plt.ylabel("loyer")

plt.plot(house_data['surface'], house_data['loyer'], 'ro', markersize=4)
plt.show()

"""
lorsque la surface devient trop grande, les données semblent devenir moins
modélisables facilement, il y a plus de variabilité. On va considérer le
problème de prédiction pour les loyers inférieurs à 10,000€, afin de conserver
une robustesse du modèle
"""
"""
lorsque la surface devient trop grande, les données semblent devenir moins
modélisables facilement, il y a plus de variabilité. On va considérer le
problème de prédiction pour les loyers inférieurs à 8000€, afin de conserver
une robustesse du modèle
"""

# on a des outliers sur les loyers>=8000
house_data = house_data[house_data['loyer'] < 10000]


########## Regression Linéaire ##########

X = np.matrix([np.ones(house_data.shape[0]), house_data['surface'].values]).T

y = np.matrix(house_data['loyer']).T

# On effectue le calcul exact du paramètre theta
theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# loyer = 30.7 × surface + 266.4
print(theta)  # [ [266.45460292][ 30.66119596] ]



########## Tracé ##########

# On affiche le nuage de points dont on dispose
plt.xlabel("surface")
plt.ylabel("loyer")

plt.plot(house_data['surface'], house_data['loyer'], 'ro', markersize=4)

# On affiche la droite entre 0 et 250
plt.plot([0,250], [theta.item(0), theta.item(0) + 250 * theta.item(1)],
         linestyle='--', c='#000000')

plt.show()

""" pour un appart de 30m2 le loyer = theta.item(0) + theta.item(1) * 30 """


def loyer_regL(x):
    return theta.item(0)+theta.item(1)*x



########## Echantillonnage des données = Sampling ##########
"""
np.random.randint(4)  ->  1 nombre aléatoire entre 0 et 3 inclus
np.random.randint(4, 6)  ->  1 nombre aléatoire entre 4 et 5 inclus
np.random.randint(4, 6, 2)  ->  1 tableau np de 2elem entre 4 et 5 inclus
np.random.randint(4, size=2)   ->  1 tableau np de 2elem entre 0 et 3 inclus
"""

# on récupère le nombre d'observations
size = house_data.shape[0]

# on crées un tableau numpy = 10% du dataset
sample = np.random.randint(size, size=int(size*0.8))

# sélectionne aléatoirement des échantillons dont les indices sont dans sample
sampled_data = house_data.iloc[sample]


########## Plus rapidement avec scikit-learn ##########

X = house_data[["surface"]]    # les entrées
y = house_data["loyer"]    # la prédiction

# sampling
xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8)

# apprentissage par le modèle et un exemple de prédiction
regr = linear_model.LinearRegression()
regr.fit(xtrain, ytrain)
regr.predict(xtest)


###############################################################
###############################################################


#################### Modèle avec 2 features (surface, lieu) ####################


########## Exploration-Nettoyage des données ##########

# récupération des données
data = pd.read_csv('/Users/taoufiq/Documents/machine learning/house_data2.csv')


data.describe()  # un résumé des données

data.info()  # des infos sur les colonnes de la table

data.columns  # afficher les colonnes du tableau

data.shape  # (827, 3) = taille du tableau

data.isna().sum()  # trouver le nombre de valeurs manquantes

data = data.dropna()  # supprime les lignes avec des valeurs manquantes

"""
lorsque la surface devient trop grande, les données semblent devenir moins
modélisables facilement, il y a plus de variabilité. On va considérer le
problème de prédiction pour les loyers inférieurs à 8000€, afin de conserver
une robustesse du modèle
"""
data = data[data['price'] < 8000] # on a des outliers sur les loyers>=8000

data = data.reset_index(drop = True)  # On réindexe



########## Affichage des données ##########

# On affiche les données nettoyées
data.plot.scatter("price", "surface", c="arrondissement", colormap='viridis')
plt.show()


# représentation du loyer en fonction de l'arrondissement
plt.xlabel("arrondissement")
plt.ylabel("price")
plt.plot(data['arrondissement'], data['price'], 'ro', markersize=4)
plt.show()

# répartition du loyer en fonction de l'arrondissement
ax1 = sns.violinplot(x="arrondissement", y="price", data=data, hue='arrondissement')
ax1.minorticks_on()
ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
ax1.grid(which='minor', axis='x', linewidth=1)
plt.show()


# représentation du loyer en fonction de la surface
plt.xlabel("surface")
plt.ylabel("price")
plt.plot(data['surface'], data['price'], 'ro', markersize=4)
plt.show()



# changer la valeur de l'arrondissement (10)
arr = data['arrondissement'][:]
arr[arr == 10] = 5

# changer le type de l'arrondissement en entier
data['arrondissement'] = data['arrondissement'].astype(int)


# afficher en 3d le loyer en fonction de la surface et de l'arrondissement
fig = plt.figure().gca(projection='3d')

fig.scatter(arr, data['surface'], data['price'], c=tmp_arr, cmap="viridis")
plt.show()



########## Régression sur chaque arrondissement ##########

X = data[["surface", "arrondissement"]]  # les entrées
y = data["price"]   # prédiction

# X = pd.get_dummies(data=X, drop_first=True)

# sampling normalement avec 0.8
xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=0.3)


# apprentissage par le modèle et un exemple de prédiction
lr = linear_model.LinearRegression()
lr_baseline = lr.fit(xtrain[["surface"]], ytrain)
baseline_pred = lr_baseline.predict(xtest[["surface"]])

plt.plot(xtest[["surface"]], ytest, 'bo', markersize = 5)
plt.plot(xtest[["surface"]], baseline_pred, color="skyblue", linewidth = 2)
plt.show()

# calculons la somme quadratique des résidus R2 comme val d'éval de la reg
def sumsq(x,y):
    return sum((x - y)**2)

def r2score(pred, target):
    return 1 - sumsq(pred, target) / sumsq(target, np.mean(target))

score_bl = r2score(baseline_pred[:], ytest)
print(score_bl)


lrs = []
for i in np.unique(xtrain["arrondissement"]):
    # On génère un jeu de données par arrondissement
    tr_arr = xtrain['arrondissement']==i
    te_arr = xtest['arrondissement']==i

    xtrain_arr = xtrain[tr_arr]
    ytrain_arr = ytrain[tr_arr]

    xtest_arr = xtest[te_arr]
    ytest_arr = ytest[te_arr]

    lr = linear_model.LinearRegression()
    lr.fit(xtrain_arr[["surface"]], ytrain_arr)
    lrs.append(lr)


########## Régression finale ##########

# prédiction finale en combinant les différents modèles par arrondissement
final_pred = []

for idx,val in xtest.iterrows():
    final_pred.append(lrs[int(val["arrondissement"]-1)].predict([[val["surface"]]])[0])

r2score(final_pred, ytest)


# afficher cette prédiction finale
plt.plot(xtest[["surface"]], ytest, 'bo', markersize = 5)
plt.plot(xtest[["surface"]], lrs[0].predict(xtest[["surface"]]), color="#00FFFF", linewidth = 2)
plt.plot(xtest[["surface"]], lrs[1].predict(xtest[["surface"]]), color="#0000FF", linewidth = 2)
plt.plot(xtest[["surface"]], lrs[2].predict(xtest[["surface"]]), color="#00FF00", linewidth = 2)
plt.plot(xtest[["surface"]], lrs[3].predict(xtest[["surface"]]), color="#FF0000", linewidth = 2)
plt.plot(xtest[["surface"]], lrs[4].predict(xtest[["surface"]]), color="#FFFF00", linewidth = 2)

plt.show()