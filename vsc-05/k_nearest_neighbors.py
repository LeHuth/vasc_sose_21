import numpy as np
import matplotlib.pyplot as plt
import glob
from collections import Counter


def distance(a, b):
    """calculates the distance between two vectors (or matrices)"""
    # 2.1.1 Berechnen Sie die Distanz zwischen zwei Matritzen/Bildern
    return np.linalg.norm(b - a)


def knn(query, data, labels, k):
    """
    Calculates the k-NN and returns the most common label appearing in the k-NN
    and the number of occurrences.
    For each data-record i the record consists of the datapoint (data[i]) and
    the corresponding label (label[i]).

    :param query: ndarray representing the query datapoint.
    :param data: list of datapoints represents the database together with labels.
    :param labels: list of labels represents the database together with data.
    :param k: Number of nearest neighbors to consider.
    :return: the label that occured the most under the k-NN and the number of occurrences.
    """
    dist_list = []
    # 2.1 Berechnen Sie die Distanzen von query zu allen Elementen in data
    # Implementieren Sie dazu die Funktion distance
    for idx,sample in enumerate(data):
        dist_list.append(distance(sample, query))

    #print(dist_list)
    #print(labels)
    # 2.2 Finden Sie die k nächsten datenpunkte in data
    dist_sorted, labels_sorted = zip(*sorted(zip(dist_list, labels)))
    #print(dist_sorted)
    #print(labels_sorted)
    # 2.3 Geben Sie das Label, welches am häufigsten uner den k nächsten Nachbar
    # vorkommt und die Häufigkeit als tuple zurück.
    # Tipp: Counter(["a","b","c","b","b","d"]).most_common(1)[0]

    #print(li)
    #print(Counter(labels_sorted).most_common(3)[0])
    # returned das häufigste Element der Liste und deren Anzahl also ("b", 3)
    return Counter(labels_sorted).most_common(k)[0]


# ---------------------------------------------------------------------------
# k Nearest Neighbors
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Bauen Sie die Datenbank auf. Sie besteht aus zwei Listen.
    # Einmal die Datenpunkte (die Bilder) und die dazugehörigen Label.
    # Die beiden Listen werden seperat gespeichert, aber gehören zusammen, dh.
    # Liste_der_Datenpunkte[i] gehört zu Liste_der_Labels[i].
    # Die Listen sind also gleich lang.
    # Tipp:
    # mit glob.glob("images/db/test/*") bekommen Sie eine Liste mit allen Dateien in dem angegebenen Verzeichnis
    #print(glob.glob("images/db/train/*"))
    datapoints = []
    labels = []
    for idx, dir in enumerate(glob.glob("images/db/train/*")):
        for img in glob.glob(dir+"/*"):
            datapoints.append(plt.imread(img))
            labels.append(dir.split('\\')[-1])
    # 2. Implementieren Sie die Funktion knn.
    #q = plt.imread("images/db/train/cars/car001.jpg")



    query = []
    for q in glob.glob("images/db/test/*"):
        result = knn(plt.imread(q), datapoints, labels, 7)
        print(q, "=", result[0])


    # 3. Laden Sie die Testbilder aus dem Ordner "images/db/test/" und rufen Sie
    # auf der Datenbank knn auf. Geben Sie zu jedem Testbild das prognostizierte Label aus.
    # Varieren Sie den Parameter k.
    # Hinweis: Mit k = 5 sollte das beste Ergebnis erzielt werden.
