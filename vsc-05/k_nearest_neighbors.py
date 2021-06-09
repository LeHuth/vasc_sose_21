import numpy as np
import matplotlib.pyplot as plt
import glob
from collections import Counter


def distance(a, b):
    """calculates the distance between two vectors (or matrices)"""
    # 2.1.1 Berechnen Sie die Distanz zwischen zwei Matritzen/Bildern
    return np.linalg.norm(a - b)


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
        dist_list.append((sample[0], distance(sample[1], query)))


    # 2.2 Finden Sie die k nächsten datenpunkte in data
    dist_list.sort(key=lambda tup: tup[1])
    #print(dist_list)
    # 2.3 Geben Sie das Label, welches am häufigsten uner den k nächsten Nachbar
    # vorkommt und die Häufigkeit als tuple zurück.
    # Tipp: Counter(["a","b","c","b","b","d"]).most_common(1)[0]
    li = [i[0] for i in dist_list]
    #print(li)
    print(Counter(li[:k]).most_common(1)[0])
    # returned das häufigste Element der Liste und deren Anzahl also ("b", 3)
    return Counter(li[:k]).most_common(1)[0]


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
    data_list = []
    for idx, dir in enumerate(glob.glob("images/db/train/*")):
        for img in glob.glob(dir+"/*"):
            data_list.append((idx,plt.imread(img)))

    label_list = ["car", "face", "flower"]
    # 2. Implementieren Sie die Funktion knn.
    #q = plt.imread("images/db/train/cars/car001.jpg")



    query = []
    for test in glob.glob("images/db/test/*"):
        query.append(plt.imread(test))

    for idx, q in enumerate(query):
        result = knn(q, data_list, label_list, 7)
        print(result)
        print(glob.glob("images/db/test/*")[idx], "=", label_list[result[0]])
    # 3. Laden Sie die Testbilder aus dem Ordner "images/db/test/" und rufen Sie
    # auf der Datenbank knn auf. Geben Sie zu jedem Testbild das prognostizierte Label aus.
    # Varieren Sie den Parameter k.
    # Hinweis: Mit k = 5 sollte das beste Ergebnis erzielt werden.
