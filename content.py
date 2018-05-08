# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 2: k-NN i Naive Bayes
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

from __future__ import division
import numpy as np



def hamming_distance(X, X_train):
    """
    :param X: zbior porownwanych obiektow N1xD
    :param X_train: zbior obiektow do ktorych porownujemy N2xD
    Funkcja wyznacza odleglosci Hamminga obiektow ze zbioru X od
    obiektow X_train. ODleglosci obiektow z jednego i drugiego
    zbioru zwrocone zostana w postaci macierzy
    :return: macierz odleglosci pomiedzy obiektami z X i X_train N1xN2
    """
    x = X.toarray().astype(int) #we have to change boolean matrix to int array 1/0
    x_trainT = np.transpose(X_train.toarray()).astype(int) #we transpose to change 50,20 -> 20,50
    how_many_pos = x.shape[1]
    ham_dis1 = np.dot(x,x_trainT) #how many places EQ 1 the same
    ham_dis2 = how_many_pos - ham_dis1;
    ham_dis3 = np.dot((1-x),(1-x_trainT)) #how many places EQ 0 the same
    ham_dis = ham_dis2 - ham_dis3
    #print(ham_dis1)
    return ham_dis
    pass


def sort_train_labels_knn(Dist, y):
    """
    Funkcja sortujaca etykiety klas danych treningowych y
    wzgledem prawdopodobienstw zawartych w macierzy Dist.
    Funkcja zwraca macierz o wymiarach N1xN2. W kazdym
    wierszu maja byc posortowane etykiety klas z y wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist
    :param Dist: macierz odleglosci pomiedzy obiektami z X
    i X_train N1xN2
    :param y: wektor etykiet o dlugosci N2
    :return: macierz etykiet klas posortowana wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist. Uzyc algorytmu mergesort.
    """

    #print(np.argsort(u,kind="mergesort"))#sort and get positions in matrix

    return y[np.argsort(Dist,kind='mergesort')] #take values froms y depends on position in Dist
    pass


def p_y_x_knn(y, k):
    """
    Funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla
    kazdej z klas dla obiektow ze zbioru testowego wykorzystujac
    klasfikator KNN wyuczony na danych trenningowych
    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najblizszuch sasiadow dla KNN
    :return: macierz prawdopodobienstw dla obiektow z X
    """
    # resized = np.delete(y, range(k,y.shape[1]), axis=1) #we take k nearest neighbours removing colums from 5 to 49
    #
    #
    # bin = np.bincount(resized[0], None, k)
    # bin2 = np.bincount(resized[1], None, k)
    # resultMatrix = np.vstack((bin,bin2))
    #
    #
    # for x in range(2,y.shape[0]):             #how many times for category 0 1 2 3 4
    #     bin = np.bincount(resized[x],None,k)
    #     resultMatrix = np.vstack([resultMatrix,bin])
    #
    # resultMatrix = np.delete(resultMatrix,0,axis=1) #remove column 0, because we dont have category 0
    #
    # return resultMatrix/k

    number_of_classes = 4
    resized = np.delete(y, range(k, y.shape[1]), axis=1)
    summed_with_zero = np.vstack(np.apply_along_axis(np.bincount, axis=1, arr=resized, minlength=number_of_classes + 1))
    summed = np.delete(summed_with_zero, 0, axis=1)


    return summed / k

    pass


def classification_error(p_y_x, y_true):
    """
    Wyznacz blad klasyfikacji.
    :param p_y_x: macierz przewidywanych prawdopodobienstw
    :param y_true: zbior rzeczywistych etykiet klas 1xN.
    Kazdy wiersz macierzy reprezentuje rozklad p(y|x)
    :return: blad klasyfikacji
    """

    number_of_classes = p_y_x.shape[1]
    reversed_rows = np.fliplr(p_y_x)
    predicted = number_of_classes - np.argmax(reversed_rows, axis=1)
    difference = predicted - y_true
    return np.count_nonzero(difference) / y_true.shape[0]

    pass


def model_selection_knn(Xval, Xtrain, yval, ytrain, k_values):
    """
    :param Xval: zbior danych walidacyjnych N1xD
    :param Xtrain: zbior danych treningowych N2xD
    :param yval: etykiety klas dla danych walidacyjnych 1xN1
    :param ytrain: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartosci parametru k, ktore maja zostac sprawdzone
    :return: funkcja wykonuje selekcje modelu knn i zwraca krotke (best_error,best_k,errors), gdzie best_error to najnizszy
    osiagniety blad, best_k - k dla ktorego blad byl najnizszy, errors - lista wartosci bledow dla kolejnych k z k_values
    """

    #first we have to get sorted labels
    hammingDist = hamming_distance(Xval,Xtrain) #we compare orginalText with trainig values
    sortedLabels = sort_train_labels_knn(hammingDist,ytrain)
    #errors = list(map(lambda k: classification_error(p_y_x_knn(sortedLabels, k), yval), k_values))
    ce = []
    ce.append(classification_error(p_y_x_knn(sortedLabels, k_values[0]), yval))
    ce.append(classification_error(p_y_x_knn(sortedLabels, k_values[1]), yval))
    ce.append(classification_error(p_y_x_knn(sortedLabels, k_values[2]), yval))
    ce.append(classification_error(p_y_x_knn(sortedLabels, k_values[3]), yval))
    ce.append(classification_error(p_y_x_knn(sortedLabels, k_values[4]), yval))




    return

    pass


def estimate_a_priori_nb(ytrain):
    """
    :param ytrain: etykiety dla dla danych treningowych 1xN
    :return: funkcja wyznacza rozklad a priori p(y) i zwraca p_y - wektor prawdopodobienstw a priori 1xM
    """
    pass


def estimate_p_x_y_nb(Xtrain, ytrain, a, b):
    """
    :param Xtrain: dane treningowe NxD
    :param ytrain: etykiety klas dla danych treningowych 1xN
    :param a: parametr a rozkladu Beta
    :param b: parametr b rozkladu Beta
    :return: funkcja wyznacza rozklad prawdopodobienstwa p(x|y) zakladajac, ze x przyjmuje wartosci binarne i ze elementy
    x sa niezalezne od siebie. Funkcja zwraca macierz p_x_y o wymiarach MxD.
    """
    pass


def p_y_x_nb(p_y, p_x_1_y, X):
    """
    :param p_y: wektor prawdopodobienstw a priori o wymiarach 1xM
    :param p_x_1_y: rozklad prawdopodobienstw p(x=1|y) - macierz MxD
    :param X: dane dla ktorych beda wyznaczone prawdopodobienstwa, macierz NxD
    :return: funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla kazdej z klas z wykorzystaniem klasyfikatora Naiwnego
    Bayesa. Funkcja zwraca macierz p_y_x o wymiarach NxM.
    """
    pass


def model_selection_nb(Xtrain, Xval, ytrain, yval, a_values, b_values):
    """
    :param Xtrain: zbior danych treningowych N2xD
    :param Xval: zbior danych walidacyjnych N1xD
    :param ytrain: etykiety klas dla danych treningowych 1xN2
    :param yval: etykiety klas dla danych walidacyjnych 1xN1
    :param a_values: lista parametrow a do sprawdzenia
    :param b_values: lista parametrow b do sprawdzenia
    :return: funkcja wykonuje selekcje modelu Naive Bayes - wybiera najlepsze wartosci parametrow a i b. Funkcja zwraca
    krotke (error_best, best_a, best_b, errors) gdzie best_error to najnizszy
    osiagniety blad, best_a - a dla ktorego blad byl najnizszy, best_b - b dla ktorego blad byl najnizszy,
    errors - macierz wartosci bledow dla wszystkich par (a,b)
    """
    pass
