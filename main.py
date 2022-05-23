import math
import random
import copy
import numpy
import numpy as np

file = open("australian.dat", "r")
l = []
for line in file:
    l.append(line.split())

wynik = []
for i in l:
    wynik.append(list(map(lambda e: float(e), i)))

mojalista = wynik

def MetrykaEuklidesowa(listaA, listaB):
    tmp = 0
    for i in range(len(listaA)-1):
        tmp += (listaA[i] - listaB[i])**2
    return math.sqrt(tmp)


def zadanie1(lista):
    slownik = {}
    for i in lista[1:]:
        if i[14] not in slownik.keys():
            slownik[i[14]] = [MetrykaEuklidesowa(lista[0], i)]
        else:
            slownik[i[14]].append(MetrykaEuklidesowa(lista[0], i))
    return slownik


# print(MetrykaEuklidesowa(mojalista[0], mojalista[3]))
# print(zadanie1(mojalista)[1.0])

m = [[1,2,3], [3,4,5], [2,4,5]]


def wskaznik(macierz, wynik=0):
    indeksy = list(range(len(macierz)))

    if len(macierz) == 2 and len(macierz[0]) == 2:
        wartosc = macierz[0][0] * macierz[1][1] - macierz[1][0] * macierz[0][1]
        return wartosc

    for fc in indeksy:
        macierz_kopia = macierz.copy()
        macierz_kopia = macierz_kopia[1:]
        wysokosc = len(macierz_kopia)
        for i in range(wysokosc):
            macierz_kopia[i] = macierz_kopia[i][0:fc] + macierz_kopia[i][fc + 1:]

        znak = (-1) ** (fc % 2)
        pod_wskaznik = wskaznik(macierz_kopia)
        wynik += znak * macierz[0][fc] * pod_wskaznik

    return wynik

# print(wskaznik(m))


def MetrykaEuklidesowaInaczej(listaA, listaB):
    tmp = sum((elem1-elem2)**2 for elem1, elem2 in zip(listaA, listaB))
    return math.sqrt(tmp)


def odlegosciOdx(lista, x):
    wynik = []
    for i in lista:
        para = (i[-1], (MetrykaEuklidesowa(x, i)))
        wynik.append(para)
    return wynik


def segregacjaOdleglosci(lista):
    slownik = {}
    for i in lista:
        if i[0] not in slownik.keys():
            slownik[i[0]] = [i[1]]
        else:
            slownik[i[0]].append(i[1])
    return slownik


def sumowanieOdleglosci(lista, k):
    slownik = {}
    for i in lista.keys():
        tmp_list = lista[i]
        tmp_list.sort()
        slownik[i] = sum(tmp_list[0:k])
    return slownik


def getList(dict):
    list = []
    for key in dict.keys():
        list.append(key)
    return list

def decyzja(lista):
    min = lista[0.0]
    dec = 0
    for i in getList(lista)[1:]:
        if lista[i] == min:
            return None
        if lista[i] < min:
            min = lista[i]
            dec = i
    return dec


def MetrykaEuklidesowa2(listaA, listaB, czyOstatni=True):
    tmp = 0
    if czyOstatni:
        listaA=listaA[:-1]
        listaB=listaB[:-1]
    v1 = np.array(listaA)
    v2 = np.array(listaB)
    c = v1 - v2
    tmp =np.dot(c,c)
    return math.sqrt(tmp)


def decyzja2(lista, x, k):
    odleglosc = odlegosciOdx(lista, x)
    slownik = segregacjaOdleglosci(odleglosc)
    sumaodleglosci = sumowanieOdleglosci(slownik, k)
    buff_lista = [(k, v) for k, v in sumaodleglosci.items()]
    min = buff_lista[0][1]
    dec = 0
    for para in buff_lista[1:]:
        if para[1] == min:
            return None
        if para[1] < min:
            min = para[1]
            dec = para[0]
    return dec

argx = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
print(decyzja2(mojalista, argx, 5))
# print(MetrykaEuklidesowa(mojalista[0], mojalista[3]))
# print(MetrykaEuklidesowaInaczej(mojalista[0], mojalista[3]))
# print(segregacjaOdleglosci(odlegosciOdx(mojalista, argx)))
# print(sumowanieOdleglosci(segregacjaOdleglosci(odlegosciOdx(mojalista, argx)), 5))
# print(decyzja(sumowanieOdleglosci(segregacjaOdleglosci(odlegosciOdx(mojalista, argx)), 5)))
# print("------------------------------------------")
# print(MetrykaEuklidesowa(mojalista[0], mojalista[3]))
# print(MetrykaEuklidesowa2(mojalista[0], mojalista[3]))
# slow = {0.0: 17.9, 1.0: 1.2, 3.0: 1.2}
# print(decyzja(slow))


def calki_monte_carlo(f, a, b, n):
    result = 0
    for i in range(n):
        result += f(random.uniform(a, b))
    return (result / n) * (b - a)


#print(calki_monte_carlo(lambda x: x**2, 0, 1, 5000))


def calki_kwadraty(f, a, b, n):
    step = (b - a) / n
    result = 0
    for i in range(n):
        result += f(a + i * step) * step

    return result

#print(calki_kwadraty(lambda x: x**2, 0, 1, 5000))

def segregacjaKolorowan(lista):
    slownik = {}
    for i in lista:
        if i[-1] not in slownik.keys():
            slownik[i[-1]] = [i[0:-1]]
        else:
            slownik[i[-1]].append(i[0:-1])
    return slownik

def losoweKolorowanie(lista, iloscTypowZachowan):
    # losowe kolorowanie
    for i in lista:
        i[-1] = float(random.randint(0, iloscTypowZachowan - 1))
    return lista

def kMeans2(lista):
    buff_lista = copy.deepcopy(lista)
    slownikWynik = {}
    slownik = segregacjaKolorowan(buff_lista)
    punktyCiezkosci = {}
    for klasa in slownik:
        minimalna = float(math.inf)

        for element in slownik[klasa]:
            sumaOdleglosci = 0

            for i in range(len(slownik[klasa])):
                sumaOdleglosci += MetrykaEuklidesowa2(element, slownik[klasa][i])

            sredniaOdleglosci = sumaOdleglosci / len(slownik[klasa])
            if sredniaOdleglosci < minimalna:
                punktyCiezkosci[klasa] = (element, sredniaOdleglosci)
                minimalna = sredniaOdleglosci

    for klasa in slownik:

        for element in slownik[klasa]:
            minimalna = float(math.inf)
            punkt = ()

            for klasaCiezkosci in punktyCiezkosci:
                odlegloscDoPunktuCiezkosci = MetrykaEuklidesowa2(element, punktyCiezkosci[klasaCiezkosci][0])
                if odlegloscDoPunktuCiezkosci < minimalna:
                    punkt = punktyCiezkosci[klasaCiezkosci]
                    minimalna = odlegloscDoPunktuCiezkosci

            for klasaCiezkosci in punktyCiezkosci:
                if punkt == punktyCiezkosci[klasaCiezkosci]:
                    if klasaCiezkosci not in slownikWynik.keys():
                        slownikWynik[klasaCiezkosci] = [element]
                    else:
                        slownikWynik[klasaCiezkosci].append(element)

    listaWynik = []
    for klasa in slownikWynik:
        for element in slownikWynik[klasa]:
            element.append(klasa)
            listaWynik.append(element)

    if listaWynik == buff_lista:
        return listaWynik
    else:
        return kMeans2(listaWynik)





# print(mojalista)
# kopia_mojalista = copy.deepcopy(mojalista)
# listaPoKmeans = kMeans2(losoweKolorowanie(kopia_mojalista, 2))
# mojalista.sort()
# print(mojalista)
# listaPoKmeans.sort()
# print(listaPoKmeans)


def sredniaArytmetyczna(listaA, czyOstatni=True):
    if czyOstatni:
        listaA=listaA[:-1]
    v1 = np.array(listaA)
    ilosc = len(v1)
    srednia = sum(v1)/ilosc
    return srednia


def sredniaArytmetycznaWektorowo(listaA, wektorJedynek):
    v1 = np.array(listaA)
    tmp = np.dot(v1, wektorJedynek)
    srednia = tmp/len(v1)
    return srednia

c = [1, 1, 1, 1]
print(sredniaArytmetycznaWektorowo([1,2,5,6], c))
#print(sredniaArytmetyczna([1,2,3,4,5],False))

def wariancja(listaA, czyOstatni=True):
    srednia = sredniaArytmetyczna(listaA, czyOstatni)
    if czyOstatni:
        listaA=listaA[:-1]
    v1 = np.array(listaA)
    sum = 0
    for i in v1:
        sum += (i - srednia)**2
    war = sum/len(v1)
    return war

def wariancjaWektorowo(listaA, c):
    sr = sredniaArytmetycznaWektorowo(listaA, c)
    v1 = np.array(listaA)
    vectorOnes = np.ones(len(listaA))
    v2 = v1 - sr * vectorOnes
    c = np.dot(v2, v2)
    return c / len(listaA)

# print(wariancjaWektorowo([1,2,5,6],c))

def odchylenieStandardowe(listaA, czyOstatni=True):
    war = wariancja(listaA, czyOstatni)
    return math.sqrt(war)

# print(odchylenieStandardowe([7, 4, -2], False))

def sredniaWektorow(lista, czyOstatni=True):
    lista_wynik = []
    if czyOstatni:
        for elem in lista:
            elem = elem[:-1]
            lista_wynik.append(elem)
    else:
        lista_wynik = copy.deepcopy(lista)
    return [sum(x) / len(x) for x in zip(*lista_wynik)]

#print(sredniaWektorow([[1, 2, 3], [1, 2, 3], [6, 9, 4], [4, 6, 1]], True))

def wariancjaWektorow(lista, czyOstatni=True):
    srednia = sredniaWektorow(lista, czyOstatni)
    lista_buff = []
    if czyOstatni:
        for elem in lista:
            elem = elem[:-1]
            lista_buff.append(elem)
    else:
        lista_buff = copy.deepcopy(lista)
    return [
        sum([(x - srednia[i]) ** 2 for i, x in enumerate(elem)]) / len(elem)
        for elem in lista_buff
    ]

# print(wariancjaWektorow([[1, 2, 3], [2, 4, 3], [6, 9, 4], [5, 1, 4]], True))

def odchylenieStandardoweWektorow(lista, czyOstatni=True):
    return [math.sqrt(x) for x in wariancjaWektorow(lista, czyOstatni)]

#print(odchylenieStandardoweWektorow([[1, 2, 3], [2, 4, 3], [6, 9, 4]], False))

#(2,1)
#(5,2)
#(7,3)
#(8,3)
# Wynik beta0 =2/7 beta1=5/14

def regersjaLiniowa(list):
    x = np.array([i[0] for i in list])
    x_transposed = np.array([
        np.ones(len(x)),
        x
    ])
    x = np.transpose(x_transposed)
    y = np.transpose(np.array([i[1] for i in list]))
    x_t = np.linalg.inv(np.dot(x_transposed, x))
    r = np.dot(x_t, x_transposed)
    r = np.dot(r, y)
    return r


list = [[2, 1], [5, 2], [7, 3], [8, 3]]

A = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]])
A1 = np.array([[1, 0, 1, 0, 1], [1, 1, 0, 1, 0], [0, 1, 1, 0, 0], [0, 1, 0, 1, 1], [1, 0, 0, 1, 1]])
def funkcajaRzA(list, Q):
    return np.dot(np.transpose(Q), list)

def funkcjaProj(vector_v, vecor_u):
    L = numpy.dot(vector_v, vecor_u)
    M = numpy.dot(vecor_u, vecor_u)
    projekcja = (L/M) * vecor_u
    return projekcja

def funkcajaQzA(list):
    dlugosc_u1 = math.sqrt(numpy.dot(list[0:,0],list[0:,0]))
    e1 = (1/dlugosc_u1) * list[0:,0]
    Q = np.array([e1])
    U = np.array([list[0:,0]])
    U = np.transpose(U)
    for i in range(0, np.shape(A)[1]-1):
        proj_buff = 0
        for y in range(i+1):
            p = funkcjaProj(list[0:, i+1], U[0:, y])
            proj_buff += p
        u = list[0:, i+1] - proj_buff
        U = np.transpose(U)
        U = numpy.append(U, [u], axis=0)
        U = np.transpose(U)
        dlugosc_u = math.sqrt(numpy.dot(u, u))
        e = (1/dlugosc_u) * u
        Q = numpy.append(Q, [e], axis=0)
    return np.transpose(Q)

print("------------------------------------")
Q = funkcajaQzA(A)
Q = np.matrix.round(Q, 3)
print("Macierz Q z A:")
print(Q)
print("---------------------------")
R = funkcajaRzA(A, funkcajaQzA(A))
R = np.matrix.round(R, 3)
print("Macierz R z A:")
print(R)

def A_nastepna(A):
    Q = funkcajaQzA(A)
    return np.dot(np.dot(np.transpose(Q), A), Q)


def czyMacierzGornoTrojkatna(list):
    rozmiar = np.shape(list)[1]
    if (np.diag(list)-np.transpose(np.dot(list, np.ones((rozmiar, 1))))).all() > 0.00001:
        return True
    else:
        return False


def wartosciWlasne(list):
    buff_A = copy.deepcopy(list)
    while czyMacierzGornoTrojkatna(buff_A):
        buff_A = A_nastepna(buff_A)
    return np.diag(buff_A)

A2 = np.array([[1, 2, 3],
              [4, 1., 5],
              [7, 5., 1]])

print("Wartości własne A2:")
print(wartosciWlasne(A2))

def gaussJordan(list):
    rozmiar = np.shape(list)[0]
    wektor = []
    for i in range(rozmiar):
        if list[i][i] == 0.0:
            return "Wykryto zero!"

        for j in range(rozmiar):
            if i != j:
                ratio = list[j][i] / list[i][i]

                for k in range(rozmiar + 1):
                    list[j][k] = list[j][k] - ratio * list[i][k]

    for x in range(rozmiar):
        wektor.append(list[x][rozmiar] / list[x][x])
    return wektor


def odejmoanieWarotsciWlasnej(list, wartoscWlasna):
    buff_list = copy.deepcopy(list)
    rozmiar = np.shape(list)[1]
    for i in range(rozmiar):
        for j in range(rozmiar):
            if i == j:
                buff_list[i][j] = list[i][j] - wartoscWlasna
    return buff_list


def dodanieKolumnyZer(list, wartosciWlasne):
    wynik = {}
    rozmiar = np.shape(list)[1]
    zera = np.zeros((rozmiar, 1))
    x = 0
    for i in wartosciWlasne:
        wynik[x] = np.hstack((odejmoanieWarotsciWlasnej(list, i), zera))
        x+=1
    return wynik


def wektoryWlasne(list):
    macierze = dodanieKolumnyZer(list, wartosciWlasne(list))
    wektory = []
    for i in macierze:
        macierze[i] = np.delete(macierze[i], len(macierze) - 1, 0)
        wektory.append((np.round(gaussJordan(macierze[i]) + [-1.], 3) * -1).tolist())
    return wektory

print("Wektory własne A2:")
print(wektoryWlasne(A2))

A3 = np.array([[1,1,1,0,1,0,0,0],
               [1,1,1,0,-1,0,0,0],
               [1,1,-1,0,0,1,0,0],
               [1,1,-1,0,0,-1,0,0],
               [1,-1,0,1,0,0,1,0],
               [1,-1,0,1,0,0,-1,0],
               [1,-1,0,-1,0,0,0,1],
               [1,-1,0,-1,0,0,0,-1]])

def czyOrtogonalnaMacierz(macierz):
    macierz_buff = np.dot(np.transpose(macierz), macierz)
    x = np.count_nonzero(macierz_buff - np.diag(np.diagonal(macierz_buff)))
    if x == 0:
        return True
    else:
        return False


def ortonormalizacja(macierz):
    macierz = np.transpose(macierz)
    macierz_buff = []
    for i in macierz:
        dlugosc_wektora = math.sqrt(np.dot(i,i))
        print(dlugosc_wektora)
        macierz_buff.append(i/dlugosc_wektora)

    macierz_wynik = np.dot(np.transpose(macierz_buff), macierz_buff)
    return macierz_buff, macierz_wynik  # macierz_buff macierz ortonormalna macierz_wynik b* (b^-1)

# print(czyOrtogonalnaMacierz(A3))

wektorA =np.array([8,6,2,3,4,6,6,5])

def Btr_przez_wektor_A(macierz ,wektorA):
    return np.dot(macierz, wektorA)

macierz_ortonormalna, jednostkowa = ortonormalizacja(A3)
print(np.round(jednostkowa,3))
print(np.round(Btr_przez_wektor_A(macierz_ortonormalna,wektorA), 3))