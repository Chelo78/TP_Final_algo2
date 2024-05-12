from typing import Generic, Optional, TypeVar, Callable, Any 
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from copy import deepcopy  # Perdon mariano
from abc import ABC, abstractmethod

class Nodo(ABC):
    def __init__(self, data: pd.DataFrame, target: pd.Series) -> None:
        self.atributo: Optional[str] = None # se setea cuando se haga un split
        self.categoria: Optional[str] = None # LA CATEGORIA RESULTANTE DEL SPLIT ANTERIOR
        self.data: pd.DataFrame = data
        self.target: pd.Series = target
        self.clase: Optional[str] = self.target.value_counts().idxmax()
        self.subs: list[ArbolDecision] = []

    @abstractmethod
    def _mejor_split(self) -> str: # mejor split id3
        raise NotImplementedError

    @abstractmethod
    def _split(self, atributo: str) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def entropia(self) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def _information_gain(self, atributo: str) -> float:
        raise NotImplementedError
        
    @abstractmethod
    def es_raiz(self):
        raise NotImplementedError
    
    @abstractmethod
    def es_hoja(self):
        raise NotImplementedError


class NodoID3(Nodo):
    def __init__(self, data: pd.DataFrame, target: pd.Series) -> None:
        super().__init__(data, target)
        
    def _mejor_split(self) -> str: 
        mejor_ig = -1
        mejor_atributo = None
        atributos = self.data.columns

        for atributo in atributos:
            ig = self._information_gain(atributo)
            if ig > mejor_ig:
                mejor_ig = ig
                mejor_atributo = atributo
        
        return mejor_atributo

    def _split(self, atributo: str) -> None:
        self.atributo = atributo # guardo el atributo por el cual spliteo
        for categoria in self.data[atributo].unique():
            nueva_data = self.data[self.data[atributo] == categoria]
            nueva_data = nueva_data.drop(atributo, axis = 1) # la data del nuevo nodo sin el atributo por el cual ya se filtró
            nuevo_target = self.target[self.data[atributo] == categoria]
            nuevo = NodoID3(nueva_data, nuevo_target)
            nuevo.categoria = categoria
            nuevo_arbol = ArbolDecision("ID3")
            nuevo_arbol.raiz = nuevo
            self.subs.append(nuevo_arbol)
    
    def entropia(self) -> float:
        entropia = 0
        proporciones = self.target.value_counts(normalize= True)
        target_categorias = self.target.unique()
        for c in target_categorias:
            proporcion = proporciones.get(c, 0)
            entropia += proporcion * np.log2(proporcion)
        return -entropia if entropia != 0 else 0
    
    def _information_gain(self, atributo: str) -> float:
        entropia_actual = self.entropia()
        len_actual = len(self.data)

        nuevo = deepcopy(self)
        nuevo._split(atributo)

        entropias_subarboles = 0 # no son las entropias, son |D_v|/|D| * Entropia(D_v)

        for subarbol in nuevo.subs:
            entropia = subarbol.raiz.entropia()
            len_subarbol = len(subarbol.raiz.data)
            entropias_subarboles += ((len_subarbol/len_actual)*entropia)

        information_gain = entropia_actual - entropias_subarboles
        return information_gain
    
    def es_raiz(self):
        return self.categoria is None
    
    def es_hoja(self):
        return self.subs == []
    
class ArbolDecision:
    def __init__(self, type: str) -> None:
        self.raiz: Optional[Nodo] = None
        self.target_categorias: Optional[list[str]] = None
        self.type = type

    def __len__(self) -> int:
        if self.raiz.es_hoja():
            return 1
        else:
            return 1 + sum([len(subarbol) for subarbol in self.raiz.subs])

    def _values(self):
        recuento_values = self.raiz.target.value_counts()
        values = []
        for valor in self.target_categorias:
            value = recuento_values.get(valor, 0)
            values.append(value)
        return values

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.raiz = NodoID3(X,y)
        def _fit_id3(arbol):
            arbol.target_categorias = y.unique()
            if not (len(arbol.raiz.target.unique()) == 1 or len(arbol.raiz.data.columns) == 0):
                mejor_atributo = arbol.raiz._mejor_split()
                arbol.raiz._split(mejor_atributo)
                for sub_arbol in arbol.raiz.subs:
                    _fit_id3(sub_arbol)
        if self.type == "ID3":
            _fit_id3(self)
        elif self.type == "C45":
            pass
    
    def predict(self, X: pd.DataFrame) -> list[str]:
        predicciones = []

        def _interna(arbol, X):
            nodo = arbol.raiz
            if nodo.es_hoja():
                predicciones.append(nodo.clase)
            else:
                atributo = nodo.atributo
                valor_atributo = X[atributo].iloc[0]
                for i, subarbol in enumerate(nodo.subs):
                    if valor_atributo == subarbol.raiz.categoria:
                        _interna(arbol.raiz.subs[i], X)

        for _, row in X.iterrows():
            _interna(self, pd.DataFrame([row]))
        
        return predicciones
    
    def altura(self) -> int:
        altura_actual = 0
        for subarbol in self.raiz.subs:
            altura_actual = max(altura_actual, subarbol.altura())
        return altura_actual + 1
    
    def imprimir(self, prefijo: str = '  ', es_ultimo: bool = True) -> None:
        nodo = self.raiz
        simbolo_rama = '└─── ' if es_ultimo else '├─── '
        split = "Split: " + str(nodo.atributo)
        rta = "Valor: " + str(nodo.categoria)
        entropia = f"Entropia: {round(self.raiz.entropia(), 2)}"
        samples = f"Samples: {len(self.raiz.data)}"
        values = f"Values: {str(self._values())}"
        clase = 'Clase:' + str(nodo.clase)
        if nodo.es_raiz():
            print(entropia)
            print(samples)
            print(values)
            print(clase)
            print(split)

            for i, sub_arbol in enumerate(nodo.subs):
                ultimo: bool = i == len(nodo.subs) - 1
                sub_arbol.imprimir(prefijo, ultimo)

        elif not nodo.es_hoja():
            print(prefijo + "│")
            print(prefijo + simbolo_rama + rta)
            prefijo2 = prefijo + " " * (len(simbolo_rama)) if es_ultimo else prefijo +"│" + " " * (len(simbolo_rama) - 1)
            print(prefijo2 + entropia)
            print(prefijo2 + samples)
            print(prefijo2 + values)
            print(prefijo2 + clase)
            print(prefijo2 + split)
            
            prefijo += ' '*10 if es_ultimo else '│' + ' '*9
            for i, sub_arbol in enumerate(nodo.subs):
                ultimo: bool = i == len(nodo.subs) - 1
                sub_arbol.imprimir(prefijo, ultimo)
        else:
            prefijo_hoja = prefijo + " "*len(simbolo_rama) if es_ultimo else prefijo + "│" + " "*(len(simbolo_rama) -1)
            print(prefijo + "│")
            print(prefijo + simbolo_rama + rta)
            print(prefijo_hoja + entropia)
            print(prefijo_hoja + samples)
            print(prefijo_hoja + values)
            print(prefijo_hoja + clase)


def accuracy_score(y_true: list[str], y_pred: list[str]) -> float:
        if len(y_true) != len(y_pred):
            raise ValueError()
        correctas = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
        precision = correctas / len(y_true)
        return precision


def probar(df, target:str):
    X = df.drop(target, axis=1)
    y = df[target]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    arbol = ArbolDecision("ID3")
    arbol.fit(x_train, y_train)
    arbol.imprimir()
    y_pred = arbol.predict(x_test)
    print(f"accuracy: {accuracy_score(y_test.tolist(), y_pred)}")
    print(f"cantidad de nodos: {len(arbol)}")
    print(f"altura: {arbol.altura()}\n")


if __name__ == "__main__":
    #https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link
    patients = pd.read_csv("TP_Final_algo2/TP_Final/cancer_patients.csv", index_col=0)
    patients = patients.drop("Patient Id", axis = 1)
    bins = [0, 15, 20, 30, 40, 50, 60, 70, float('inf')]
    labels = ['0-15', '15-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70+']
    patients['Age'] = pd.cut(patients['Age'], bins=bins, labels=labels, right=False)

    tennis = pd.read_csv("TP_Final_algo2/TP_Final/PlayTennis.csv", index_col=0)

    print("Pruebo con patients\n")
    probar(patients, "Level")
    print("Pruebo con Play Tennis\n")
    probar(tennis, "Play Tennis")
