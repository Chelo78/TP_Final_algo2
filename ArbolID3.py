from typing import Generic, Optional, TypeVar, Callable, Any 
from functools import wraps
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from copy import deepcopy

class Nodo:
    def __init__(self, data: pd.DataFrame, target: pd.Series) -> None: #data : data de entrenamiento
        self.atributo: Optional[str] = None # se setea cuando se haga un split
        self.categoria: Optional[str] = None # lo mismo
        self.data: pd.DataFrame = data
        self.target: pd.Series = target
        self.clase: Optional[str] = None # cuando sea hoja deberia tener la clase predicha
        self.subs: list[ArbolID3] = []

    def split(self, atributo: str) -> None:
        for categoria in self.data[atributo].unique():
            nueva_data = self.data[self.data[atributo] == categoria]
            nuevo_target = self.target[self.data[atributo] == categoria]
            nuevo = Nodo(nueva_data, nuevo_target)
            nuevo.atributo = atributo
            nuevo.categoria = categoria
            self.subs.append(ArbolID3(nuevo))
            
    
    def entropia(self) -> float:
        entropia = 0
        proporciones = self.target.value_counts(normalize= True)
        target_categorias = self.target.unique()
        for c in target_categorias:
            proporcion = proporciones.get(c, 0)
            entropia += proporcion * np.log2(proporcion)
        return -entropia
    
class ArbolID3:
    def __init__(self, nodo: Nodo) -> None:
        self.raiz: Nodo = nodo

    @staticmethod
    def crear_arbol(df: pd.DataFrame, target: pd.Series):
        nodo = Nodo(df, target)
        return ArbolID3(nodo)
    
    def _mejor_split(self) -> str:
        mejor_ig = -1
        mejor_atributo = None
        atributos = self.raiz.data.columns

        for atributo in atributos:
            for categoria in self.raiz.data[atributo].unique():
                ig = self.information_gain(atributo)
                if ig > mejor_ig:
                    mejor_ig = ig
                    mejor_atributo = atributo
        
        return mejor_atributo

    def fit(self) -> None:
        print('1\n',)
        if len(self.raiz.target.unique()) == 1 or len(self.raiz.data.columns) == 0:
            self.raiz.clase = self.raiz.target.value_counts().idxmax()
        else:
            mejor_atributo = self._mejor_split()
            self.raiz.split(mejor_atributo)
            [sub_arbol.fit() for sub_arbol in self.raiz.subs]
            
        
    def information_gain(self, atributo: str) -> float:
    # Recopilar información necesaria para el cálculo
        entropia_actual = self.raiz.entropia()
        len_actual = len(self.raiz.data)
        information_gain = entropia_actual

        # Hacer el split: "atributo = categoria ?"
        nuevo = deepcopy(self)
        nuevo.raiz.split(atributo)

        entropias_subarboles = []
        lens_subarboles = []
        for sub_arbol in nuevo.raiz.subs:
            entropia_subarbol, len_subarbol = self._calcular_entropia_longitud(sub_arbol)
            entropias_subarboles.append((len_subarbol / len_actual) * entropia_subarbol)
            lens_subarboles.append(len_subarbol)

        information_gain -= sum(entropias_subarboles)
        return information_gain

    def _calcular_entropia_longitud(self, sub_arbol) -> tuple[float, int]:
        if sub_arbol.raiz.clase:
            # Si el nodo es una hoja, devolvemos la entropía como 0 y la longitud de los datos en el nodo
            return 0, len(sub_arbol.raiz.data)
        else:
            # Si el nodo no es una hoja, calculamos la entropía y la longitud recursivamente para cada subárbol
            entropia_subarbol = sub_arbol.raiz.entropia()
            len_subarbol = 0
            for sub_sub_arbol in sub_arbol.raiz.subs:
                entropia, longitud = self._calcular_entropia_longitud(sub_sub_arbol)
                entropia_subarbol += entropia
                len_subarbol += longitud
            return entropia_subarbol, len_subarbol
    
    def imprimir(self, prefijo: str = '  ', es_ultimo: bool = True, es_raiz: bool = True) -> None:
        nodo = self.raiz
        simbolo_rama = '└─no── ' if es_ultimo else '├─si── '
        if es_raiz:
            print(str(nodo.atributo) + " = " + str(nodo.categoria) + "?")
            for i, sub_arbol in enumerate(nodo.subs):
                sub_arbol.imprimir(prefijo, i == len(nodo.subs) - 1, False)
        elif nodo.atributo is not None:
            print(prefijo + simbolo_rama + str(nodo.atributo) + " = " + str(nodo.categoria) + "?")
            prefijo += ' '*10 if es_ultimo else '│' + ' '*9
            for i, sub_arbol in enumerate(nodo.subs):
                sub_arbol.imprimir(prefijo, i == len(nodo.subs) - 1, False)
        else:
            print(prefijo + simbolo_rama + 'Clase:', str(nodo.clase))


    def predict(self, X: pd.DataFrame) -> list[str]:
        predicciones = []

        def _interna(arbol, X):
            nodo = arbol.raiz
            if nodo.clase is not None:  # es hoja
                predicciones.append(nodo.clase)
            else:
                atributo = nodo.atributo
                categoria = nodo.categoria
                valor_atributo = X[atributo].iloc[0]
                if valor_atributo == categoria:
                    _interna(arbol.raiz.si, X)
                else:
                    _interna(arbol.raiz.sd, X)

        for _, row in X.iterrows():
            _interna(self, pd.DataFrame([row]))
        
        return predicciones


if __name__ == "__main__":
    #https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link
    #df = pd.read_csv("G:/algo2/TP_Final/cancer_patients.csv", index_col=0)
    
    df = pd.read_csv("C:/Documentos/n67745/Repositorios GitHub/UNSAM/TP Algoritmos 2/TP_Final_algo2/cancer_patients.csv", index_col=0)
    df = df.drop("Patient Id", axis = 1)
    bins = [0, 15, 20, 30, 40, 50, 60, 70, float('inf')]
    labels = ['0-15', '15-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70+']
    df['Age'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

    df = df.drop('Snoring', axis=1)
    df = df.drop('Dry Cough', axis=1)
    df = df.drop('Frequent Cold', axis=1)
    df = df.drop('Clubbing of Finger Nails', axis=1)
    df = df.drop('Swallowing Difficulty', axis=1)
    df = df.drop('Wheezing', axis=1)
    
    X = df.drop('Level', axis=1)
    y = df['Level']

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    arbol = ArbolID3.crear_arbol(x_train, y_train)

    arbol.fit() # acá deberian ir x_train e y_train, no en crear_arbol

    arbol.imprimir()

    # y_pred = arbol.predict(x_test)

    # def accuracy_score(y_true: list[str], y_pred: list[str]) -> float:
    #     if len(y_true) != len(y_pred):
    #         raise ValueError()
    #     correctas = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    #     precision = correctas / len(y_true)
    #     return precision
    
    # print(accuracy_score(y_test.tolist(), y_pred))




