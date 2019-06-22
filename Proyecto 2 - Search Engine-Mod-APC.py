#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:46:46 2019

@author: Jesús Alejandro Rizo Domínguez
"""

import os
import re
import string
import pandas as pd
from nltk.corpus import stopwords
from TMalexlib import RepeatReplacer
from TMalexlib import Tokenize
from TMalexlib import computeTF
from TMalexlib import computeIDF
from TMalexlib import computeTFIDF
from nltk.stem import LancasterStemmer
import numpy as np


bag_of_tokens = {}
#lista_archivo = []
# 1
def getListaDePalabras(files):

    lista_de_palabras = []
    for file in files:
        lista_archivo = file.read().lower().split()
        for word in lista_archivo:
            lista_de_palabras.append(word)
    #print(lista_de_palabras)
    return lista_de_palabras
# 2
def quitaSignosPuntuacion(lista_de_palabras):
    
    x = re.compile('[%s]' % re.escape(string.punctuation))
    lista_palabras_sin_puntuacion = []
    for word in lista_de_palabras:
        word_sp = re.sub(x,'',word)     # word_sp = word sin puntuación
        if word_sp != '':
            lista_palabras_sin_puntuacion.append(word_sp)    
    #print(lista_palabras_sin_puntuacion)
    return  lista_palabras_sin_puntuacion
#
# 3
def quitaStopWords(lista_palabras_sin_puntuacion):
    stops = set(stopwords.words('english'))
    lista_palabras_sp_ssw = []  #sin puntuación y sin stopwords
    for word in lista_palabras_sin_puntuacion:
        if word not in stops:
            lista_palabras_sp_ssw.append(word)
    #print(lista_palabras_sp_ssw)
    return lista_palabras_sp_ssw

def quitaCaracteresRepetidos(lista_palabras_sp_ssw):
    lista_palabras_sp_ssw_scr = []
    reemplazador = RepeatReplacer()
    for word in lista_palabras_sp_ssw:
        lista_palabras_sp_ssw_scr.append(reemplazador.replace(word))
    #print(lista_palabras_sp_ssw_scr)
    return lista_palabras_sp_ssw_scr

def lemmatization(lista_palabras_sp_ssw_scr):
    stemmerlan = LancasterStemmer()
    lista_palabras_sp_ssw_scr_stem = []
    for word in lista_palabras_sp_ssw_scr:
        lista_palabras_sp_ssw_scr_stem.append(stemmerlan.stem(word))
    tokens = set(lista_palabras_sp_ssw_scr_stem)
    
    #print(tokens)
    return tokens
    
    
def preProcess(files):

    # 2.1 Primero genero una lista con todas las palabras separadas por espacio
    # en minúsculas    
    lista_de_palabras = getListaDePalabras(files)            
    #print(lista_de_palabras)
    # 2.2 Quitando los signos de puntuación
    lista_palabras_sin_puntuacion = quitaSignosPuntuacion(lista_de_palabras)
    #print(lista_palabras_sin_puntuacion)        
    # 2.3 Quitando las stop words
    lista_palabras_sp_ssw = quitaStopWords(lista_palabras_sin_puntuacion)
    #print(lista_palabras_sp_ssw)        
    # 2.4 Quitando caracteres repetidos
    lista_palabras_sp_ssw_scr = quitaCaracteresRepetidos(lista_palabras_sp_ssw)     
    #print(lista_palabras_sp_ssw_scr)
    # 2.5 Aplicando Lemmatization con Lancaster Stemmer
    tokens = lemmatization(lista_palabras_sp_ssw_scr)
    return tokens

def processQuery(qry):
    lista_de_palabras = []
    for word in qry.lower().split():
        lista_de_palabras.append(word)
    #print(lista_de_palabras)        
    # 2.2 Quitando los signos de puntuación
    lista_palabras_sin_puntuacion = quitaSignosPuntuacion(lista_de_palabras)
    #print(lista_palabras_sin_puntuacion)        
    # 2.3 Quitando las stop words
    lista_palabras_sp_ssw = quitaStopWords(lista_palabras_sin_puntuacion)
    #print(lista_palabras_sp_ssw)        
    # 2.4 Quitando caracteres repetidos
    lista_palabras_sp_ssw_scr = quitaCaracteresRepetidos(lista_palabras_sp_ssw)     
    #print(lista_palabras_sp_ssw_scr)
    # 2.5 Aplicando Lemmatization con Lancaster Stemmer
    tokens = lemmatization(lista_palabras_sp_ssw_scr)
    
    return tokens


if __name__ == '__main__':
    # 1. Generando una lista de objetos tipo archivo.
    # path = '/home/alex/OneDrive/Iteso/MSC/Text Mining/Search Engine/Example Files/'
    # path = '/home/alex/OneDrive/Iteso/MSC/Text Mining/Search Engine/Stories Files/'
    
    path = os.getcwd() + '\\Examples\\'
    files = []
    for filename in os.listdir(path):
        files.append(open(path+filename, 'r'))
        
    print(files)
    # 2. Generando el bag of tokens
    
    tokens = preProcess(files)
    
    # 3. Enter query 
    tokqry = processQuery(input("Enter query for search: "))
    print("Tokenized Query: {0}".format(tokqry))
    
    
    
    print(tokens)  
    
    #add tokens from qry + docs
    tokens = tokens.union(tokqry)
    
    
    wordDict = []
    for file in range(len(files)):
        wordDict.append(dict.fromkeys(tokens, 0))
    
    documentBOW = []
    print(wordDict)
    for i in range(len(files)):
        files[i].seek(0)
        documentBOW.append([])
        lista_archivo = files[i].read().lower().split()
        for word in lista_archivo:
            wordToken = Tokenize(word)
            if(wordToken != False):
                wordDict[i][wordToken]+=1
                documentBOW[i].append(wordToken)
                
    for dic in wordDict:
        print(dic, '\n')

    """
    qryBOW = []
    for word in tokqry:
        qryBOW[i].append(word)
    """
    
    
    #%%
    # En este punto ya tenemos en la variable wordDict el arreglo con el diccionario
    # de palabras para cada archivo
    
    listaTFs = []
    for i in range(len(documentBOW)):
        listaTFs.append(computeTF(wordDict[i], documentBOW[i]))
        
    IDFs = computeIDF(wordDict)
    #print(IDFs)
    listaTFIDFs = []
    for tf in listaTFs:
        listaTFIDFs.append(computeTFIDF(tf,IDFs))
    
    print(pd.DataFrame(listaTFIDFs))
        
    #%%    
    for file in files:
        file.close()
        
        