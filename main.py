# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 12:14:37 2022

@author: omer
"""
import sys
import GA
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from MainPy import *
import numpy as np

#----------------------UYGULAMA OLUŞTUR-------------------#
#---------------------------------------------------------#
Uygulama=QApplication(sys.argv)
penAna=QMainWindow()
ui=Ui_MainWindow()
ui.setupUi(penAna)
penAna.show()
ui.sonucTablo.setColumnWidth(0, 75)
ui.sonucTablo.setColumnWidth(1, 150)
ui.sonucTablo.setColumnWidth(2, 843)
ui.sonucTablo.setColumnWidth(3, 150)


def Hesapla():
    girdiler = ui.girdilerEdit.text()
    girdiArray = np.array(girdiler.split(","))

    equation_inputs = girdiArray.astype(np.float)   #girdiler
    num_weights = int(ui.agirlikEdit.text())   #Ağırlıkların sayısı, 

    #Bir sonraki adım, başlangıç ​​popülasyonunu tanımlamaktır. 
    import numpy
    sol_per_pop = int(ui.populasyonEdit.text())  #Popülasyon başına çözüm sayısı
    pop_size = (sol_per_pop,num_weights) #Herbiri 6 adet genden oluşan 8 kromozom

    #Başlangıç popülasyonunun numpy.random.uniform ile random oluşturulması
    new_population = numpy.random.uniform(low=float(ui.minEdit.text()), high=float(ui.maxEdit.text()), size=pop_size)
    print(new_population)

    num_generations = int(ui.iterasyonEdit.text())
    num_parents_mating = int(ui.havuzEdit.text())  #eşleşme havuzundaki birey sayısı
    for generation in range(num_generations):
        ui.sonucTablo.setRowCount(generation+1)
        
        
        print("Generation : ", generation)
        # Popülasyondaki her kromozom için uygunluk değerini hesapla
        fitness = GA.cal_pop_fitness(equation_inputs, new_population)
        #eşleşme havuzundaki en iyi bireylerin seçimi
        parents = GA.select_mating_pool(new_population, fitness, num_parents_mating)
        
        #Çaprazlama ile yeni birey üretimi
        offspring_crossover = GA.crossover(parents,
                                           offspring_size=(pop_size[0]-parents.shape[0], num_weights))
        #Mutasyon uygulanması
        offspring_mutation = GA.mutation(offspring_crossover)
        
        #Yeni popülasyon oluşturulması
        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation
        
        #Geçerli iterasyondaki en iyi sonuç
        print("Best result : ", numpy.max(numpy.sum(new_population*equation_inputs, axis=1)))
        #Tüm nesilleri bitirmeyi yineledikten sonra en iyi çözümü elde etmek için 
        #İlk olarak, son nesildeki her bir çözüm için uygunluk hesaplanır.
        fitness = GA.cal_pop_fitness(equation_inputs, new_population)
        
        #Ardından, bu çözümün en iyi uygunluğa karşılık gelen indeksini döndürün.
        best_match_idx = numpy.where(fitness == numpy.max(fitness))
        print("Best solution : ", new_population[best_match_idx, :])
        print("Best solution fitness : ", fitness[best_match_idx])
        enIyiCozum = str(new_population[best_match_idx, :])
        enIyiCozum = enIyiCozum.replace("[", "")
        enIyiCozum= enIyiCozum.replace("]", "")
        ui.sonucTablo.setItem(generation, 0, QTableWidgetItem(str(generation+1)))
        ui.sonucTablo.setItem(generation, 1, QTableWidgetItem(str(numpy.max(numpy.sum(new_population*equation_inputs, axis=1)))))
        ui.sonucTablo.setItem(generation, 2, QTableWidgetItem(enIyiCozum))
        ui.sonucTablo.setItem(generation, 3, QTableWidgetItem(str(fitness[best_match_idx])))

ui.hesaplButon.clicked.connect(Hesapla)

Uygulama.exec_()

#Uygulama.exec_()
#sys.exit(Uygulama.exec_())