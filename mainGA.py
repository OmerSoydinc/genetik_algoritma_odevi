# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 11:54:59 2022

@author: omer
"""

# =============================================================================
# Y = w1x1 + w2x2 + w3x3 + w4x4 + w5x5 + w6x6
# Girdi değerleri (x1,x2,x3,x4,x5,x6)=(4,–2,7,5,11,1)
# Bu denklemi maksimize eden parametreleri (ağırlıkları) bulmaya çalışıyoruz. 
# Pozitif giriş, olası en büyük pozitif sayı ile çarpılacak ve 
# negatif sayı, mümkün olan en küçük negatif sayı ile çarpılacaktır.
# GA, pozitif girdilerle pozitif ağırlıkları ve negatif girdilerle 
# negatif ağırlıkları kullanmanın daha iyi olduğunu bilmelidir.
# =============================================================================

import GA

equation_inputs = [4,-2,3.5,5,-11,-4.7]   #girdiler
num_weights = 6   #Ağırlıkların sayısı, 

#Bir sonraki adım, başlangıç ​​popülasyonunu tanımlamaktır. 
import numpy
sol_per_pop = 8  #Popülasyon başına çözüm sayısı
pop_size = (sol_per_pop,num_weights) #Herbiri 6 adet genden oluşan 8 kromozom

#Başlangıç popülasyonunun numpy.random.uniform ile random oluşturulması
new_population = numpy.random.uniform(low=-4.0, high=4.0, size=pop_size)
print(new_population)

num_generations = 100
num_parents_mating = 4  #eşleşme havuzundaki birey sayısı
for generation in range(num_generations):
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

