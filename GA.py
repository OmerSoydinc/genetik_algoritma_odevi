# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 11:54:57 2022

@author: omer
"""

import numpy



def cal_pop_fitness(equation_inputs, pop):
    """
    Fitness fonksiyonu, her bir girdi ile buna karşılık gelen ağırlıkları çarpar
    ve toplar.
    """
    fitness = numpy.sum(pop*equation_inputs, axis=1)
    return fitness


def select_mating_pool(pop, fitness, num_parents):
    """
    Eşleşme havuzunda yeni bireylerin tutulması için parents adlı boş dizi
oluşturulur. max_fitness_idx ile en iyi fitness değerine sahip birey tutulur.
Bu indekse karşılık elde edilen çözüm parents dizisine aktarılır. Bu çözümü 
tekrar seçmemek için uygunluk değeri –999999999999 olarak ayarlanır ki bu çok
küçük bir değerdir. Bu değer, çözümün tekrar seçilme olasılığını ortadan kaldırır.   
Gerekli parent sayısı seçildikten sonra, parents dizisi geri döndürülür.
    """
    parents = numpy.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents


def crossover(parents, offspring_size):
    """
    Tek noktalı çaprazlama kullandığımız için, çaprazlamanın gerçekleştiği 
    noktayı belirtmemiz gerekiyor. Çözümü iki eşit parçaya bölmek için nokta 
    seçilir. O zaman çaprazlamak için iki ebeveyni seçmemiz gerekiyor. 
    Bu ebeveynlerin indeksleri, parent1_idx ve parent2_idx'te saklanır.
    """
    offspring = numpy.empty(offspring_size)
    # İki ebeveyn arasında geçişin gerçekleştiği nokta. Genellikle merkezdedir.
    crossover_point = numpy.uint8(offspring_size[1]/2)
    for k in range(offspring_size[0]):
        # Eşleşen ilk ebeveynin indeksi.
        parent1_idx = k%parents.shape[0]
        # Eşleştirilecek ikinci ebeveynin indeksi.
        parent2_idx = (k+1)%parents.shape[0]
        # Yeni yavru, genlerinin ilk yarısını ilk ebeveynden almış olacak.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # yeni yavru, genlerinin ikinci yarısını ikinci ebeveynden alacak.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring


def mutation(offspring_crossover):
    """
    Mutasyon, her yavruda rastgele tek bir geni değiştirir.Bunun için her
    yavru bireyde döngü ile ulaşılır ve –1.0 ile 1.0 aralığında rastgele
    oluşturulmuş bir sayı, rastgele oluşturulmuş bir indekse eklenir.
    Sonuçlar “offspring_crossover” değişkeninde saklanır ve 
    fonksiyon tarafından döndürülür.
    """
    for idx in range(offspring_crossover.shape[0]):
        # Gene eklenecek rastgele değer.
        random_value = numpy.random.uniform(-1.0, 1.0, 1)
        offspring_crossover[idx, 4] = offspring_crossover[idx, 4] + random_value
    return offspring_crossover