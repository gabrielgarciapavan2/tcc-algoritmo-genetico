# -*- coding: utf-8 -*-

# Módulos a importar
import numpy as np
import pandas as pd
import math
import copy as cp
import random as rd
import datetime as dt
import sys

original_stdout = sys.stdout # Referência do stdout original

# Constantes
TOL = 1e-6
TOL_PESOS = 1e-5

# Classe de dados
class Dados:
    def __init__(self, teste, i=0):
        self.teste = teste
        self.u_csv = pd.read_csv(teste["ucsv"][i],sep=";").to_numpy()
        self.a_csv = pd.read_csv(teste["acsv"][i],sep=";").to_numpy()
        self.num_periodos, self.num_ativos = self.a_csv.shape
        self.retorno = teste["retorno"][i]
        self.min_k = teste["kmin"][i]
        self.max_k = teste["kmax"][i]
        self.min_inv = teste["invmin"][i]
        self.max_inv = teste["invmax"][i]
        self.gen_pop = teste["genpop"][i]
        self.gen_lim = teste["genlim"][i]
        self.t0 = dt.datetime.now()
        
        if len(np.argwhere(np.isnan(self.a_csv.astype(float)))) > 0:
            print(np.argwhere(np.isnan(self.a_csv.astype(float))))
            input("Erro no Dataset A")
            
        if len(np.argwhere(np.isnan(self.u_csv.astype(float)))) > 0:
            print(np.argwhere(np.isnan(self.u_csv.astype(float))))
            input("Erro no Dataset U")

    def print(self,i=0):
        print(self.teste.loc[i,:])
    
    def get_time(self):
        return dt.datetime.now() - self.t0

class Solucao:
    def __init__(self):
        self.mapa_pesos  = {}
        self.risco_array = np.array([])
        self.factivel   = False
        self.soma_pesos  = 0.0
        self.retorno    = 0.0
        self.risco      = 0.0
        self.num_ativos = 0

    def copy_status (self,sol):
        self.factivel   = sol.factivel
        self.soma_pesos  = sol.soma_pesos
        self.retorno    = sol.retorno
        self.risco      = sol.risco
        self.num_ativos = sol.num_ativos

    def print(self, disp = 'h', print_risco_array = False): 
        if disp == 'v':
            for i in self.mapa_pesos.items():
                print (i)
        else:
            print (list(self.mapa_pesos.items()))
        if print_risco_array:
            with np.printoptions(precision=5, suppress=True, linewidth=np.inf):
                print (self.risco_array)
        print(self.factivel)
        print("Soma dos Pesos:",round(self.soma_pesos,5))
        print("Retorno:",round(self.retorno,5))
        print("Risco:",round(self.risco,5))

    def print_log(self,dados): 
        print(self.factivel, end='; ')
        print(round(self.soma_pesos,5), end='; ')
        print(round(self.retorno,5), end='; ')
        print(round(self.risco,5), end='; ')
        print(dados.get_time(), end='; ')

    def __soma_pesos(self):
        self.soma_pesos = sum(self.mapa_pesos.values())
  
    def __retorno(self, dados):
        self.retorno = 0.0
        for key, value in self.mapa_pesos.items():
            self.retorno += (value * dados.u_csv[0][key])

    def __risco(self, dados):
        self.risco = 0.0
        self.risco_array = np.zeros(dados.num_periodos)
        for key,value in self.mapa_pesos.items():
            self.risco_array += value*dados.a_csv[:,key]
        self.risco = np.sum(np.abs(self.risco_array))/dados.num_periodos

    def factibilidade(self, dados):
        # Considera-se a solução factivel, até que prove o contrário
        self.factivel = True

        # Verifica a soma dos pesos
        self.__soma_pesos()
        if not math.isclose(1.0,self.soma_pesos,rel_tol=TOL,abs_tol=TOL): 
            self.factivel = False
            # input("ERRO na soma dos pesos")
        
        # Verifica o número de ativos
        if self.num_ativos < dados.min_k or self.num_ativos > dados.max_k:
            self.factivel = False

        # Verifica o retorno (principal fonte de infactibilidade)
        self.__retorno(dados)
        if self.retorno < dados.retorno:
            self.factivel = False

        # Calcula o risco    
        self.__risco(dados)

        # Retorna a factibilidade da solução
        return self.factivel

    def recalcula_factibilidade(self, dados, change):
        '''
        Recalcula os status de soluções com modificações. 
        change = [(ativo, investimento anterior, investimento atual)]
        [(1,0.5,0.75),(7,0.25,0.0)]
        '''

        # Se não há modificação, retorna a factibilidade atual da solução. 
        if len(change) < 1:
            return self.factivel
        
        # Considera-se a solução factivel, até que prove o contrário 
        self.factivel = True

        # A soma dos pesos modificados deve ser igual à soma dos pesos anteriores
        soma1, soma2 = 0.0, 0.0
        for i in change:
            soma1 += i[1]
            soma2 += i[2]
        if abs(soma1 - soma2) > TOL_PESOS:
            self.factivel = False
            print (soma1, soma2)
            # input("ERRO nos pesos recalculados")
        
        # Recalcula o número de ativos investidos
        if self.num_ativos < dados.min_k or self.num_ativos > dados.max_k:
            self.factivel = False

        # Recalcula o retorno
        for i in change:
            self.retorno = self.retorno + (i[2]-i[1])*dados.u_csv[0][i[0]]
        if self.retorno < dados.retorno:
            self.factivel = False

        # Recalcula o risco
        for i in change:
            self.risco_array = self.risco_array + (i[2]-i[1])*dados.a_csv[:,i[0]]
        self.risco = np.sum(np.abs(self.risco_array))/dados.num_periodos

        # Retorna a factibilidade
        return self.factivel
        
    def is_better(self,sol):
        '''
        Retorna True se self é melhor que sol
        Retorna Falso se self é pior ou igual a sol
        '''
        if self.factivel:
            if sol.factivel: 
                if self.risco < sol.risco: return True
            else: return True
        elif not sol.factivel and self.retorno > sol.retorno: return True
        return False

    def random_solution(self,dados):
        ativos_sorteados = rd.sample(range(dados.num_ativos),dados.max_k)
        for ativo in ativos_sorteados[0:dados.min_k]:
            self.mapa_pesos[ativo] = dados.min_inv
        self.num_ativos = dados.min_k
        peso_restante = 1.0 - dados.min_inv*dados.min_k
        rd.shuffle(ativos_sorteados)
        while (peso_restante > TOL):
            for ativo in ativos_sorteados:
                #print ("Peso restante", peso_restante)
                if ativo in self.mapa_pesos: 
                    if self.mapa_pesos[ativo] + peso_restante < dados.max_inv:
                        self.mapa_pesos[ativo] += peso_restante 
                        peso_restante = 0
                        break
                    elif rd.random() <0.5:
                        peso_restante -= (dados.max_inv - self.mapa_pesos[ativo])
                        self.mapa_pesos[ativo] = dados.max_inv
                    else: 
                        peso_sorteado = round(rd.uniform(0,dados.max_inv-self.mapa_pesos[ativo]),ndigits=5)
                        peso_restante -= peso_sorteado
                        self.mapa_pesos[ativo] += peso_sorteado
                elif peso_restante < dados.min_inv: 
                    continue
                elif peso_restante <= dados.max_inv:
                    self.mapa_pesos[ativo] = peso_restante
                    peso_restante = 0
                    self.num_ativos += 1
                    break
                elif rd.random()<0.5:
                    peso_restante -= dados.max_inv
                    self.mapa_pesos[ativo] = dados.max_inv
                    self.num_ativos += 1
                else: 
                    peso_sorteado = round(rd.uniform(dados.min_inv,dados.max_inv),ndigits=5)
                    peso_restante -= peso_sorteado
                    self.mapa_pesos[ativo] = peso_sorteado
                    self.num_ativos += 1    
            #self.print()
            #print ("Peso restante", peso_restante)
            #input("Press Enter to continue...")

    def move_all (self,dados,A,B):
        '''
        Tenta mover todo o investimento do ativo A para o Ativo B. 
        É obrigatório que o ativo A já tenha algum investimento. 
        Retorna um vetor com as tuplas de mudanças. 
        '''

        change = list()

        # Se A = 0, Retorna sem mudanças
        if (not A in self.mapa_pesos):
            return change

        # A partir daqui, A > 0

        # Se B = 0, A ==> 0, B ==> A
        if not B in self.mapa_pesos:
            change.append((B,0.0,self.mapa_pesos[A]))
            change.append((A,self.mapa_pesos[A],0.0))
            self.mapa_pesos[B] = self.mapa_pesos[A]
            del self.mapa_pesos[A]
            return change
        
        # A partir daqui, A > 0, B > 0
        
        # Se B = max_inv, Retorna sem mudanças
        if self.mapa_pesos[B] + TOL_PESOS > dados.max_inv:
            return change
        
        # A partir daqui, A > 0, 0 < B < max_inv

        # Se A + B <= max_inv, A ==> 0, B ==> B+A
        if self.mapa_pesos[A] + self.mapa_pesos[B] < dados.max_inv + TOL_PESOS:
            change.append((B,self.mapa_pesos[B],self.mapa_pesos[B] + self.mapa_pesos[A]))
            change.append((A,self.mapa_pesos[A],0.0))
            self.mapa_pesos[B] += self.mapa_pesos[A]
            del self.mapa_pesos[A]
            self.num_ativos -= 1
            return change

        # A partir daqui, A > 0, 0 < B < max_inv, A + B > maxinv
        
        # Se A+B <= inv_min + inv_max, A ==> min_inv e B ==> B + A - min_inv 
        if self.mapa_pesos[A] + self.mapa_pesos[B] < dados.min_inv + dados.max_inv + TOL_PESOS:
            # Se A = min_inv, Retorna sem mudanças
            if (self.mapa_pesos[A] < dados.min_inv + TOL_PESOS):
                return change
            change.append((B,self.mapa_pesos[B],self.mapa_pesos[B] + self.mapa_pesos[A] - dados.min_inv))
            change.append((A,self.mapa_pesos[A],dados.min_inv))
            self.mapa_pesos[B] += (self.mapa_pesos[A] - dados.min_inv)
            self.mapa_pesos[A] = dados.min_inv
            return change
        # A partir daqui, A > 0, 0 < B < max_inv, A + B > min_inv + max_inv
        # A ==> A + B - max_inv e B ==> max_inv
        else: 
            change.append((B,self.mapa_pesos[B],dados.max_inv))
            change.append((A,self.mapa_pesos[A],self.mapa_pesos[A] + self.mapa_pesos[B] - dados.max_inv))
            self.mapa_pesos[A] += (self.mapa_pesos[B] - dados.max_inv)
            self.mapa_pesos[B] = dados.max_inv
            return change
        
        return change

    def move_all_less_min_inv (self,dados,A,B):
        '''
        Tenta mover todo o investimento do ativo A para o Ativo B, exceto min_inv. 
        É obrigatório que o ativo A já tenha algum investimento. 
        Retorna um vetor com as tuplas de mudanças. 
        '''

        change = list()

        # Se A = 0, Retorna sem mudanças
        if (not A in self.mapa_pesos) or (self.mapa_pesos[A] < dados.min_inv + TOL_PESOS):
            return change
        
        # A partir daqui, A > min_inv

        # Se B = 0, A ==> min_inv, B ==> A - min_inv
        if (not B in self.mapa_pesos):
            # Se A < 2*min_inv, Retorna sem mudanças
            if self.mapa_pesos[A] + TOL_PESOS < 2*dados.min_inv:
                return change
            change.append((B,0.0,self.mapa_pesos[A]-dados.min_inv))
            change.append((A,self.mapa_pesos[A],dados.min_inv))
            self.mapa_pesos[B] = self.mapa_pesos[A]-dados.min_inv
            self.mapa_pesos[A] = dados.min_inv
            return change
        
        # A partir daqui, A > min_inv, B > 0
        
        # Se B = max_inv, Retorna sem mudanças
        if self.mapa_pesos[B] + TOL_PESOS > dados.max_inv:
            return change
        
        # A partir daqui, A > min_inv, 0 < B < max_inv
        
        # Se A+B <= inv_min + inv_max, A ==> min_inv e B ==> B + A - min_inv 
        if self.mapa_pesos[A] + self.mapa_pesos[B] < dados.min_inv + dados.max_inv + TOL_PESOS:
            change.append((B,self.mapa_pesos[B],self.mapa_pesos[B] + self.mapa_pesos[A] - dados.min_inv))
            change.append((A,self.mapa_pesos[A],dados.min_inv))
            self.mapa_pesos[B] += (self.mapa_pesos[A] - dados.min_inv)
            self.mapa_pesos[A] = dados.min_inv
            return change
        # A partir daqui, A > 0, 0 < B < max_inv, A + B > min_inv + max_inv
        # A ==> A + B - max_inv e B ==> max_inv
        else: 
            change.append((B,self.mapa_pesos[B],dados.max_inv))
            change.append((A,self.mapa_pesos[A],self.mapa_pesos[A] + self.mapa_pesos[B] - dados.max_inv))
            self.mapa_pesos[A] += (self.mapa_pesos[B] - dados.max_inv)
            self.mapa_pesos[B] = dados.max_inv
            return change
        
        return change

class Genetico:
    def __init__(self, dados):
        self.pop_size = dados.gen_pop
        self.num_geracoes = dados.gen_lim
        self.num_factivel = 0
        self.pop = list()

    def print_pop(self, n=1, stop = False):
        for i in range(n):
            self.pop[i].print()
            if stop: input("Press enter to continue...")

    def print_pop_log (self, dados, n=1):
        print (self.pop_factivel(),end=";")
        for i in range(n):
            self.pop[i].print_log(dados)
            print()

    def pop_factivel(self):
        '''
        Retorna o número de soluções factíveis na população do genético. 
        '''
        count = 0
        for i in self.pop:
            if i.factivel: count += 1
        return count

    def gerarPopInicial(self,dados):
        self.pop = list()
        for i in range(self.pop_size): #quantidade de indivíduos
            self.pop.append(Solucao())
            self.pop[i].random_solution(dados)
            self.pop[i].factibilidade(dados)
        self.pop.sort(key = lambda e: (not e.factivel, e.factivel*e.risco, -e.retorno))
        self.num_factivel = self.pop_factivel()
        print('P0',end=';')
        self.print_pop_log(dados)
        print()

    @staticmethod
    def crossover_recalc(pai_1, pai_2, dados): 
        # Copia os filhos
        filho_1, filho_2 = cp.deepcopy(pai_1), cp.deepcopy(pai_2)
        
        # Obtem a lista de ativos não-nulos
        ativos_filho_1 = list(filho_1.mapa_pesos.keys())
        ativos_filho_2 = list(filho_2.mapa_pesos.keys())
        rd.shuffle(ativos_filho_1)
        rd.shuffle(ativos_filho_2)

        # Obtem uma quantidade de troca
        qtd_troca = rd.randint(1,int(math.log2(1+min(filho_1.num_ativos,filho_2.num_ativos))))
        change1 = list()
        change2 = list()
        for i in range(qtd_troca):
            # Se os ativos forem o mesmo, não faz sentido trocar
            if ativos_filho_1[i] == ativos_filho_2[i]:
                continue
            change1.extend(filho_1.move_all(dados,ativos_filho_1[i],ativos_filho_2[i]))
            change2.extend(filho_2.move_all(dados,ativos_filho_2[i],ativos_filho_1[i]))
        filho_1.recalcula_factibilidade(dados,change1)
        filho_2.recalcula_factibilidade(dados,change2)
        del change1
        del change2
        return [filho_1, filho_2]

    @staticmethod
    def mutacao_1_recalc(dados, sol): 
        old_sol = Solucao()
        old_sol.copy_status(sol)
        ativos_sol = list(sol.mapa_pesos.keys())
        ativo_novo = rd.randint(0, dados.num_ativos - 1)
        while ativo_novo in ativos_sol:
            ativo_novo = rd.randint(0, dados.num_ativos - 1)
        ativo_velho = rd.choice(ativos_sol)
        change = sol.move_all(dados,ativo_velho,ativo_novo)
        sol.recalcula_factibilidade(dados,change)
        if not sol.is_better(old_sol):
            change = sol.move_all(dados,ativo_novo,ativo_velho)
            sol.recalcula_factibilidade(dados,change)
        del old_sol
        del change

    @staticmethod
    def mutacao_2(dados, sol): 
        old_sol = Solucao()
        old_sol.copy_status(sol)
        ativos_sol = list(sol.mapa_pesos.keys())
        ativo_novo = rd.randint(0, dados.num_ativos - 1)
        while ativo_novo in ativos_sol:
            ativo_novo = rd.randint(0, dados.num_ativos - 1)
        ativo_velho = rd.choice(ativos_sol)
        sol.move_all_less_min_inv(dados,ativo_velho,ativo_novo)
        sol.factibilidade(dados)
        if not sol.is_better(old_sol):
            sol.move_all(dados,ativo_novo,ativo_velho)
            sol.factibilidade(dados)
        del old_sol

    def run(self, dados):
        for geracao in range(1,self.num_geracoes+1): #quantidade de gerações
            pop_shuffle = list(range(self.pop_size))
            rd.shuffle(pop_shuffle)
            for i in range(0,self.pop_size,2):
                pai_1 = self.pop[pop_shuffle[i]]
                pai_2 = self.pop[pop_shuffle[i+1]]
                self.pop.extend (self.crossover_recalc(pai_1,pai_2,dados))
            self.pop.sort(key = lambda e: (not e.factivel, e.factivel*e.risco, -e.retorno))
            del self.pop[self.pop_size:]

            individuos_sorteados = rd.sample(range(self.pop_size),int(0.2*self.pop_size))
            for i in individuos_sorteados:                
                # if rd.random() < 0.5:
                    self.mutacao_1_recalc (dados, self.pop[i])
                # else:
                    # self.mutacao_2(dados, self.pop[i])
            self.pop.sort(key = lambda e: (not e.factivel, e.factivel*e.risco, -e.retorno))
            self.num_factivel = self.pop_factivel()
            print(f'P{geracao}',end=';')
            self.print_pop_log(dados)
            
rd.seed(1223649)
teste = pd.read_csv("Teste.csv",sep=";")

for i in range(2199,2496):
    with open(f'logs/log-{i}.txt', 'w') as f:
        sys.stdout = f
        dados = Dados(teste,i)
        genetico = Genetico(dados)
        genetico.gerarPopInicial(dados)
        genetico.run(dados)
        sys.stdout = original_stdout
        print(f'Teste {i} finalizado! *****')
    