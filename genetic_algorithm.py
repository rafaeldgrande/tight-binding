
from __future__ import division
import numpy as np
from scipy.interpolate import interp1d
import time
# import random
# import runpy
import os

np.random.seed(2)

# from joblib import Parallel, delayed
# import multiprocessing
# num_cores = 4    #multiprocessing.cpu_count()
# print('processadores', num_cores)

import matplotlib.pyplot as plt
plt.rc('text', usetex=True)

from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('Resultados.pdf')

# Parametros

VERBOSITY = True

# arq_ref = 'TB_ntcs/Dispersao_ntcH_4cells_DFT_ref'
arq_ref = '10_0/Dispersao_ntc_10_0_QP.out'
#arq_ref = '/home/rafael/Dropbox/Research/Excitons_CNTs/cnt_8_0/pure/GWbse_ptsK_24co_192fi/8-absorption_z/Dispersao_ntc_8_0_DFT.out'
# arq_ref = 'Dispersao.out_teste'
arq_tb = 'Dispersao_cell_ntc_10_0.out'

Max_iteracoes = 5
Min_Passos = 10

Tot_individuos = 20

Nparents = int(np.sqrt(Tot_individuos))
Tot_individuos_novas_ger = Nparents**2
Nparents_surv = int(0.1*Tot_individuos)
Populacao = []

taxa_mut = 0.01  # Taxa de mutacao
var_mut = 0.2

Erros_min = []
Erro_max_parents = []
Erros_mean = []
Erros_mean_parents = []
Best_params = []

# faixa_params = [[0.0, 0.3], [-3.0, -2.2], [-1.0, 0.0], [-1.0, 0.0]]  # Parametros grafeno - overlap, hopping 1s viz, hopping 2s viz e hopping 3s viz
# faixa_params = [[0.0, 0.3], [-4.0, -3.0], [-0.6, 0.0], [-0.5, 0.0]]  # Paramtros ntc
faixa_params = [[0.14, 0.15], [-4.4, -4.35], [-0.65, -0.63], [-0.1, -0.05]]
faixa_params = [[0.0, 0.4], [-6.5, 0], [-0.9, 0.0], [-0.9, 0.0]]
faixa_params = [[0.0, 0.4], [-6.5, 0]]
#faixa_params = [[0.10, 0.13], [-3.2, -3.17], [-0.6, -0.0], [-0.6, -0.00]]  # Parametros ntc
# faixa_params = [[11, 17], [-14, -8]] # CNT-H -> Energia on site da impureza e hopping impureza ao carbono q esta ligado

Npontos_interpolacao = 200

for i in range(len(faixa_params)):
    Best_params.append([])

conv_std_over_mean = 1e-3
conv_erro_medio = 10.0
tolerancia = 1e-5
tol_diff_erro_min = 1e-2

arq_log = open('log', 'w')
arq_log.write('Erro p0 p1 p2 ...\n')

def escreve_params(individuo, arq_params):

    arq_out = open(arq_params, 'w')

    arq_out.write("\n# Especies atomicas\n")  # grafeno e nanotubos
    arq_out.write("A 6 \n \n")
    arq_out.write("# Energias on site\n")
    arq_out.write("E0 6   0.0 \n\n")
    arq_out.write("# Overlap \n")
    arq_out.write("S 6 6     "+str(individuo[0])+" \n\n")
    arq_out.write("# Hopping intra \n")
    # arq_out.write("t0 6 6  "+str(individuo[1])+"  "+str(individuo[2])+"  "+str(individuo[3])+"\n\n")   # 3 Vizinhos
    # arq_out.write("t0 6 6  "+str(individuo[1])+"  "+str(individuo[2])+"  0.0 \n\n")  # 2 vizinhos
    arq_out.write("t0 6 6  "+str(individuo[1])+"  0.0 0.0 \n\n")  # 1 vizinho
    arq_out.write("# Hopping inter\n")
    arq_out.write("t1 6 6      -0.3 0.0 0.0\n")


    # arq_out.write("\n# Especies atomicas\n")  # Nanotubo com impureza
    # arq_out.write("A 1 6 6.1 \n \n")
    # arq_out.write("# Energias on site\n")
    # arq_out.write("E0 1    "+str(individuo[0])+"\n")
    # arq_out.write("E0 6    0.0 \n")
    # arq_out.write("E0 6.1  0.0 \n\n")
    # # arq_out.write("E0 6.1    "+str(-individuo[0])+"\n")
    # arq_out.write("# Overlap \n")
    # arq_out.write("S 6 6       0.19093920302507722 \n")
    # arq_out.write("S 6 6.1     0.19093920302507722 \n")
    # arq_out.write("S 6.1 6     0.19093920302507722 \n")
    # arq_out.write("S 6.1 6.1   0.19093920302507722 \n\n")
    # arq_out.write("# Hopping intra \n")
    # arq_out.write("t0 6 6      -2.5960098670505882  -0.41258481810654984  -0.1630278281457373 \n")
    # arq_out.write("t0 6 6.1    -2.5960098670505882  -0.41258481810654984  -0.1630278281457373 \n")
    # arq_out.write("t0 6.1 6    -2.5960098670505882  -0.41258481810654984  -0.1630278281457373 \n")
    # arq_out.write("t0 6.1 6.1  -2.5960098670505882  -0.41258481810654984  -0.1630278281457373 \n")
    #
    # arq_out.write("# Hopping inter\n")
    # arq_out.write("t1 1 6.1 "+str(individuo[1])+"\n")
    # arq_out.write("t1 6.1 1 "+str(individuo[1])+"\n")

    arq_out.close()

def ler_dispersoes(arq_disp):

    Q, E = [], []
    Q_out, E_out = [], []
    arq = open(arq_disp)

    for line in arq:
        linha = line.split()
        if len(linha) == 0:
            Q.append(np.array([]))
            E.append(np.array([]))
        elif len(linha) == 2:
            Q[-1] = np.append(Q[-1], float(linha[0]))
            E[-1] = np.append(E[-1], float(linha[1]))

    for banda in range(len(Q)):

        Qmax = max(Q[banda])
        Q[banda] = Q[banda]/Qmax

        interpolacao = interp1d(Q[banda], E[banda], kind='cubic')
        Q_out.append(np.linspace(0, 1.0, num=Npontos_interpolacao, endpoint=True))
        E_out.append(interpolacao(Q_out[-1]))

    return Q_out, E_out

def compara_dispersoes(arq_ind, E_ref):

    Q_ind, E_ind = ler_dispersoes(arq_ind)

    erro = 0

    tot_kpoints = len(E_ind[0])

    # for banda in range(len(Q_ind)):
    #     for k in range(len(Q_ind[banda])):
    #         erro += (1/tot_kpoints)*(E_ind[banda][k] - E_ref[banda][k])**2

    # So olhando pontos Gamma e X

    fator = 0.001

    for banda in range(len(Q_ind)):
        if banda > 0:
            if E_ref[banda - 1][0] == 0.0:  # Banda de valência que tem energia 0 em gamma
                erro += (E_ind[banda][0] -  E_ref[banda][0] )**2
                erro += fator*(E_ind[banda][-1] -  E_ref[banda][-1] )**2
            else:
                erro += fator*(E_ind[banda][0] -  E_ref[banda][0] )**2
                erro += fator*(E_ind[banda][-1] -  E_ref[banda][-1] )**2
        else:
            erro += fator*(E_ind[banda][0] -  E_ref[banda][0] )**2
            erro += fator*(E_ind[banda][-1] -  E_ref[banda][-1] )**2

    erro = np.sqrt(erro)


    return erro


def Crossover(Parents):

    Pop_temp = []

    # for r in range(Nparents_surv):
    #
    #     Pop_temp.append([])
    #
    #     for param in range(len(faixa_params)):
    #
    #         valor_param = Parents[r][param]
    #         # Mutacao
    #         # chance = np.random.uniform(low=0, high=1)
    #         # if chance <= taxa_mut:
    #         #     # valor_param = np.random.uniform(faixa_params[param][0], faixa_params[param][1])
    #         #     # valor_param += np.random.uniform(low=-1, high=1)*variacao
    #         #     variacao = (faixa_params[param][1]-faixa_params[param][0])*var_mut
    #         #     valor_param = np.random.normal(loc=valor_param, scale=variacao)
    #         Pop_temp[-1].append(valor_param)



    # for r in range(Tot_individuos-Nparents_surv):
    #
    #     Pop_temp.append([])
    #
    #     ind1, ind2 = np.random.randint(low=0, high=Nparents-1), np.random.randint(low=0, high=Nparents-1)
    #
    #     # peso = np.random.uniform(low=0, high=1.0)
    #     peso = Erros_em_ordem[ind2]/(Erros_em_ordem[ind1]+Erros_em_ordem[ind2])
    #
    #     for param in range(len(faixa_params)):
    #
    #         valor_param = Parents[ind1][param]*peso + Parents[ind2][param]*(1-peso)
    #
    #         # Mutacao
    #         chance = np.random.uniform(low=0, high=1)
    #         if chance <= taxa_mut:
    #             # valor_param = np.random.uniform(faixa_params[param][0], faixa_params[param][1])
    #             # valor_param += np.random.uniform(low=-1, high=1)*variacao
    #             variacao = (faixa_params[param][1]-faixa_params[param][0])*var_mut
    #             valor_param = np.random.normal(loc=valor_param, scale=variacao)
    #
    #         Pop_temp[-1].append(valor_param)

    for ind1 in range(Nparents):
        for ind2 in range(ind1, Nparents):

            Pop_temp.append([])

            # ind1, ind2 = np.random.randint(low=0, high=Nparents-1), np.random.randint(low=0, high=Nparents-1)

            # peso = np.random.uniform(low=0, high=1.0)
            peso = Erros_em_ordem[ind2]/(Erros_em_ordem[ind1]+Erros_em_ordem[ind2])

            for param in range(len(faixa_params)):

                valor_param = Parents[ind1][param]*peso + Parents[ind2][param]*(1-peso)

                # Mutacao
                chance = np.random.uniform(low=0, high=1)
                if chance <= taxa_mut:
                    # valor_param = np.random.uniform(faixa_params[param][0], faixa_params[param][1])
                    # valor_param += np.random.uniform(low=-1, high=1)*variacao
                    variacao = (faixa_params[param][1]-faixa_params[param][0])*var_mut
                    valor_param = np.random.normal(loc=valor_param, scale=variacao)

                Pop_temp[-1].append(valor_param)

    return Pop_temp


def run_tight_binding(Individuo, arq_params):

    escreve_params(Individuo, arq_params)

    try:
        #runpy.run_path('tight_binding_GA.py')
        # os.system('python ../tight_binding.py '+arq_params)
        os.system('python ../tight_binding.py')
    except:
        print('Deu ruim')
        pass

    try:
        erro = compara_dispersoes(arq_tb, E_ref)
    except:
        erro = 200

    if erro == 0.0:
        erro = 500

    return erro

# Construindo populacao inicial

for i in range(Tot_individuos):
    Populacao.append([])
    for j in range(len(faixa_params)):
        vmin, vmax = faixa_params[j][0], faixa_params[j][1]
        Populacao[-1].append(np.random.uniform(low=vmin, high=vmax))

# Carregando dispersao de referencia

Q_ref, E_ref = ler_dispersoes(arq_ref)

# Comecando o algoritmo genetico

passo = 0

while passo <= Max_iteracoes:

    print('Passo '+str(passo)+' de '+str(Max_iteracoes))
    print('Total individuos: '+str(len(Populacao)))

    # Calculando dispersao para cada individuo

    # Erros = Parallel(n_jobs=num_cores)(delayed(run_tight_binding)(individuo, 'tight_binding_params_'+str(Populacao.index(individuo))) for individuo in Populacao)
    Erros = [run_tight_binding(individuo, 'tight_binding_params') for individuo in Populacao]
    # Erros = []
    # Tot_individuos = len(Populacao)
    # for i in range(Tot_individuos):
    #     Erros.append(run_tight_binding(Populacao[i], 'tight_binding_params_'+str(i)))

    # Pegando melhor individuo dessa geracao

    ind_min = Erros.index(min(Erros))
    best_indiv = Populacao[ind_min]
    for param in range(len(Populacao[ind_min])):
        Best_params[param].append(Populacao[ind_min][param])

    run_tight_binding(best_indiv, 'tight_binding_params')  # Gerando melhor dispersao dessa geracao para ser plotada dps
    Q_ind, E_ind = ler_dispersoes(arq_tb)

    # Comparando com melhor individuo de toda simulacao

    if passo == 0:
        best_indiv_global = best_indiv
        passo_best_ind_global = passo
        Erro_best_ind_global = min(Erros)
    else:
        if Erro_best_ind_global > min(Erros):
            best_indiv_global = best_indiv
            passo_best_ind_global = passo
            Erro_best_ind_global = min(Erros)

    # Plotando graficos

    plt.figure(1)
    plt.plot([passo for i in range(len(Erros))], Erros, 'ro')
    # plt.plot([passo], np.mean(Erros), 'kx')
    # plt.plot([passo], min(Erros), 'bs')

    for j in range(len(faixa_params)):
        plt.figure(j+2)
        plt.plot([passo for i in range(len(Populacao))], [Populacao[i][j] for i in range(len(Populacao))], 'ro')
        plt.plot([passo], [Populacao[ind_min][j]], 'bo')

    plt.figure(len(faixa_params) + 2 + passo)
    plt.title('Passo '+str(passo), fontsize=18)
    for j in range(len(Q_ref)):
        plt.plot(Q_ref[j], E_ref[j], 'k-')
        plt.plot(Q_ind[j], E_ind[j], 'r--')

    plt.plot([0], [0], 'k-', label='Ref')
    plt.plot([0], [0], 'r--', label='TB')

    plt.legend()
    plt.tight_layout()
    plt.savefig(pp, format='pdf')
    plt.close()


    # Escrevendo log
    arq_log.write(str(min(Erros))+' ')
    for param in Populacao[Erros.index(min(Erros))]:
        arq_log.write(str(param)+' ')
    arq_log.write('\n')


    # Determinando pais da proxima geracao

    Parents = []

    # print(Erros)
    Erros_em_ordem = Erros.copy()
    Erros_em_ordem.sort()
    # print(Erros)
    # print(Erros_em_ordem)

    # Parents = [Populacao[Erros.index(Erros_em_ordem[indice])] for indice in range(Nparents)]
    for indice in range(Nparents):
        # print(Erros[Erros.index(Erros_em_ordem[indice])], Erros_em_ordem[indice], Populacao[Erros.index(Erros_em_ordem[indice])])
        Parents.append(Populacao[Erros.index(Erros_em_ordem[indice])])

    # Criando proxima geracao

    Populacao = Crossover(Parents)

    # Pegando dados relacionados aos erros

    Erros_min.append(min(Erros))
    Erro_max_parents.append(Erros_em_ordem[Nparents-1])
    Erros_mean.append(np.mean(Erros))
    Erros_mean_parents.append(np.mean(Erros_em_ordem[:Nparents]))

    # Criterios convergencia
    # if (Erro_max_parents[-1] - Erros_min[-1]) <= tolerancia:
    #     print('Convergencia alcancada')
    #     break
    if passo > Min_Passos:
        if abs(Erros_min[-1]-Erros_min[-2]) <= tol_diff_erro_min:
            print('Convergencia do erro min')
            break

    # Alterando contador
    passo += 1

# os.system('rm tight_binding_params_*')
print('Melhores parametros:', best_indiv_global)
print('No passo ', passo_best_ind_global)

plt.figure(1)
plt.plot(np.arange(len(Erros_min)), Erros_min, 'b--', label='Menor erro')
plt.plot(np.arange(len(Erro_max_parents)), Erro_max_parents, 'g--', label ='Maior erro parents')
plt.plot(np.arange(len(Erros_mean)), Erros_mean, 'k--', label='Erro medio')
plt.plot(np.arange(len(Erros_mean_parents)), Erros_mean_parents, 'm--', label='Erro medio - parents')

plt.legend()


plt.xlabel('Step', fontsize=16)
plt.ylabel('Erro', fontsize=16)
# plt.ylim(min(Erros_min), Erros_mean[0]*1.05)
plt.tight_layout()
plt.savefig(pp, format='pdf')

for i in range(len(faixa_params)):
    plt.figure(i+2)
    plt.ylabel(r'$p_'+str(i+1)+'$', fontsize=16)
    plt.xlabel('Step', fontsize=16)
    plt.plot(np.arange(len(Best_params[i])), Best_params[i], 'b--')
    plt.tight_layout()
    plt.savefig(pp, format='pdf')


arq_log.write('Melhor passo '+str(passo_best_ind_global)+'\n')
arq_log.write('Erro ='+str(Erro_best_ind_global)+'\n')
arq_log.write('Parametros: ')
for param in best_indiv_global:
    arq_log.write(str(param)+'  ')
arq_log.write('\n')


pp.close()
arq_log.close()


plt.show()
