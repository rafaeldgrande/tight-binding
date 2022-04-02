
# Importantdo modulos

from __future__ import division
import numpy as np
from numpy import linalg as LA
from scipy.linalg import eigh
import time
import sys
import multiprocessing as mp
from joblib import Parallel, delayed

def ler_string_true(texto):
    if texto.lower() in ['true', 'yes']:
        return(True)
    else:
        return(False)

# Lendo arquivo com informacoes

PARAMETROS_INPUT = {'PREFIX': 'cell_ntc_10_0',
                    'UNITS': 'ANGSTROM',  # ANGSTROM ou BOHR
                    'PARAMS_TB': 'tight_bind_params_ajustados/tight_binding_params_ntc_10_0_3viz_c_curv_ref_QP_pesomaior_Gamma',
                    'NKpoints': 20,
                    'CAMINHO': [ [0,0,0], [0,0,1/2] ],  # caminho padrao na 1ZB Gamma -> M em ntcs
                    'NPROCS': 1,
                    'VERBOSITY': True,
                    'PLOT': True,
                    'VAL_BANDS_PLOT': 6,
                    'COND_BANDS_PLOT': 6,
                    'BANDA_VAL_INDEX': 0,
                    'OPTICAL_ABSORPTION': False,
                    'EMIN_VAL_BAND': 5.0,
                    'EMIN_COND_BAND': 5.0,
                    'BROADENING_ABS': 1e-3,
                    'EVALUATE_DOS': False,
                    'EVALUATE_LDOS': False,
                    'CURVATURE': False,
                    'WRITE_EIGVECS': False}


# # Base ntc
# Gamma = [0,0,0]
# M = [0,0,1/2]
# M2 = [0,0,-1/2]
# Caminho = [Gamma, M]

print('Reading input_TB\n')
arq_in = open('input_TB')

for line in arq_in:
    linha = line.split()
    if len(linha) >= 3:
        if linha[0] in PARAMETROS_INPUT:
            entrada = ''
            for item in linha[2:]:
                if item == '#':
                    break
                entrada += item
            PARAMETROS_INPUT[linha[0]] = entrada
        else:
            print(linha[0]+' is not a valid command.')

PREFIX = PARAMETROS_INPUT['PREFIX']
UNITS = PARAMETROS_INPUT['UNITS']
arquivo_params_TB = PARAMETROS_INPUT['PARAMS_TB']
NKpoints = int(PARAMETROS_INPUT['NKpoints'])

# pegando o Caminho de pontos k

Caminho = []
PtsK = PARAMETROS_INPUT['CAMINHO'].split(';')  # formato -> 0, 0, 0; 0, 0, 1/2
for ptK in PtsK:
    temp = ptK.split(',')
    kx, ky, kz = float(temp[0]), float(temp[1]), float(temp[2])
    Caminho.append([kx, ky, kz])


num_proc = int(PARAMETROS_INPUT['NPROCS']) #mp.cpu_count()

FLAG_PLOT = ler_string_true(PARAMETROS_INPUT['PLOT'])
if FLAG_PLOT is True:
    import matplotlib.pyplot as plt
    # plt.rc('text', usetex=True)

VERBOSITY = ler_string_true(PARAMETROS_INPUT['VERBOSITY'])

if VERBOSITY is True:
    print('Verbosity ON \n\n')
else:
    print('Verbosity OFF \n\n')

if VERBOSITY is True:
    print('Running with '+str(num_proc)+' processors')

EVALUATE_OPTICAL_ABSORPTION = ler_string_true(PARAMETROS_INPUT['OPTICAL_ABSORPTION'])  # Calcular absorcao optica na direcao z
EVALUATE_DOS = ler_string_true(PARAMETROS_INPUT['EVALUATE_DOS']) # Calcular densidade de estados
EVALUATE_LDOS = ler_string_true(PARAMETROS_INPUT['EVALUATE_LDOS']) # Calcular densidade de estados local
CURVATURA = ler_string_true(PARAMETROS_INPUT['CURVATURE'])  # Incluir curvatura - para nanotubos
ESCREVER_VECS = ler_string_true(PARAMETROS_INPUT['WRITE_EIGVECS']) # Escrever autovetores


val_band = int(PARAMETROS_INPUT['BANDA_VAL_INDEX'])-1

if VERBOSITY is True:
    print('Banda de val: '+ str(val_band + 1)+'\n')

val_bands_plot = int(PARAMETROS_INPUT['VAL_BANDS_PLOT'])  # colocar val_bands_plot = 0 para plotar tudo!
cond_bands_plot = int(PARAMETROS_INPUT['COND_BANDS_PLOT'])


####### PARAMETROS #########################

NimagesX, NimagesY, NimagesZ = 0, 0, 1

#NKpoints =  40  # Pontos K no caminho.
# NKpoints = 500  # Pontos K no caminho.

if UNITS == 'ANGSTROM':
    bohr2A = 1 
elif UNITS == 'BOHR':
    bohr2A = 0.529177  

OrdemMax = 3 # Calculo ate 3s vizinhos

d_tol_intra = 4.2     # Distancia maxima entre vizinhos na mesma molecula
d_tol_inter = 1.1    # Distancia maxima entre vizinhos em moleculas diferentes


# arquivo_params_TB = sys.argv[1]  # Pra rodar o codigo no genetic_algorithm.py!
#arquivo_params_TB = 'tight_binding_params'
#arquivo_params_TB = 'tight_bind_params_ajustados/tight_binding_params_ntc_10_0_3viz_c_curv_ref_QP_pesomaior_Gamma'
#arquivo_params_TB = 'tight_binding_params_ref'


Arq_posicoes = open(PREFIX+'.xyz')
dispersao_file = open('Dispersao_'+PREFIX+'.out', 'w')
# dispersao_file = open('TB_ntcs/Dispersao_ntc_8_0_com_curv_ref_QP.out', 'w') # Pra rodar no codigo de algoritmo genetico!
# dispersao_file = open('TB_ntcs/Dispersao_ntc_8_0_com_curv_ref_DFT_fit_Gamma.out', 'w')


def banda_val(Tot_atoms):
    return int(Tot_atoms/2) - 1   # para ntc pure


#### OPTICAL ABSORPTION ####################

broadening = float(PARAMETROS_INPUT['BROADENING_ABS'])

dE_abs = 0.01
E_photon = np.arange(0, 4, dE_abs)  # Emin, Emax, dE
Emin_val_abs = float(PARAMETROS_INPUT['EMIN_VAL_BAND'])
Emax_cond_abs = float(PARAMETROS_INPUT['EMIN_COND_BAND'])

# Range de energia no calculo. Soh inclui bandas de val (cond)
# que tenham energia -Emin_val (Emax_cond) acima da banda de val.

Sigma_zz = np.zeros((len(E_photon)), dtype=complex)

Epsilon2 = np.zeros((len(E_photon)), dtype=complex)

def FD_dist(E, E_fermi):
    if E > E_fermi:
        return 0
    else:
        return 1

def delta_func(broadening, t):
    return broadening/(np.pi*(t**2 + broadening**2))

######### DENSITY OF STATES ###############

Emin_dos, Emax_dos = -7.5, 12
dE_dos = 0.08

E_DOS = np.arange(Emin_dos, Emax_dos, dE_dos)  # Emin, Emax, dE
DOS = np.zeros(len(E_DOS))

################## LOCAL DENSITY OF STATES #######################

Posicoes = [np.array([8.142315006,   0.000000000,   0.022827370])]
LDOS = []
E_LDOS = []

########## EFEITOS CURVATURA - NANOTUBOS ####################

if CURVATURA is True:
    # Vou calcular de maneira preguicosa assumindo q tem simetria cilindrica

    def fita_circle(Atoms):   # Cada item da lista Atoms eh: [ind_atom, ind_mol, array(x,y,z)]

        X, Y = [], []   # Posicoes X e Y dos atomos
        Radial_Distances = []

        for atom in Atoms:
            if int(atom[2]) == 6:
                X.append(atom[3][0])
                Y.append(atom[3][1])

        x0 = np.mean(X)
        y0 = np.mean(Y)

        for ind_atom in range(len(X)):
            Radial_Distances.append( np.sqrt( (X[ind_atom]-x0)**2 + (Y[ind_atom]-y0)**2) )

        R = np.mean(Radial_Distances)

        # Plot pra ver se deu certo

        if FLAG_PLOT is True:

            plt.figure(2)
            for ind_atom in range(len(X)):
                plt.plot(X[ind_atom], Y[ind_atom], 'ro')

            xplot = np.arange(min(X), max(X), 0.01)
            yplot = np.sqrt(R**2 - (xplot - x0)**2) + y0
            yplot_minus = -np.sqrt(R**2 - (xplot - x0)**2) + y0
            plt.plot(xplot, yplot, 'b--')
            plt.plot(xplot, yplot_minus, 'b--')

            plt.savefig('ajuste_circ.png')

        return x0, y0, R

    def angulo_polar(x0, y0, R, atom):
        x, y = atom[3][0], atom[3][1]
        R = np.sqrt((x-x0)**2 + (y-y0)**2)
        # print(x-x0, R)
        theta = np.arccos((x-x0)/R)
        if (y-y0) < 0:
            theta = 2*np.pi - theta
        return(theta)

    def curvatura(n1, n2, r21):
        r21_uni = r21/np.sqrt(np.dot(r21, r21))
        n1_perp = n1 - np.dot(n1, r21_uni)*r21_uni
        n2_perp = n2 - np.dot(n2, r21_uni)*r21_uni
        return np.dot(n1_perp, n2_perp)


########## TIGHT BINDING #####################################

Time0 = time.time()  # Contagem de tempo
start_time = time.time()

Atoms = []            # Lista com atomos
Atomic_Numbers = []   # Lista com numeros atomicos usados
                      #-> eh pra indicar que tipos de atomos estao inclusos no Calculo
                      # numeros floats sao aceitos para possibilitar parametros diferentes
                      # para mesmo tipos de atomos. Ex: calculo pode ter carbonos sp2 e sp3

Lista_VizIntra = []  # Lista de Vizinhos dentro da mesma molecula
Lista_VizInter = []  # Lista de Vizinhos em moleculas diferentes

E_q = []  # Dispersao

#################################################################################
# Definindo caminho na 1ZB pro calculo de dispersao

# Base retangular -> b1 = 2pi/a1(1,0,0) e b2 = 2pi/a2(0,0,1)
#Gamma = [0,0,0]
#X1 = [0,1/2,0]
#X2 = [1/2,1/2,0]
#Caminho = [X2, Gamma, X1]

# Base grafeno -> a1 =
#Gamma = [0,0,0]
#M = [1/2, 0, 0]
#K = [1/3.0, -1/3.0, 0]
#K2 = [2/3, 1/3, 0]
#Caminho = [Gamma, M, K2, Gamma]
#Caminho = [M, K, Gamma]

# # Base ntc
# Gamma = [0,0,0]
# M = [0,0,1/2]
# M2 = [0,0,-1/2]
# Caminho = [Gamma, M]

#################################################################################
# Lendo parametros tight binding
# Vou assumir que valores estao organizados em ordem crescente de numero atomico

E0, t_intra, t_inter, Overlap = [], [], [], []

Tipos_atomicos = []

arq_parameters = open(arquivo_params_TB)


for line in arq_parameters:
    linha = line.split()
    if len(linha) > 0:
        if linha[0] == 'A': # eh uma linha so
            for item in linha:
                if item != 'A':
                    Tipos_atomicos.append(float(item))

Tipos_atomicos.sort()

for a in range(len(Tipos_atomicos)):
    t_inter.append([])
    t_intra.append([])
    Overlap.append([])
    for b in range(len(Tipos_atomicos)):
        t_inter[-1].append([])
        t_intra[-1].append([])
        Overlap[-1].append(0)
        for ordem in range(OrdemMax):
            t_inter[-1][-1].append(0)
            t_intra[-1][-1].append(0)

arq_parameters = open(arquivo_params_TB)

for line in arq_parameters:
    linha = line.split()
    if len(linha) > 0:
        if linha[0] == 'A':
            for item in linha[1:]:
                Atomic_Numbers.append(float(item))
        if linha[0] == 'E0':
            E0.append(float(linha[2]))
        if linha[0] == 'S':
            ind1, ind2 = Tipos_atomicos.index(float(linha[1])), Tipos_atomicos.index(float(linha[2]))
            Overlap[ind1][ind2] = float(linha[3])
        if linha[0] == 't0':
            ind1, ind2 = Tipos_atomicos.index(float(linha[1])), Tipos_atomicos.index(float(linha[2]))
            for i in range(min(OrdemMax, len(linha)-3)):
                t_intra[ind1][ind2][i] = float(linha[i+3])
                t_intra[ind2][ind1][i] = float(linha[i+3])
        if linha[0] == 't1':
            ind1, ind2 = Tipos_atomicos.index(float(linha[1])), Tipos_atomicos.index(float(linha[2]))
            for i in range(min(OrdemMax, len(linha)-3)):
                t_inter[ind1][ind2][i] = float(linha[i+3])
                t_inter[ind2][ind1][i] = float(linha[i+3])

Atomic_Numbers.sort()
# Parametros botando a mao para testes

#E0 = [0.0, 0.0]  # On site energy teste
#t_intra = [ [ [-2.9, -0.0, -0.0, 0.0], [-2.9, -0.0, -0.0, 0.0] ],
#            [ [-2.9, -0.0, -0.0, 0.0], [-2.9, -0.0, -0.0, 0.0] ] ]
#t_inter = [ [ [-0.3, -0.0, -0.0, 0.0], [-0.3, -0.0, -0.0, 0.0] ],
#            [ [-0.3, -0.0, -0.0, 0.0], [-0.3, -0.0, -0.0, 0.0] ] ]
#Overlap = [ [ 0.0, 0.0 ],
#            [ 0.0, 0.0 ] ]
#Overlap = [ [ 0.13, 0.13 ],
#            [ 0.13, 0.13 ] ]

#################################################################################

# Pegando vetores da rede e posicoes atomicas
# Cara do arquivo com posicoes atomicas
# ind molecula, numero atomico, x, y, z
# 1 6 0.0 0.0 0.0
# 1 6 2.84 0.0 0.0

if VERBOSITY is True:
    print('Lendo arquivo com posicoes atomicas')

ind = 0  # Indice de cada atomo lido

for line in Arq_posicoes:
    linha = line.split()
    if len(linha) == 4:
        if linha[0] != '#':
            if linha[0] == 'a1':
                a1 = bohr2A*np.array([float(linha[1]), float(linha[2]), float(linha[3])])
            elif linha[0] == 'a2':
                a2 = bohr2A*np.array([float(linha[1]), float(linha[2]), float(linha[3])])
            elif linha[0] == 'a3':
                a3 = bohr2A*np.array([float(linha[1]), float(linha[2]), float(linha[3])])
    elif (len(linha)) == 5:
        if linha[0] != '#':
            Atoms.append([ind, int(linha[0]), float(linha[1]),
            bohr2A*np.array([float(linha[2]), float(linha[3]), float(linha[4])])])
            Lista_VizIntra.append([])
            Lista_VizInter.append([])
            ind += 1
            if Atomic_Numbers.count(float(linha[1])) == 0:
               print('Faltou especificar o elemento com o seguinte num atomico:'+linha[1])

elapsed_time = round((time.time() - start_time)/60.0, 2)
if VERBOSITY is True:
    print('Se passaram '+str(elapsed_time)+' minutos')
start_time = time.time()

Tot_atoms = len(Atoms)  # Total de atomos


if CURVATURA is True:
    x0, y0, R = fita_circle(Atoms)

    Angulos_Polares = []
    Vetores_normais = []

    for atom in Atoms:
        Angulos_Polares.append(angulo_polar(x0, y0, R, atom))

    for angulo in Angulos_Polares:
        Vetores_normais.append(np.array([np.cos(angulo), np.sin(angulo), 0]))

#################################################################################

# Vetores da rede reciproca
b1 = 2*np.pi*np.cross(a2, a3)/(np.dot(a1, np.cross(a2, a3)))
b2 = 2*np.pi*np.cross(a3, a1)/(np.dot(a1, np.cross(a2, a3)))
b3 = 2*np.pi*np.cross(a1, a2)/(np.dot(a1, np.cross(a2, a3)))

# Construindo caminho na primeira zona de Brillouin

Q = []  # Coordenada no espaco reciproco para plot

Points = []
for PontoK in Caminho:
    Points.append(PontoK[0]*b1 + PontoK[1]*b2 + PontoK[2]*b3)

LengthKpath = 0.0
for i in range(len(Points)-1):
    LengthKpath += LA.norm(Points[i]-Points[i+1])

KPath = []

EIGVECS, EIGVALS = [[]], [[]]
KPath.append(Points[0])
Q.append(0)

for i in range(len(Points)-1):
    trecho = Points[i+1]-Points[i]
    tam_trecho = LA.norm(trecho)
    Npoints_trecho = int(NKpoints*tam_trecho/LengthKpath)
    for j in range(Npoints_trecho):
        desloc = trecho/Npoints_trecho
        KPath.append(KPath[-1]+desloc)
        Q.append(Q[-1]+LA.norm(desloc))
        EIGVALS.append([])
        EIGVECS.append([])

# Dispersao
for i in range(Tot_atoms):
    E_q.append([])
    for j in range(len(KPath)):
        E_q[-1].append(0)

#################################################################################
# Construindo listas de vizinhos

if VERBOSITY is True:
    print('\nCriando lista de vizinhos')

for atom1 in Atoms:
    v1 = atom1[3]
    ind1 = Atoms.index(atom1)
    for atom2 in Atoms:
        v2 = atom2[3]
        ind2 = Atoms.index(atom2)
        for i in range(-NimagesX, NimagesX+1):   # Olhando imagens periodicas
            for j in range(-NimagesY, NimagesY+1):
                for k in range(-NimagesZ, NimagesZ+1):
                    v21 = (v2 + i*a1 + j*a2 + k*a3) - v1
                    lenght = LA.norm(v21)
                    if atom1[1] == atom2[1]:
                        if 0 < lenght <= d_tol_intra:
                            Lista_VizIntra[ind1].append([ind2, v21, -1, 0.0])
                    else:
                        if 0 < lenght <= d_tol_inter:
                            Lista_VizInter[ind1].append([ind2, v21, -1, 0.0])

# Determinando hoppings e overlaps

if VERBOSITY is True:
    print('\nDeterminando hoppings e overlaps')

for Lista_Viz, Hopping in [(Lista_VizInter, t_inter), (Lista_VizIntra, t_intra)]:
    for ind1 in range(len(Lista_Viz)):

        # Analisando distancias

        Distances = []
        atomic_number1 = Atomic_Numbers.index(Atoms[ind1][2])

        for j in range(len(Lista_Viz[ind1])):
            d_round = round(LA.norm(Lista_Viz[ind1][j][1]), 1)
            if Distances.count(d_round) == 0:
                Distances.append(d_round)

        # Botando lista em ordem
        Distances.sort()

        for vizinho in range(len(Lista_Viz[ind1])):

            ind2 = Lista_Viz[ind1][vizinho][0]
            atomic_number2 = Atomic_Numbers.index(Atoms[ind2][2])

            r21 = Lista_Viz[ind1][vizinho][1]
            ordem_viz = Distances.index(round(LA.norm(r21), 1))  # Determinando se e primeiro, segundo, etc, vizinho
            Lista_Viz[ind1][vizinho][2] = ordem_viz

            if ordem_viz < OrdemMax:
                t0 = Hopping[atomic_number1][atomic_number2][ordem_viz]

                if int(atomic_number1) == int(atomic_number2) == 6:

                    if CURVATURA is True:
                        t0 = t0*curvatura(Vetores_normais[ind1], Vetores_normais[ind2], r21)

                Lista_Viz[ind1][vizinho][3] = t0


elapsed_time = round((time.time() - start_time)/60.0, 2)
if VERBOSITY is True:
    print('Se passaram '+str(elapsed_time)+' minutos')
start_time = time.time()

#################################################################################
# Calculando dispersao

if VERBOSITY is True:
    print('\nCalculando dispersao')

def hamiltoniano(q):

    H = np.zeros((Tot_atoms, Tot_atoms), dtype=complex) # hamiltoniano
    S = np.zeros((Tot_atoms, Tot_atoms), dtype=complex) # overlap

    for i in range(Tot_atoms):  # Termos diagonais

        S[i][i] = 1.0           # Overlap do orbital consigo mesmo
        ind_num_atomico = Atomic_Numbers.index(Atoms[i][2])

        H[i][i] = H[i][i] + E0[ind_num_atomico]  # Energias on-site

    for Lista_Viz in [Lista_VizInter, Lista_VizIntra]:
        for ind1 in range(len(Lista_Viz)):

            atomic_number1 = Atomic_Numbers.index(Atoms[ind1][2])

            for vizinho in Lista_Viz[ind1]:
                ind2 = vizinho[0]  # Indice do segundo atomo
                r21 = vizinho[1]   # Posicao relativa do atomo 2 em relacao ao atomo 1 = r2 - r1
                ordem_viz = vizinho[2]

                atomic_number2 = Atomic_Numbers.index(Atoms[ind2][2])

                fase = np.dot(q, r21)
                gamma = np.cos(fase) + 1.0j*np.sin(fase)  # exp(iqr)

                if ordem_viz <= OrdemMax - 1:
                    try:
                        t0 = vizinho[3]
                        if ordem_viz == 0:  # Overlap entre primeiros vizinhos somente
                            s = Overlap[atomic_number1][atomic_number2]
                        else:
                            s = 0.0
                        H[ind1][ind2] = H[ind1][ind2] + t0*gamma  # hopping
                        S[ind1][ind2] = S[ind1][ind2] + s*gamma
                    except:
                        print('deu ruim - definindo hopping')

    return H, S

def Matriz_distancias(q):

    Rij = []
    for i in range(Tot_atoms):
        Rij.append([])
        for j in range(Tot_atoms):
            Rij[-1].append(np.array([0,0,0]))

    for Lista_Viz in [Lista_VizInter, Lista_VizIntra]:
        for ind1 in range(len(Lista_Viz)):
            for vizinho in Lista_Viz[ind1]:
                ind2 = vizinho[0]  # Indice do segundo atomo
                r21 = vizinho[1]   # Posicao relativa do atomo 2 em relacao ao atomo 1 = r2 - r1
                ordem_viz = vizinho[2]
                if ordem_viz <= OrdemMax - 1:
                    Rij[ind1][ind2] = r21

    return Rij

def tight_bind_calc(q_index):

    q = KPath[q_index]

    H, S = hamiltoniano(q)

    # Calculando autovalores e autovetores

    try:
        eigvals, eigvecs = eigh(H, S, eigvals_only=False)
    except:
        # print('Deu ruim na diagonalizacao - problemas com parametros')
        eigvals, eigvecs = np.zeros((Tot_atoms)), np.zeros((Tot_atoms))
    #eigvals_sorted = sorted(eigvals) # Botando o array em ordem crescente de valores

    return [eigvals, eigvecs]


# metodo 1 parallel
if num_proc > 1:
    pool = mp.Pool(num_proc)
    results = pool.map(tight_bind_calc, [q_index for q_index in range(len(KPath))])
    pool.close()
else:
    results = [tight_bind_calc(q_index) for q_index in range(len(KPath))]

# Definindo 0 de energia como o topo da banda de valencia. Supondo q o caminho comece em gamma
E_zero = results[0][0][val_band]

# Organizando dados
for q_index in range(len(KPath)):
    EIGVALS[q_index], EIGVECS[q_index] = results[q_index][0]-E_zero, results[q_index][1]
    for j in range(len(EIGVALS[q_index])):
        E_q[j][q_index] = EIGVALS[q_index][j]

# Escrevendo autovetores

if ESCREVER_VECS is True:
    print('\nEscrevendo autovetores')
    arq_out_vecs = open(PREFIX+'_eigenvecs.out', 'w')


    for p in range(len(EIGVECS)):
        kx, ky, kz = KPath[p]
        arq_out_vecs.write('k point '+str(kx)+' '+str(ky)+' '+str(kz)+'\n')
        for i in range(len(EIGVECS[p])):
            arq_out_vecs.write('E = '+str(EIGVALS[p][i])+'\n')
            for j in range(len(EIGVECS[p][i])):
                # realpart, imagpart = np.real(EIGVECS[p][i][j]), np.imag(EIGVECS[p][i][j])
                realpart, imagpart = np.real(EIGVECS[p][j][i]), np.imag(EIGVECS[p][j][i])
                arq_out_vecs.write(str(realpart)+'+'+str(imagpart)+'j ')
            arq_out_vecs.write('\n')

    arq_out_vecs.close()

elapsed_time = round((time.time() - start_time)/60.0, 2)

if VERBOSITY is True:
    print('Se passaram '+str(elapsed_time)+' minutos')
start_time = time.time()

############################## ABSORCAO ######################################################

def absorcao_optica(item_lista):

    beta, alfa, q_index = item_lista[0], item_lista[1], item_lista[2]

    q = KPath[q_index]
    H, S = hamiltoniano(q)

    temp_Sigma = np.zeros((len(E_photon)), dtype=complex)
    Diff_energy = EIGVALS[q_index][alfa] - EIGVALS[q_index][beta]

    alfa_V_beta = 0.0  # Elemento de matriz <alfa | V | beta>

    for m in range(Tot_atoms):
        for n in range(Tot_atoms):
            H_mn = H[m][n]
            # DeltaR = (Atoms[m][3] - Atoms[n][3])[2]  # So pegando componente Z
            DeltaR = Rij[m][n][2]
            #alfa_V_beta += 1.0j*H_mn*DeltaR*np.conj(EIGVECS[q_index][alfa][m])*EIGVECS[q_index][beta][n]
            alfa_V_beta += 1.0j*H_mn*DeltaR*np.conj(EIGVECS[q_index][m][alfa])*EIGVECS[q_index][n][beta]

    alfa_V_beta_2 = alfa_V_beta*np.conj(alfa_V_beta)  # modulo ao quadrado de alfa_V_bet

    for l in range(len(E_photon)):
        temp_Sigma[l] = temp_Sigma[l] + 1.0j*alfa_V_beta_2/( Diff_energy*(Diff_energy-E_photon[l]+1.0j*broadening) )

    return [beta, alfa, temp_Sigma]


if EVALUATE_OPTICAL_ABSORPTION is True:

    PartialSigmas = []

    if VERBOSITY is True:
        print('\nCalculating optical absorption')

    Rij = Matriz_distancias(np.array([0,0,0]))

    # for beta in range(Tot_atoms):  # alfa = conducao , beta = valencia
    #    if -Emin_val <= EIGVALS[0][beta] <= 0.1:  # Bandas de valencia
    #       for alfa in range(Tot_atoms):
    #           if 0.1 < EIGVALS[0][alfa] <= Emax_cond: # Bandas de conducao

    temp1, temp2 = tight_bind_calc(0)
    ENERGIAS_GAMMA = temp1 - E_zero

    lista_bandas_val_abs, lista_bandas_cond_abs = [], []

    for p in range(len(ENERGIAS_GAMMA)):
        if 0 < ENERGIAS_GAMMA[p] < Emax_cond_abs:
            lista_bandas_cond_abs.append(p)
        elif -Emin_val_abs < ENERGIAS_GAMMA[p] <= 0:
            lista_bandas_val_abs.append(p)

    if VERBOSITY is True:
        print('Valence bands included in absorption calculation : ', lista_bandas_val_abs)
        print('Total of valence bands '+str(len(lista_bandas_val_abs)))
        print('Conduction bands included in absorption calculation : ', lista_bandas_cond_abs)
        print('Total of conduction bands '+str(len(lista_bandas_cond_abs)))
    
    # if val_bands_abs == 0:
    #     lim_val_bands = 0
    # else:
    #     lim_val_bands = val_band - val_bands_abs + 1

    # if cond_bands_abs == 0:
    #     lim_cond_bands = Tot_atoms
    # else:
    #     lim_cond_bands = val_band + cond_bands_abs + 1

    # for beta in range(val_band-val_bands_abs+1, val_band+1):
    
#    for beta in range(lim_val_bands, val_band+1):
#        for alfa in range(val_band+1, lim_cond_bands):
#            PartialSigmas.append([beta, alfa, np.zeros((len(E_photon)), dtype=complex)])
#            for q_index in range(len(KPath)):
#                List_3var.append([beta, alfa, q_index])

    List_3var = []

    for beta in lista_bandas_val_abs:
        for alfa in lista_bandas_cond_abs:
            PartialSigmas.append([beta, alfa, np.zeros((len(E_photon)), dtype=complex)])
            for q_index in range(len(KPath)):
                List_3var.append([beta, alfa, q_index])


    pool = mp.Pool(num_proc)
    results = pool.map(absorcao_optica, [item_lista for item_lista in List_3var])
    pool.close()

    for result in results:
        Sigma_zz = Sigma_zz + result[2]
        beta, alfa = result[0], result[1]
        for i in range(len(PartialSigmas)):
            if PartialSigmas[i][0] == beta and PartialSigmas[i][1] == alfa:
                PartialSigmas[i][2] = PartialSigmas[i][2] + result[2]

    print('Writing absorption data')

    arq_out_abs_total = open(PREFIX+'_absorption_tot.out', 'w')

    for i in range(len(E_photon)):
        arq_out_abs_total.write(str(E_photon[i])+' '+str(np.real(Sigma_zz[i]))+'\n')

    arq_out_abs_total.close()

#    for i in range(len(PartialSigmas)):
#        arq_out_abs = open(PREFIX+'_partial_absorption_v'+str(val_band - PartialSigmas[i][0] + 1)+'_c'+str(PartialSigmas[i][1] - val_band)+'.out', 'w')
#        for j in range(len(E_photon)):
#            arq_out_abs.write(str(E_photon[j])+' '+str(np.real(PartialSigmas[i][2][j]))+'\n')

#        arq_out_abs.close()

    # for alfa in range(Tot_atoms):
    #     for beta in range(Tot_atoms):
    #         if (-Emin_val <= EIGVALS[0][beta] <= 0.1) and (0.1 < EIGVALS[0][alfa] <= Emax_cond):
    #
    #             for q_index in range(len(KPath)):
    #
    #                 q = KPath[q_index]
    #                 H, S = hamiltoniano(q)
    #
    #                 Diff_energy = EIGVALS[q_index][alfa] - EIGVALS[q_index][beta]
    #                 alfa_V_beta = 0
    #
    #                 for m in range(Tot_atoms):
    #                     for n in range(Tot_atoms):
    #                         H_mn = H[m][n]
    #                         # DeltaR = (Atoms[m][3] - Atoms[n][3])[2]  # So pegando componente Z
    #                         DeltaR = Rij[m][n][2]
    #                         alfa_V_beta += 1.0j*H_mn*DeltaR*np.conj(EIGVECS[q_index][alfa][m])*EIGVECS[q_index][beta][n]
    #
    #                 alfa_V_beta_2 = alfa_V_beta*np.conj(alfa_V_beta)  # modulo ao quadrado de alfa_V_beta
    #
    #                 for l in range(1, len(E_photon)):
    #                     Epsilon2[l] = Epsilon2[l] + min(1, 1/E_photon[l]**2)*alfa_V_beta_2*delta_func(broadening, E_photon[l] - Diff_energy)



elapsed_time = round((time.time() - start_time)/60.0, 2)

if VERBOSITY is True:
    print('Se passaram '+str(elapsed_time)+' minutos')
start_time = time.time()

# Escrevendo arquivos de saida

dispersao_file.write('# Q E_q\n')

for i in range(len(E_q)):
    if val_bands_plot == 0:
        dispersao_file.write('\n')
        for j in range(len(E_q[i])):
            dispersao_file.write(str(Q[j])+'  '+str(E_q[i][j])+'\n')
        if FLAG_PLOT is True:
            plt.figure(1)
            plt.plot(Q, E_q[i], 'k')
            # plt.ylim([-2,2])
    else:
        if val_band - val_bands_plot + 1 <= i <= val_band + cond_bands_plot:
            dispersao_file.write('\n')
            for j in range(len(E_q[i])):
                dispersao_file.write(str(Q[j])+'  '+str(E_q[i][j])+'\n')
            if FLAG_PLOT is True:
                plt.figure(1)
                plt.plot(Q, E_q[i], 'k')
                # plt.ylim([-2,2])

    if -Emin_val_abs <= E_q[i][0] <= Emax_cond_abs:
        if FLAG_PLOT is True:
            plt.figure(1)
            plt.plot(Q, E_q[i], 'r')

if FLAG_PLOT is True:
    plt.figure(1)
    #plt.ylim([-1.2*Emin_val_abs, 1.2*Emax_cond_abs])
    plt.savefig('Dispersao_'+PREFIX+'.png')

dispersao_file.close()

if EVALUATE_OPTICAL_ABSORPTION is True:
    maxEps2 = max(Epsilon2)
    maxSigmazz = max(np.real(Sigma_zz))
    if FLAG_PLOT is True:
        plt.figure()
        plt.plot(E_photon, np.real(Sigma_zz)/maxSigmazz, 'k-')
        plt.savefig('absorption_'+PREFIX+'.png')
        # plt.plot(E_photon, Epsilon2/maxEps2, 'r-')


# Calculando DOS

if EVALUATE_DOS is True:

    if VERBOSITY is True:
        print('\nCalculando DOS')

    for i in range(len(E_q)):
        for j in range(len(E_q[i])):
            if Emin_dos < E_q[i][j] < Emax_dos:
                index = int((E_q[i][j] - Emin_dos)/dE_dos)
                DOS[index] = DOS[index] + 1

    if FLAG_PLOT is True:
        plt.figure()
        plt.plot(E_DOS, DOS)
        plt.savefig('DOS_'+PREFIX+'.png')

    arq_out = open(PREFIX+'_DOS.out', 'w')
    for i in range(len(E_DOS)):
        arq_out.write(str(E_DOS[i])+'  '+str(DOS[i])+'\n')

    arq_out.close()

# if FLAG_PLOT is True:
#    plt.show()

if VERBOSITY is True:
    print('Done')
