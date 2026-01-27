#!/usr/bin/env python
"""\
Esse arquivo contém funções que resolvem computacionalmente algumas situações de SSTC.

As funções mais importantes* são destacadas nos notebooks e podem auxiliar o estudante na resolução de exercícios.

Outras funções que, em geral, compões as funções principais usadas nos notebooks podem ser manipuladas abaixo.

* : resolver_sistema_LCIT(Q, x, conds)
1 - Funções de análise no tempo
2 - Funções de análise de Fourier
3 - Funções sobre transformada de Laplace
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from IPython.display import display

t = sp.symbols('t')
s = sp.symbols('s')


#################
#
#       1
#
#################


def resolver_sistema_LCIT(Q, x, conds, P=[1], T=6, deslocamentos_dirac=[0]):
    # Obtém a resposta total e as funções de interesse nesse processo:
    y_z  = resolve_entrada_nula(Q, conds)
    h, H = resposta_ao_impulso(Q, P)
    y_s  = resolve_estado_nulo(x, h)
    y    = resolve_EDO(Q, x, P, conds)

    # Exibe as funções obtidas:
    funcs = {'y(t)'  : y,   'x(t)'  : x,
             'y_z(t)': y_z, 'y_s(t)': y_s,
             'h(t)'  : h}
    plota_sistema_LCIT(funcs, Q, P, H, t, T, deslocamentos_dirac)


def resolve_entrada_nula(Q, conds):
    y_z = sp.Function('y_z')(t)

    # Monta a equação para ser resolvida:
    EDO = Q[-1]*y_z
    for n in range(len(Q[:-1])):
        EDO = EDO + Q[-n - 2] * y_z.diff(t, n + 1)

    # Prepara dict de condições iniciais:
    conds_iniciais = {}
    for n in range(len(conds)):
        conds_iniciais.update({y_z.diff(t, n).subs(t, 0): conds[n]})

    # Resolve a EDO homogênea sujeita às condições iniciais:
    y_z = sp.dsolve(EDO, y_z, ics=conds_iniciais)
    y_z = y_z.rhs * sp.Heaviside(t)

    return y_z


def resposta_ao_impulso(Q, P):
    # Monta função de transferência e faz a transformada de Laplace inversa para obter a resposta ao impulso:
    Qs = 0
    Ps = 0

    for n in range(len(Q)):
        Qs += Q[-n - 1] * s**(n)
    for n in range(len(P)):
        Ps += P[-n - 1] * s**(n)

    H = Ps / Qs
    h = sp.inverse_laplace_transform(H, s, t)

    return h, H


def resolve_estado_nulo(x, h):
    # Convolução causal entre a entrada e a resposta ao impulso para obter a saída de estado nulo:
    tau = sp.symbols('tau')
    y_s = sp.integrate(h.subs(t, tau) * x.subs(t, t - tau), (tau, 0, t))

    return y_s


def plota_sistema_LCIT(funcs, Q, P, H, t, T, deslocamentos_dirac):
    # Exibe a EDO do sistema, cada função de interesse e troca o delta típico pelo delta simulado:
    y_p = sp.Function('y')(t)
    EDO = Q[-1]*y_p
    for n in range(len(Q[:-1])):
        EDO = EDO + Q[-n - 2] * y_p.diff(t, n + 1)

    x_p = sp.Function('x')(t)
    entrada = P[-1]*x_p
    for n in range(len(P[:-1])):
        entrada = entrada + P[-n - 2] * x_p.diff(t, n + 1)
    
    display(sp.simplify(sp.Eq(EDO, entrada)))

    for func in funcs:
        display(sp.Eq(sp.Symbol(func), funcs[func]))
        funcs[func] = dirac_simulado(t, funcs[func], deslocamentos_dirac)
    display(t >= 0)

    propriedades = [verifica_memoria(funcs['h(t)']),
                    verifica_causalidade(funcs['h(t)']),
                    verifica_BIBO(funcs['h(t)']),
                    verifica_Lyapunov(Q),
                    'Função de Transferência: ' + str(H)]
    
    # Versão Numérica de cada função para o plot:
    y   = sp.lambdify(t, funcs['y(t)'])
    x   = sp.lambdify(t, funcs['x(t)'])
    y_z = sp.lambdify(t, funcs['y_z(t)'])
    y_s = sp.lambdify(t, funcs['y_s(t)']) 
    h   = sp.lambdify(t, funcs['h(t)'])

    t = np.linspace(0, T, 1000)
    fig, axs = plt.subplots(3, 2, figsize=(9, 8))

    axs[0][0].plot(t, y(t),   color='purple')
    axs[0][1].plot(t, x(t),   color='magenta')
    if (isinstance(y_z(t), int) or isinstance(y_z(t), float)):
        axs[1][0].plot(t, [0] * 1000, color='red')
    else:
        axs[1][0].plot(t, y_z(t), color='red')
    axs[1][1].plot(t, y_s(t), color='blue')
    axs[2][0].plot(t, h(t),   color='orange')

    # Caixa separada para as propriedades
    axs[2][1].axis((0, T, 0, T))
    axs[2][1].set_yticks([])
    axs[2][1].set_xticks([])
    axs[2][1].spines[['top', 'right', 'bottom', 'left']].set_visible(False)

    for i in range(len(propriedades)):
        axs[2][1].text(0, T * 0.1 * (9 - i), propriedades[i], ha='left', wrap=True)

    titulos = ["Resposta Total", "Entrada", "Resposta de Entrada Nula", 
               "Resposta de Estado Nulo", "Resposta ao Impulso", "Propriedades"]
    rotulos = ["$y$", "$x$", "$y_z$", "$y_s$", "$h$"]
    titulo = 0

    for i in range (3):
        for j in range(2):
            axs[i][j].set_title(titulos[titulo])
            if (not(i == 2 and j == 1)):
                axs[i][j].grid(True)
                axs[i][j].set_xlabel('$t$')
                axs[i][j].set_ylabel(rotulos[titulo], rotation=1)
            titulo += 1

    plt.tight_layout()
    plt.show()


def verifica_memoria(h):
    if(isinstance(h/sp.DiracDelta(t), sp.Number)):
        return "Possui Memória: Não"
    else:
        return "Possui Memória: Sim"

    
def verifica_causalidade(h):
    if(h.has(sp.Heaviside)):
        return "Causalidade: Causal"
    elif(h.subs(t, -1).simplify()):
        return "Causal: Causal"    # Teste pontual, não 100% confiável
    else:
        return "Causal: Não Causal"


def verifica_BIBO(h):
    BIBO = sp.integrate(sp.Abs(h), (t, -sp.oo, sp.oo))
    if(BIBO == sp.oo):
        return "BIBO Estabilidade: Instável"
    else:
        return "BIBO Estabilidade: Estável"
    

def verifica_Lyapunov(Q):
    # Monta o polinômio característico:
    polinomio_caracteristico = 0
    for n in range(len(Q)):
        polinomio_caracteristico += Q[-n - 1] * t**(n)
    
    # Obtém as raízes:
    raizes = sp.roots(polinomio_caracteristico)

    # Teste de instabilidade:
    for raiz in raizes:
        if (sp.re(raiz) > 0 or (sp.re(raiz) == 0 and sp.im(raiz) != 0 and raizes[raiz] > 1)):
            return "Estabilidade Assintótica: Instável"
    
    # Teste de estabilidade marginal:
    estabilidade_marginal = False
    for raiz in raizes:
        if (sp.re(raiz) == 0 and sp.Abs(sp.im(raiz)) >= 0 and raizes[raiz] == 1):
            estabilidade_marginal = True
    if (estabilidade_marginal):
        return "Estabilidade Assintótica: Marginal"

    # Opção restante: estável
    return "Estabilidade Assintótica: Estável"


def delta_dirac_gaussiano(t, sigma=0.05):
    # Simula o delta de Dirac numericamente:
    return sp.exp(-t**2/(2*sigma**2)) / (sigma*sp.sqrt(2*sp.pi))

def delta_dirac_gaussiano_derivada_n(t, s=0.05):
    # Simula as derivadas do delta de Dirac numericamente:
    return sp.exp(-(t+s)**2/(2*s**2))/(s*sp.sqrt(2*sp.pi))-sp.exp(-(t-s)**2/(2*s**2))/(s*sp.sqrt(2*sp.pi))

def dirac_simulado(t, f, deslocamentos=[0]):
    # Substitui os deltas analíticos e suas derivadas de até segunda ordem pelos deltas numéricos:
    for deslocamento in deslocamentos:
        if(f.has(sp.DiracDelta(t - deslocamento))) :
            f = f.subs(sp.DiracDelta(t - deslocamento), delta_dirac_gaussiano(t - deslocamento))
        if(f.has(sp.diff(sp.DiracDelta(t - deslocamento), t))) : 
            f = f.subs(sp.diff(sp.DiracDelta(t - deslocamento)), delta_dirac_gaussiano_derivada_n(t - deslocamento))
        if(f.has(sp.diff(sp.DiracDelta(t - deslocamento), t, t))) :
            f = f.subs(sp.diff(sp.DiracDelta(t - deslocamento), t, t), delta_dirac_gaussiano_derivada_n(t - deslocamento))
    return f


def resolve_EDO(Q, x, P, conds):
    y = sp.Function('y')(t)
    EDO = Q[-1]*y
    for n in range(len(Q[:-1])):
        EDO = EDO + Q[-n - 2] * y.diff(t, n + 1)

    entrada = P[-1]*x
    for n in range(len(P[:-1])):
        entrada = entrada + P[-n - 2] * x.diff(t, n + 1)

    conds_iniciais = {}
    for n in range(len(conds)):
        conds_iniciais.update({y.diff(t, n).subs(t, 0): conds[n]})

    y = sp.dsolve(sp.Eq(EDO, entrada), y, ics=conds_iniciais)

    return y.rhs


#################
#
#       2
#
#################


def plota_espectros_s(x, N, w_0):
    amplitude = []
    fase = []
    nw_0 = []
    T_0  = 2*np.pi / w_0

    for n in range(-N, N + 1):
        D_n = sp.integrate(x * sp.exp(-sp.I*n*w_0*t), (t, 0, T_0), conds='none') / T_0
        
        # Obtém amplitude da harmônica:
        D_n = complex(sp.N(D_n))
        re = D_n.real
        im = D_n.imag
        modulo = np.sqrt(re**2 + im**2)
        amplitude.append(modulo)
        
        # Obtém fase da harmônica:
        arg = np.angle(D_n)
        if (np.isnan(arg)):
            arg = 0
        fase.append(arg)

        # Legenda do eixo horizontal do plot:
        match n:
            case -1:
                nw_0.append(r"$-\omega_0$")
            case 0:
                nw_0.append("0")
            case 1:
                nw_0.append(r"$\omega_0$")
            case _:
                nw_0.append(str(n) + r"$\omega_0$")

    # Plot:
    fig, axs = plt.subplots(2, 1, figsize=(8, 7))
    axs[0].stem(nw_0, amplitude)
    axs[0].set_ylabel(r"$|D_n$|", rotation=1)
    axs[0].set_title("Espectro de Amplitude")

    axs[1].stem(nw_0, fase)
    axs[1].set_ylabel(r"$∠D_n$", rotation=1)
    axs[1].set_title("Espectro de Fase")
    plt.show()

