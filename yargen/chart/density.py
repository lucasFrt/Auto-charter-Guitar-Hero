"""Teto de densidade: um so lugar.

Tanto o mapper (Expert) quanto a reducao de dificuldade precisam da mesma
operacao - "corte as notas de menor score ate caber no limite de notas por
segundo" - so que sobre tipos diferentes. A logica mora aqui e os dois
chamam.
"""

from __future__ import annotations

import bisect
import math
from typing import Sequence


def effective_window(max_nps: float, window_sec: float,
                     min_slots: int = 4) -> tuple[float, int]:
    """Escolhe a janela e o limite inteiro de notas dentro dela.

    O limite e um numero inteiro de notas, entao uma janela curta nao consegue
    representar um teto fracionario: com 1.5 notas/s em janela de 1s, o limite
    inteiro vira 1 e o Easy sai com 1.0 nota/s em vez de 1.5 - um terco mais
    esparso do que o pedido.

    A saida e alargar a janela ate caber pelo menos `min_slots` notas, o que
    limita o erro de arredondamento a 1/min_slots. O efeito colateral e
    desejavel: sobre uma janela maior, o teto passa a valer na media e permite
    uma rajada curta seguida de descanso, que e mais musical do que espacar
    tudo de forma uniforme.
    """
    if max_nps <= 0:
        return window_sec, 0
    window = max(window_sec, min_slots / max_nps)
    return window, max(1, int(round(max_nps * window)))


def select(items: Sequence[tuple[float, float]], max_nps: float,
           window_sec: float) -> list[int]:
    """Indices a manter, dados [(tempo_em_segundos, score)].

    Guloso por score decrescente: a nota de maior score entra primeiro e so e
    aceita se nao estourar o limite em nenhuma janela que a contenha.

    O maximo de uma janela deslizante sempre acontece em alguma janela que
    comeca exatamente em cima de uma nota, entao basta checar essas - o teste
    e exato, nao uma aproximacao, e custa O(notas na vizinhanca).
    """
    if max_nps <= 0 or not items:
        return list(range(len(items)))
    window, limit = effective_window(max_nps, window_sec)

    order = sorted(range(len(items)), key=lambda i: (-items[i][1], i))
    accepted: list[float] = []
    keep: set[int] = set()
    for i in order:
        t = items[i][0]
        pos = bisect.bisect_left(accepted, t)
        trial = accepted[:pos] + [t] + accepted[pos:]
        lo = bisect.bisect_left(trial, t - window)
        hi = bisect.bisect_right(trial, t + window)
        # Janela semiaberta [t, t+w): bisect_left e nao bisect_right, senao
        # uma nota exatamente em t+w conta nas duas janelas e uma sequencia
        # exatamente na taxa-limite e rejeitada.
        if all(bisect.bisect_left(trial, trial[j] + window) - j <= limit
               for j in range(lo, hi)):
            accepted.insert(pos, t)
            keep.add(i)
    return sorted(keep)
