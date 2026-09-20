"""Teto de densidade compartilhado entre o mapper e a reducao."""

import pytest

from yargen.chart.density import effective_window, select


def test_window_widens_so_a_fractional_ceiling_is_representable():
    """1.5 notas/s numa janela de 1s vira limite inteiro 1, ou seja 1.0 nota/s.
    A janela precisa alargar ate o teto ser representavel."""
    window, limit = effective_window(1.5, 1.0)
    assert limit / window == pytest.approx(1.5, rel=0.2)
    assert limit >= 4


def test_generous_ceiling_keeps_the_configured_window():
    window, limit = effective_window(8.0, 1.0)
    assert window == 1.0 and limit == 8


def max_in_window(times, window):
    """Maior numero de notas em qualquer janela deslizante de `window`.

    O maximo sempre ocorre numa janela que comeca em cima de uma nota, entao
    basta olhar essas.
    """
    import bisect
    return max((bisect.bisect_left(times, t + window) - i)
               for i, t in enumerate(times)) if times else 0


def test_selection_respects_the_ceiling():
    """A garantia e sobre a janela deslizante, nao sobre a media: com scores
    empatados o guloso enche o inicio de cada janela e deixa um vao depois,
    o que e permitido (e mais musical do que espacar tudo por igual)."""
    times = [i * 0.05 for i in range(200)]          # 20 notas/s na entrada
    kept = [times[i] for i in select([(t, 0.5) for t in times], 4.0, 1.0)]
    window, limit = effective_window(4.0, 1.0)
    assert max_in_window(kept, window) <= limit


def test_selection_prefers_high_scores():
    items = [(i * 0.1, 1.0 if i % 5 == 0 else 0.1) for i in range(100)]
    kept = set(select(items, 2.0, 1.0))
    strong = {i for i in range(100) if i % 5 == 0}
    assert len(kept & strong) / len(strong) > 0.8


def test_selection_is_deterministic():
    items = [(i * 0.1, (i * 37 % 11) / 10) for i in range(100)]
    assert select(items, 3.0, 1.0) == select(items, 3.0, 1.0)


def test_indices_come_back_sorted():
    items = [(i * 0.1, (i * 37 % 11) / 10) for i in range(50)]
    kept = select(items, 3.0, 1.0)
    assert kept == sorted(kept)


def test_zero_ceiling_keeps_everything():
    items = [(i * 0.1, 0.5) for i in range(10)]
    assert select(items, 0.0, 1.0) == list(range(10))


def test_empty_input():
    assert select([], 5.0, 1.0) == []
