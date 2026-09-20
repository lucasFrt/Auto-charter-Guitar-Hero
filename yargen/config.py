"""Todos os parametros ajustaveis do YARGen, em um lugar so.

O documento de projeto e explicito: as heuristicas do mapper vao ser ajustadas
dezenas de vezes, entao nenhuma constante magica pode ficar espalhada pelos
modulos. Tudo que e "numero que talvez eu queira mexer" mora aqui.

Um Config e serializavel para JSON, o que permite:
  - `yargen chart ... --config meu.json` para sobrepor valores;
  - gravar a config exata usada junto da IR, para reproduzir um resultado.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------- tempo/grade

@dataclass
class TempoConfig:
    """Beat tracking e construcao da grade."""

    resolution: int = 480
    """Ticks por semínima no notes.mid. 480 e o valor de facto da cena."""

    hop_length: int = 512
    """Hop do beat tracker, em amostras (a 22050 Hz ~= 23ms)."""

    start_bpm: float = 120.0
    """Palpite inicial do beat tracker. Ajuda em musicas de tempo extremo."""

    tightness: float = 100.0
    """Rigidez do beat tracker do librosa. Maior = penaliza mais desvios da
    periodicidade, melhor para tempo fixo; menor tolera rubato."""

    beats_per_bar: int = 4
    """Numerador da formula de compasso. Denominador e sempre 4 por enquanto."""

    max_tempo_drift: float = 0.5
    """BPM de diferenca minima entre beats consecutivos para gravar um novo
    evento set_tempo. Evita um MIDI com milhares de mudancas de tempo inuteis
    por jitter de deteccao."""

    trim_leading_silence: bool = True
    """Se a grade deve comecar no primeiro beat detectado ou no t=0."""


# ---------------------------------------------------------------------- onset

@dataclass
class OnsetConfig:
    """Deteccao de onsets por stem."""

    hop_length: int = 512
    backtrack: bool = True
    """Recua o onset para o minimo de energia anterior. Deixa o ataque mais
    proximo de onde o ouvido percebe a nota."""

    delta: float = 0.07
    """Limiar do pico. Maior = menos notas, mais conservador."""

    wait_ms: float = 30.0
    """Distancia minima entre dois onsets. Abaixo disso viram um so."""

    pre_max_ms: float = 30.0
    post_max_ms: float = 30.0
    pre_avg_ms: float = 100.0
    post_avg_ms: float = 100.0

    strength_percentile: float = 99.0
    """Percentil da envelope de onset usado para normalizar strength em 0..1.
    Usar percentil em vez do maximo evita que um unico pico esmague a escala."""

    min_strength: float = 0.04
    """Descarta onsets abaixo disso antes de qualquer outra coisa."""


# ---------------------------------------------------------------------- pitch

@dataclass
class PitchConfig:
    """Extracao do contorno f0."""

    fmin_hz: float = 65.4
    """C2. Abaixo disso e quase sempre baixo ou ruido."""

    fmax_hz: float = 1318.5
    """E6. Acima disso pyin fica caro e instavel."""

    frame_length: int = 2048
    hop_length: int = 512

    method: str = "pyin"
    """"pyin" (bom e lento) ou "yin" (rapido e mais ruidoso)."""

    voiced_threshold: float = 0.5
    """Probabilidade minima de "voiced" para aceitar o f0 de um frame."""

    median_filter_frames: int = 5
    """Suavizacao do contorno em semitons. 0 desliga."""


# --------------------------------------------------------------------- mapper

@dataclass
class MapperConfig:
    """O nucleo: pitch + onset -> trastes.

    Estes sao os numeros que mais vao mudar. Todos tem um comentario dizendo
    o que acontece quando voce mexe, porque daqui a tres semanas voce nao vai
    lembrar.
    """

    grid_division: int = 16
    """Quantizacao do Expert: 16 = semicolcheia. 8 = colcheia, mais frouxo."""

    quantize_max_shift_ms: float = 90.0
    """Se a posicao quantizada ficar mais longe que isso do onset real, a nota
    e mantida fora da grade. Protege contra grade errada estragar tudo."""

    # --- pitch -> traste
    band_window_sec: float = 8.0
    """Janela movel usada para calcular as faixas de percentil. Movel, e nao
    global, porque o registro da musica muda entre estrofe e refrao.

    O valor tem um trade-off medido (veja tests/test_mapper.py):
      - curta demais (2-4s) a janela se auto-centra: numa corrida ascendente
        longa cada nota fica no meio da propria janela e o solo inteiro sai
        amarelo em vez de varrer o braco;
      - longa demais (20s+) ela vira global e para de acompanhar a mudanca de
        registro entre secoes, que e o motivo de ela ser movel.
    8s varre corridas de forma uniforme e ainda acompanha secoes reais, que
    duram 15s ou mais."""

    band_min_notes: int = 8
    """Se a janela tiver menos notas que isso, expande ate conseguir. Evita
    faixas calculadas em cima de 2 notas no comeco da musica."""

    band_percentiles: list[float] = field(
        default_factory=lambda: [20.0, 40.0, 60.0, 80.0])
    """Os 4 cortes que dividem o pitch em 5 trastes. Deslocar para
    (10, 30, 60, 85) por exemplo concentra mais notas nos trastes do meio."""

    # --- limite de salto
    jump_window_ms: float = 150.0
    """Abaixo deste intervalo entre notas, o salto de traste e limitado."""

    max_jump: int = 2
    """Trastes de salto permitidos dentro da jump_window. 1 e muito restritivo
    (vira escala), 3 ja produz trechos intocaveis."""

    # --- ancora
    anchor_enabled: bool = True
    anchor_window_sec: float = 8.0
    """Por quanto tempo um pitch ja visto continua ancorado ao mesmo traste.
    Maior preserva riffs longos; muito maior engessa a musica inteira."""

    anchor_tolerance_semitones: float = 0.5
    """Quao perto dois pitches precisam estar para contarem como "o mesmo"."""

    # --- acordes
    chord_window_ms: float = 30.0
    """Onsets dentro desta janela viram nota simultanea."""

    chord_max_frets: int = 2
    """Numero maximo de trastes por acorde."""

    chord_adjacent_only: bool = True
    """Forca os trastes do acorde a serem adjacentes (GR, RY, YB, BO)."""

    chord_min_strength: float = 0.35
    """Onsets fracos nao viram acorde, viram nota unica. Sem isso, qualquer
    vazamento de outro instrumento gera acorde."""

    # --- sustain
    sustain_min_beats: float = 1.5
    """Intervalo minimo, em beats, para a nota virar sustentada."""

    sustain_ratio: float = 0.9
    """Fracao do intervalo que a nota ocupa quando sustenta."""

    sustain_max_beats: float = 16.0
    """Teto de sustain. Evita uma nota de 40 segundos no fim da musica."""

    # --- HOPO
    hopo_enabled: bool = True
    hopo_threshold_ticks: int | None = None
    """None = usa o padrao canonico (resolution / 3) + 1 ticks.
    Definir um numero aqui sobrepoe, e o writer grava hopo_frequency no
    song.ini para o jogo concordar com a gente."""

    # --- densidade
    max_notes_per_second: float = 8.0
    density_window_sec: float = 1.0
    """Janela deslizante em que o teto de notas/s e medido."""

    # --- star power
    star_power_enabled: bool = True
    sp_phrase_gap_sec: float = 1.0
    """Silencio que separa duas frases candidatas a Star Power."""

    sp_min_notes: int = 8
    """Frase precisa de pelo menos tantas notas para virar Star Power."""

    sp_interval_sec: float = 30.0
    """Espacamento alvo entre frases de Star Power."""

    sp_max_phrases: int = 8

    # --- diversos
    seed: int = 0
    """Semente do RNG usado no M1 (trastes aleatorios) e em desempates. Fixo
    para que a saida seja reproduzivel."""


# ----------------------------------------------------------------- dificuldade

@dataclass
class DifficultyTier:
    grid_division: int
    max_notes_per_second: float
    max_frets: int
    allow_chords: bool
    allow_sustains: bool = True


@dataclass
class DifficultyConfig:
    """Reducao Expert -> Hard/Medium/Easy.

    A tabela vem direto do documento de projeto. `beat_bonus` e `bar_bonus`
    controlam o quanto uma posicao metrica forte protege uma nota do corte.
    """

    expert: DifficultyTier = field(default_factory=lambda: DifficultyTier(16, 8.0, 5, True))
    hard: DifficultyTier = field(default_factory=lambda: DifficultyTier(8, 5.0, 5, True))
    medium: DifficultyTier = field(default_factory=lambda: DifficultyTier(8, 3.0, 3, False))
    easy: DifficultyTier = field(default_factory=lambda: DifficultyTier(4, 1.5, 3, False))

    beat_bonus: float = 0.25
    """Somado ao score da nota que cai em cima de um beat."""

    bar_bonus: float = 0.5
    """Somado ao score da nota que cai no tempo forte do compasso."""

    merge_window_ticks: int = 10
    """Notas que colidem apos re-quantizar e ficam a esta distancia sao
    fundidas. 10 ticks e o mesmo valor que GH/RB usam para snap de acorde."""


# --------------------------------------------------------------------- vocals

@dataclass
class VocalsConfig:
    enabled: bool = True

    whisper_model: str = "small"
    """Modelo do faster-whisper. "small" basta; "medium" melhora pouco e
    custa muito mais CPU."""

    whisper_compute_type: str = "int8"
    whisper_language: str | None = None
    """None = autodeteccao."""

    min_note_sec: float = 0.08
    """Palavras mais curtas que isso sao esticadas ate este minimo."""

    max_note_sec: float = 8.0

    phrase_gap_sec: float = 0.6
    """Silencio que fecha uma frase (marcador 105)."""

    phrase_max_sec: float = 12.0
    """Frase e quebrada a forca se passar disso."""

    phrase_pad_sec: float = 0.1
    """Folga que o marcador de frase ganha antes da primeira nota e depois da
    ultima. Sem isso o jogo as vezes corta a primeira silaba."""

    pitch_min_midi: int = 36
    pitch_max_midi: int = 84
    """Faixa canonica do PART VOCALS (C2..C6)."""

    unpitched_fallback_midi: int = 60
    """Altura usada quando nao ha f0 confiavel na palavra. A nota vai marcada
    com '#' (nao-tonal), entao a altura nao e cobrada do jogador."""

    voiced_coverage_min: float = 0.3
    """Fracao minima de frames com f0 na palavra para considerar tonal."""


# -------------------------------------------------------------------- stems

@dataclass
class StemsConfig:
    enabled: bool = True

    model: str = "htdemucs_6s"
    """6 stems (inclui guitar e piano). O htdemucs padrao de 4 stems joga a
    guitarra dentro de "other" junto com teclado, que e exatamente o risco
    apontado no documento. Cai para htdemucs se o modelo 6s nao existir."""

    fallback_model: str = "htdemucs"

    device: str = "cpu"
    shifts: int = 0
    """Media de N deslocamentos aleatorios. Melhora a separacao e multiplica o
    tempo por N. Em CPU, deixe 0."""

    segment: int | None = None
    """Tamanho do segmento em segundos. Menor usa menos RAM."""

    jobs: int = 1

    cache_dir: str = "~/.cache/yargen/stems"
    """Cache por hash do arquivo de entrada. Obrigatorio: sem isso, iterar no
    mapper e insuportavel porque cada rodada re-separa o audio."""

    guitar_source: str = "auto"
    """Qual stem vira PART GUITAR: "auto" usa guitar se o modelo tiver, senao
    other. Aceita tambem "other", "guitar", "mix"."""


# ---------------------------------------------------------------------- audio

@dataclass
class AudioConfig:
    sample_rate: int = 22050
    """Taxa usada na analise. 22050 e suficiente para onset/tempo e metade do
    custo de 44100."""

    mono: bool = True
    normalize: bool = True
    """Normaliza o pico para 1.0 antes de analisar, para que os limiares de
    onset signifiquem a mesma coisa em musicas com masterizacoes diferentes."""


# ------------------------------------------------------------------- saida

@dataclass
class OutputConfig:
    charter: str = "YARGen"
    icon: str = "yargen"
    offset_ms: int = 0
    """Gravado em `delay` no song.ini. Positivo atrasa as notas."""

    preview_start_frac: float = 0.35
    """Onde o preview comeca, como fracao da duracao. Usado quando nao ha uma
    escolha melhor."""

    audio_stem_name: str = "song"
    """Nome base do audio na pasta de saida. O formato reserva nomes
    especificos (song/guitar/bass/drums/vocals); nome arbitrario nao carrega."""

    write_beat_track: bool = True
    """Escreve a trilha BEAT (notas 12/13/14)."""

    write_events_track: bool = True
    """Escreve a trilha EVENTS com [end] e secoes de pratica."""

    copy_audio: bool = True
    """Copia/converte o audio para a pasta de saida."""

    open_note_style: str = "sysex"
    """Como marcar nota aberta: "sysex" (frase Phase Shift 0x01, recomendada
    pela doc por compatibilidade) ou "note" (nota base-1 + [ENHANCED_OPENS],
    mais novo e menos suportado)."""

    zero_sustain_divisor: int = 16
    """Notas sem sustain recebem resolution/este_valor ticks de comprimento,
    so para existir um note_off. 480/16 = 30 ticks, bem abaixo do corte de
    sustain do jogo (resolution/3 = 160), entao o jogo trata como nota seca."""


# -------------------------------------------------------------------- raiz

@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    tempo: TempoConfig = field(default_factory=TempoConfig)
    onset: OnsetConfig = field(default_factory=OnsetConfig)
    pitch: PitchConfig = field(default_factory=PitchConfig)
    mapper: MapperConfig = field(default_factory=MapperConfig)
    difficulty: DifficultyConfig = field(default_factory=DifficultyConfig)
    vocals: VocalsConfig = field(default_factory=VocalsConfig)
    stems: StemsConfig = field(default_factory=StemsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # -- serializacao

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        return _build(cls, data)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def merged(self, overrides: dict[str, Any]) -> "Config":
        """Devolve uma copia com `overrides` aplicado por cima (merge fundo)."""
        return Config.from_dict(_deep_merge(self.to_dict(), overrides))

    def override_path(self, dotted: str, value: Any) -> "Config":
        """`cfg.override_path("mapper.max_jump", 3)` -> nova Config.

        Usado pelo `--set` da CLI, para experimentar um parametro sem escrever
        um arquivo de config.
        """
        keys = dotted.split(".")
        patch: dict[str, Any] = {}
        cursor = patch
        for k in keys[:-1]:
            cursor[k] = {}
            cursor = cursor[k]
        cursor[keys[-1]] = value
        return self.merged(patch)


def _build(tp: type, data: Any) -> Any:
    """Reconstroi dataclasses aninhadas a partir de dicts, respeitando tipos."""
    if not is_dataclass(tp) or not isinstance(data, dict):
        return data
    kwargs: dict[str, Any] = {}
    known = {f.name: f for f in fields(tp)}
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"parametro desconhecido na config: {key!r}")
        f = known[key]
        if is_dataclass(f.type) if isinstance(f.type, type) else False:
            kwargs[key] = _build(f.type, value)
        else:
            kwargs[key] = value
    obj = tp(**{k: v for k, v in kwargs.items()})
    # dataclasses aninhadas chegam como dict porque a anotacao vem como string
    for name, f in known.items():
        current = getattr(obj, name)
        if isinstance(current, dict):
            nested = _NESTED.get((tp, name))
            if nested is not None:
                setattr(obj, name, _build(nested, current))
    return obj


_NESTED: dict[tuple[type, str], type] = {
    (Config, "audio"): AudioConfig,
    (Config, "tempo"): TempoConfig,
    (Config, "onset"): OnsetConfig,
    (Config, "pitch"): PitchConfig,
    (Config, "mapper"): MapperConfig,
    (Config, "difficulty"): DifficultyConfig,
    (Config, "vocals"): VocalsConfig,
    (Config, "stems"): StemsConfig,
    (Config, "output"): OutputConfig,
    (DifficultyConfig, "expert"): DifficultyTier,
    (DifficultyConfig, "hard"): DifficultyTier,
    (DifficultyConfig, "medium"): DifficultyTier,
    (DifficultyConfig, "easy"): DifficultyTier,
}


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


DEFAULT = Config()
