"""Levantamento dos stems: o que essa musica TEM, afinal?

O projeto nao consegue - e nao deveria tentar - adivinhar o genero sozinho e
gerar o chart certo a partir de um MP3 e mais nada. A escolha de qual fonte
conduz a trilha de 5 trastes e uma decisao musical, e ela e de quem vai tocar.

O que da para fazer e tornar essa escolha barata e informada: medir cada stem,
mostrar o que foi encontrado, e RECOMENDAR um plano de roteamento que a pessoa
aceita ou troca. "Sua musica tem 3% de guitarra e a bateria domina; quer
chartear a batida?" e uma pergunta muito melhor do que "qual o genero?".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import profiles
from ..audio.loader import Audio
from ..config import DEFAULT, Config
from . import onsets as onsets_mod
from . import pitch as pitch_mod
from . import timbre as timbre_mod


@dataclass
class StemSurvey:
    name: str
    energy_share: float
    """Fracao da energia total da musica (RMS ao quadrado)."""

    onset_rate: float
    """Onsets por segundo. Alto e denso: chimbal, palhetada rapida."""

    pitched_fraction: float
    """Fracao de frames com f0 confiavel. Separa melodico de percussivo: e o
    sinal que diz se faz sentido mapear por altura ou por brilho."""

    centroid_hz: float
    onset_count: int
    activity: float = 0.0
    """Fracao do tempo em que o stem esta soando, medida contra o nivel da
    musica inteira (nao contra o proprio pico).

    E este numero, e nao a energia, que responde "esse instrumento existe
    nesta musica". Energia responde outra pergunta - "quanto ele pesa no
    volume" - e para graves ela mente muito: um 808 leva a maior parte da
    energia bruta e quase nenhuma da energia perceptual, dependendo de como
    se mede. Atividade nao tem esse problema."""

    @property
    def present(self) -> bool:
        """Se vale a pena considerar este stem como fonte.

        O Demucs sempre devolve os seis stems: quando a musica nao tem
        guitarra, o stem de guitarra vem com vazamento e silencio, nao vazio.
        """
        return self.activity >= 0.10 and self.onset_count >= 8

    @property
    def melodic(self) -> bool:
        return self.pitched_fraction >= 0.35


@dataclass
class Survey:
    stems: dict[str, StemSurvey] = field(default_factory=dict)
    duration: float = 0.0
    recommended: str = "rock"
    reasons: list[str] = field(default_factory=list)

    @property
    def profile(self) -> profiles.GenreProfile:
        return profiles.get(self.recommended)

    def present(self) -> list[StemSurvey]:
        return sorted((s for s in self.stems.values() if s.present),
                      key=lambda s: -s.energy_share)


def active_mask(audio: Audio, hop: int, floor_db: float = -40.0,
                reference_peak: float = 0.0) -> np.ndarray:
    """Frames em que o stem realmente esta tocando.

    Medir sobre o arquivo inteiro mistura silencio com musica e mente: um
    stem vocal com dez frases em tres minutos aparece como "23% tonal" quando
    na verdade e tonal em quase todo frame em que existe. O silencio nao e
    informacao sobre o instrumento, e informacao sobre o arranjo.
    """
    import librosa

    rms = librosa.feature.rms(y=audio.samples, hop_length=hop)[0]
    if not rms.size:
        return np.zeros(0, dtype=bool)
    # Referencia externa quando ha uma: contra o proprio pico, um stem que so
    # tem vazamento inaudivel parece 100% ativo, porque o vazamento e o pico
    # dele. E justamente o caso que precisamos distinguir - "esta musica tem
    # guitarra?" - entao a referencia tem que ser o nivel da musica inteira.
    ref = reference_peak if reference_peak > 0 else float(np.max(rms))
    if ref <= 0:
        return np.zeros(len(rms), dtype=bool)
    return (20 * np.log10(np.maximum(rms, 1e-12) / ref)) > floor_db


def rms_peak(audio: Audio, hop: int = 512) -> float:
    """Maior RMS de frame do stem. Usado como referencia comum entre stems."""
    import librosa

    if audio.samples.size < hop:
        return 0.0
    return float(np.max(librosa.feature.rms(y=audio.samples, hop_length=hop)[0]))


def weighted_energy(audio: Audio) -> float:
    """Energia com ponderacao A (perceptual).

    Energia bruta e dominada por graves de um jeito que nao corresponde nem a
    audicao nem a importancia estrutural: num beat de trap, um 808 de 50 Hz
    leva 80% da energia e a bateria inteira - que e o que se quer TOCAR -
    aparece com 5%. A recomendacao ficava dizendo "rock" para trap obvio.

    A ponderacao A e a curva padrao de como o ouvido pesa cada frequencia.
    Com ela o 808 volta para um tamanho honesto e a bateria aparece.
    """
    import librosa

    y, sr = audio.samples, audio.sample_rate
    if len(y) < 2048:
        return float(np.sum(y.astype(np.float64) ** 2))
    spec = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    # O bin DC (0 Hz) nao tem ponderacao definida - log10(0) - e nao carrega
    # informacao musical. Fora dele.
    gain = np.zeros_like(freqs)
    gain[1:] = 10.0 ** (librosa.A_weighting(freqs[1:]) / 20.0)
    return float(np.sum((spec * gain[:, None]) ** 2))


# Instrumentos cuja faixa util nao cabe no padrao do pyin. Sem isto, um 808
# em 41 Hz fica abaixo do fmin padrao (65.4 Hz = C2) e o baixo e classificado
# como nao-melodico - errado, e justamente ao contrario do que ele e.
_PITCH_RANGE = {"bass": (28.0, 400.0), "vocals": (70.0, 1200.0)}


def measure(name: str, audio: Audio, total_energy: float,
            cfg: Config = DEFAULT, *, reference_peak: float = 0.0) -> StemSurvey:
    share = weighted_energy(audio) / total_energy if total_energy > 0 else 0.0

    notes = onsets_mod.detect(audio, cfg.onset)
    rate = len(notes) / audio.duration if audio.duration > 0 else 0.0

    # Amostra ate 30s: pyin e o passo caro e as proporcoes sao estaveis.
    sample = audio.slice(0.0, min(30.0, audio.duration))
    pitch_cfg = cfg.pitch
    if name in _PITCH_RANGE:
        fmin, fmax = _PITCH_RANGE[name]
        pitch_cfg = Config.from_dict(cfg.to_dict()).pitch
        pitch_cfg.fmin_hz, pitch_cfg.fmax_hz = fmin, fmax

    contour = pitch_mod.contour(sample, pitch_cfg)
    tim = timbre_mod.contour(sample, pitch_cfg)
    active = active_mask(sample, pitch_cfg.hop_length,
                         reference_peak=reference_peak)
    activity = float(np.mean(active)) if active.size else 0.0

    def over_active(values: np.ndarray, reducer) -> float:
        if not values.size:
            return 0.0
        mask = active[:len(values)] if active.size else np.ones(len(values), bool)
        chosen = values[:len(mask)][mask]
        return float(reducer(chosen)) if chosen.size else 0.0

    pitched = over_active(contour.f0, lambda v: np.mean(np.isfinite(v)))
    centroid = over_active(tim.centroid_hz,
                           lambda v: np.median(v[np.isfinite(v) & (v > 0)])
                           if np.any(np.isfinite(v) & (v > 0)) else 0.0)

    return StemSurvey(name, share, rate, pitched, centroid, len(notes), activity)


def recommend(survey: Survey) -> Survey:
    """Escolhe um perfil de genero a partir do que os stems mostram.

    A decisao e sobre UMA coisa: quem conduz a trilha de 5 trastes. Baixo e
    voz tem trilha propria e nao disputam esse lugar, entao a pergunta se
    reduz a "ha um lead melodico (guitarra ou teclado)? se nao ha, ha uma
    batida?".

    Decidimos por ATIVIDADE e contagem de onsets, nao por energia. Energia
    depende de como se pondera a frequencia e da resultados opostos conforme
    a escolha - com energia bruta o 808 de um beat de trap leva 78% e parece
    que a musica e o baixo; com ponderacao A ele cai para 4% e some. Nenhum
    dos dois responde "esta musica tem guitarra?". Atividade responde.

    As heuristicas sao deliberadamente simples e explicaveis. O valor nao esta
    em acertar sempre - nao vai - esta em dar um ponto de partida defensavel e
    DIZER POR QUE, para a pessoa discordar com conhecimento de causa. Um
    classificador de genero opaco seria pior aqui mesmo se fosse mais preciso.
    """
    s = survey.stems
    guitar, other, drums, vocals, bass = (s.get("guitar"), s.get("other"),
                                          s.get("drums"), s.get("vocals"),
                                          s.get("bass"))
    reasons: list[str] = []

    def lead(st: StemSurvey | None) -> bool:
        return st is not None and st.present and st.melodic

    guitar_lead, other_lead = lead(guitar), lead(other)
    beat = drums is not None and drums.present and drums.onset_rate >= 1.5
    voice = vocals is not None and vocals.present

    if guitar_lead and (not other_lead or guitar.activity >= other.activity):
        dense = guitar.onset_rate >= 7.0 and beat and drums.onset_rate >= 5.0
        choice = "metal" if dense else ("acoustic" if not beat else "rock")
        reasons.append(f"guitarra melodica e ativa ({guitar.activity:.0%} do tempo, "
                       f"{guitar.onset_rate:.1f} onsets/s)")
        if choice == "metal":
            reasons.append("densidade de guitarra e bateria alta")
        elif choice == "acoustic":
            reasons.append("bateria fraca ou ausente")
    elif other_lead:
        choice = "pop" if voice else "electronic"
        reasons.append(f"o lead melodico vem de teclado/sintetizador "
                       f"({other.activity:.0%} do tempo)")
        if guitar is None or not guitar.present:
            reasons.append("guitarra ausente ou so vazamento")
    elif beat:
        choice = "trap" if voice else "percussion"
        reasons.append(f"nenhum lead melodico; a bateria conduz "
                       f"({drums.activity:.0%} do tempo, {drums.onset_rate:.1f} onsets/s)")
        if guitar is not None and not guitar.present:
            reasons.append(f"guitarra ausente ({guitar.activity:.0%} do tempo)")
        reasons.append("a trilha de 5 trastes vai charteiar a BATIDA")
    else:
        choice = "rock"
        reasons.append("nada dominou com clareza; rock e o padrao conservador")

    if voice:
        reasons.append(f"voz presente ({vocals.activity:.0%} do tempo) -> PART VOCALS")
    if bass is not None and bass.present:
        reasons.append(f"graves presentes -> PART BASS")

    survey.recommended = choice
    survey.reasons = reasons
    return survey


def render(survey: Survey) -> str:
    """Relatorio legivel no terminal."""
    lines = [f"duracao: {survey.duration:.1f}s", ""]
    lines.append(f"  {'stem':10} {'ativo':>7} {'volume':>8} {'onsets/s':>9} "
                 f"{'tonal':>7} {'brilho':>9}")
    lines.append("  " + "-" * 60)
    for st in sorted(survey.stems.values(), key=lambda x: -x.activity):
        mark = "" if st.present else "   (so vazamento)"
        lines.append(f"  {st.name:10} {st.activity:6.0%} {st.energy_share:7.0%} "
                     f"{st.onset_rate:9.1f} {st.pitched_fraction:6.0%} "
                     f"{st.centroid_hz:8.0f}Hz{mark}")

    profile = survey.profile
    lines += ["", f"recomendado: --genre {survey.recommended}"]
    for reason in survey.reasons:
        lines.append(f"    - {reason}")
    lines.append("")
    for plan in profile.plans:
        available = [src for src in plan.sources
                     if src == "mix" or (src in survey.stems
                                         and survey.stems[src].present)]
        source = available[0] if available else "mix"
        lines.append(f"    {plan.track:13} <- {source:8} [{plan.strategy}]  {plan.note}")
    lines += ["", "  outros generos: " + ", ".join(
        p.name for p in profiles.canonical() if p.name != survey.recommended)]
    return "\n".join(lines)
