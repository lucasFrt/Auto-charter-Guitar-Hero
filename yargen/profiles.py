"""Perfis de genero: qual stem alimenta qual trilha, e como.

A premissa que este modulo existe para quebrar: "a trilha de guitarra vem do
stem de guitarra, mapeada por altura". Isso so vale para rock. Em trap nao ha
guitarra nenhuma - o que e gostoso de tocar e a batida e o fluxo da voz. Em
pop o lead costuma ser sintetizador ou teclado, nao guitarra.

Um perfil e um PLANO DE ROTEAMENTO: para cada trilha do jogo, de qual stem ela
sai e qual estrategia de mapeamento usar. Genero e so um jeito compacto de
escolher um plano bom de uma vez.

Isto nao e - e nao deveria ser - totalmente automatico. `yargen inspect` mede a
musica e RECOMENDA um plano; quem decide e a pessoa. O objetivo e que a
escolha seja barata e informada, nao que ela desapareca.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

# Estrategias de mapeamento.
PITCH = "pitch"
"""Altura -> traste. Faixas de percentil em janela movel. Para conteudo
melodico: guitarra, baixo, teclado, sintetizador, voz."""

PERCUSSIVE = "percussive"
"""Timbre -> traste. Faixas de percentil GLOBAIS sobre o brilho (centroide
espectral). Para bateria e percussao, que nao tem altura: bumbo e escuro,
caixa e media, chimbal e brilhante, e isso mapeia direto em grave->agudo no
braco. Globais e nao moveis porque o bumbo precisa ser verde no compasso 1 e
no compasso 80 - uma janela movel reclassificaria ele conforme o arranjo muda,
e a batida deixaria de ser reconhecivel."""

VOCAL = "vocal"
"""Trilha PART VOCALS: altura + letra. Nao usa trastes."""

STRATEGIES = (PITCH, PERCUSSIVE, VOCAL)

GUITAR_TRACK = "PART GUITAR"
BASS_TRACK = "PART BASS"
VOCALS_TRACK = "PART VOCALS"

# Nome curto usado na CLI -> trilha do jogo.
INSTRUMENT_TRACKS = {"guitar": GUITAR_TRACK, "bass": BASS_TRACK,
                     "vocals": VOCALS_TRACK}


@dataclass
class TrackPlan:
    """De onde sai uma trilha e como ela e mapeada."""

    track: str
    sources: list[str]
    """Stems em ordem de preferencia. O primeiro que existir e usado; "mix" e
    sempre o ultimo recurso e sempre funciona."""

    strategy: str = PITCH
    overrides: dict[str, Any] = field(default_factory=dict)
    """Patch de config aplicado so a esta trilha, em caminho pontilhado:
    {"mapper.grid_division": 32}. Um rolo de chimbal de trap e um objeto
    diferente de um riff de rock e nao cabe nos mesmos numeros."""

    note: str = ""
    """Frase curta explicando a escolha, mostrada no log. Quem roda precisa
    saber que a trilha de guitarra esta sendo charteada da bateria."""

    @property
    def instrument(self) -> str:
        for short, track in INSTRUMENT_TRACKS.items():
            if track == self.track:
                return short
        return self.track.lower()


@dataclass
class GenreProfile:
    name: str
    plans: list[TrackPlan]
    aliases: tuple[str, ...] = ()
    description: str = ""

    def for_instruments(self, instruments: Sequence[str]) -> list[TrackPlan]:
        wanted = {i.lower() for i in instruments}
        return [p for p in self.plans if p.instrument in wanted]


# ---------------------------------------------------------------- overrides

# Trap/drill: rolos de chimbal em fusas e semifusas sao a assinatura do genero.
# Grade de 1/32 para o rolo nao ser esmagado, e teto de densidade alto porque
# o proprio jogo transforma sequencia rapida em HOPO, que e tocavel.
_TRAP_LEAD = {
    "mapper.grid_division": 32,
    "mapper.max_notes_per_second": 9.0,
    "mapper.sustain_min_beats": 4.0,   # bateria nao sustenta
    "mapper.chord_min_strength": 0.25,
    "mapper.anchor_enabled": False,    # faixas globais ja dao estabilidade
    # Em bateria, bumbo+chimbal simultaneos sao o padrao, e essa e exatamente
    # a "casa" que o jogador quer sentir. A regra de acorde adjacente existe
    # porque uma mao nao atravessa o braco de guitarra; para bateria ela
    # destruiria a identidade da batida, entao aqui ela sai.
    "mapper.chord_adjacent_only": False,
}

# 808 escorrega de altura de proposito (glide). Janela de ancora curta para o
# glide nao ficar preso ao traste da nota anterior.
_808_BASS = {
    "mapper.anchor_window_sec": 2.0,
    "mapper.max_notes_per_second": 4.0,
    "mapper.sustain_min_beats": 1.0,   # 808 e quase todo sustain
}

_METAL_LEAD = {
    "mapper.grid_division": 16,
    "mapper.max_notes_per_second": 11.0,
    "mapper.max_jump": 3,
    "mapper.sustain_min_beats": 2.0,
}

_SYNTH_LEAD = {
    # Sintetizador sustenta muito mais que guitarra palhetada; sem isso o
    # chart vira uma parede de notas secas onde a musica tem pads longos.
    "mapper.sustain_min_beats": 1.0,
    "mapper.max_notes_per_second": 6.0,
}


# ----------------------------------------------------------------- perfis

def _vocals_plan(note: str = "voz principal") -> TrackPlan:
    return TrackPlan(VOCALS_TRACK, ["vocals", "mix"], VOCAL, note=note)


PROFILES: dict[str, GenreProfile] = {}


def _register(profile: GenreProfile) -> GenreProfile:
    PROFILES[profile.name] = profile
    for alias in profile.aliases:
        PROFILES[alias] = profile
    return profile


_register(GenreProfile(
    "rock",
    aliases=("punk", "indie", "blues", "grunge", "classic-rock"),
    description="Guitarra conduz. A premissa padrao, e a unica em que ela vale.",
    plans=[
        TrackPlan(GUITAR_TRACK, ["guitar", "other", "mix"], PITCH,
                  note="guitarra conduz"),
        TrackPlan(BASS_TRACK, ["bass", "mix"], PITCH, note="baixo"),
        _vocals_plan(),
    ]))

_register(GenreProfile(
    "metal",
    aliases=("hardcore", "thrash", "djent"),
    description="Como rock, mas mais denso e com mais salto permitido.",
    plans=[
        TrackPlan(GUITAR_TRACK, ["guitar", "other", "mix"], PITCH, _METAL_LEAD,
                  note="guitarra conduz, densidade alta"),
        TrackPlan(BASS_TRACK, ["bass", "mix"], PITCH, note="baixo"),
        _vocals_plan(),
    ]))

_register(GenreProfile(
    "acoustic",
    aliases=("folk", "mpb", "bossa", "sertanejo", "country"),
    description="Violao e voz. Sem bateria conduzindo.",
    plans=[
        TrackPlan(GUITAR_TRACK, ["guitar", "other", "mix"], PITCH,
                  {"mapper.max_notes_per_second": 5.0},
                  note="violao conduz"),
        _vocals_plan(),
    ]))

_register(GenreProfile(
    "pop",
    aliases=("rnb", "soul", "funk-pop"),
    description=("Lead vem de teclado/sintetizador ('other'), nao de guitarra. "
                 "Em pop moderno a guitarra costuma ser textura, nao o gancho."),
    plans=[
        TrackPlan(GUITAR_TRACK, ["other", "guitar", "mix"], PITCH, _SYNTH_LEAD,
                  note="teclado/sintetizador conduz (em pop raramente e a guitarra)"),
        TrackPlan(BASS_TRACK, ["bass", "mix"], PITCH, note="baixo"),
        _vocals_plan(),
    ]))

_register(GenreProfile(
    "trap",
    aliases=("hiphop", "hip-hop", "rap", "drill", "phonk"),
    description=("Nao ha guitarra. A trilha de 5 trastes vira a BATIDA - que e o "
                 "que se quer tocar - e a voz vai para PART VOCALS, para dar a "
                 "sincronia com os dois."),
    plans=[
        TrackPlan(GUITAR_TRACK, ["drums", "mix"], PERCUSSIVE, _TRAP_LEAD,
                  note="a BATIDA conduz a trilha de 5 trastes (nao ha guitarra)"),
        TrackPlan(BASS_TRACK, ["bass", "mix"], PITCH, _808_BASS,
                  note="808 / linha de graves"),
        _vocals_plan("fluxo vocal"),
    ]))

_register(GenreProfile(
    "electronic",
    aliases=("edm", "house", "techno", "dnb", "dubstep", "synthwave"),
    description="Lead sintetizado conduz; a batida vai para o baixo se nao houver 808.",
    plans=[
        TrackPlan(GUITAR_TRACK, ["other", "mix"], PITCH, _SYNTH_LEAD,
                  note="lead sintetizado conduz"),
        TrackPlan(BASS_TRACK, ["bass", "mix"], PITCH, note="baixo/sub"),
        _vocals_plan(),
    ]))

_register(GenreProfile(
    "percussion",
    aliases=("drums", "batida", "instrumental-beat"),
    description="Tudo sai da bateria. Para faixas instrumentais e beats sem voz.",
    plans=[
        TrackPlan(GUITAR_TRACK, ["drums", "mix"], PERCUSSIVE, _TRAP_LEAD,
                  note="bateria conduz"),
        TrackPlan(BASS_TRACK, ["bass", "mix"], PITCH, note="baixo"),
    ]))


def get(name: str) -> GenreProfile:
    key = (name or "rock").strip().lower()
    if key not in PROFILES:
        raise KeyError(f"genero desconhecido: {name!r}. Disponiveis: "
                       f"{', '.join(available())}")
    return PROFILES[key]


def available() -> list[str]:
    return sorted(PROFILES)


def canonical() -> list[GenreProfile]:
    """Um perfil por nome canonico, sem repetir os apelidos."""
    seen: dict[str, GenreProfile] = {}
    for profile in PROFILES.values():
        seen.setdefault(profile.name, profile)
    return [seen[k] for k in sorted(seen)]
