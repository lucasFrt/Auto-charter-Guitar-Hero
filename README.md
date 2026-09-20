# YARGen

Transforma um arquivo de áudio em uma pasta de música pronta para o
[YARG](https://yarg.in) / Clone Hero: `notes.mid` + `song.ini` + áudio.

O alvo não é um chart perfeito. É um chart **jogável e razoável** para abrir no
Moonscraper/REAPER e ajustar. Tudo roda local e offline — nenhuma API paga,
nenhuma chamada de rede em runtime (só no download inicial dos pesos dos
modelos).

```bash
yargen inspect "Musica.mp3"                  # o que essa musica tem? o que chartear?
yargen chart "Musica.mp3" --genre trap --instruments all --keep-ir
```

## Instalação

O PyTorch é o maior atrito de instalação do projeto, então ele **não** é
dependência base. Instale só o que for usar:

```bash
pip install -e .                 # DSP + MIDI. Funciona sozinho (M0, M1, M3, M4).
pip install -e ".[stems]"        # + Demucs: charteia o stem certo em vez da mixagem.
pip install -e ".[vocals]"       # + faster-whisper: letra sincronizada.
pip install -e ".[all]"
```

Em CPU, prefira instalar o torch pelo índice CPU-only (bem menor):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Sem os extras nada quebra: o pipeline detecta a ausência, avisa e segue com o
que tem. Um chart sem letra é muito melhor do que uma rodada de vários minutos
que morre no último passo.

## A decisao que mais importa: o que chartear

A premissa obvia - "a trilha de guitarra vem da guitarra" - so vale para rock.
Em trap nao ha guitarra nenhuma: o que e gostoso de tocar e a batida, e a voz
merece a propria trilha. Em pop o gancho costuma ser teclado ou sintetizador.
Se o projeto assumisse guitarra sempre, ele so serviria para um genero.

Entao o roteamento **fonte -> trilha** e uma decisao de primeira classe, e um
perfil de genero e so um jeito compacto de escolher um bom conjunto de uma vez:

| genero | PART GUITAR vem de | estrategia |
|---|---|---|
| `rock`, `metal`, `acoustic` | stem de guitarra | altura |
| `pop` | teclado/sintetizador (`other`) | altura |
| `electronic` | lead sintetizado | altura |
| `trap`, `percussion` | **bateria** | **timbre** |

`yargen genres` lista todos com o roteamento completo.

Isto **nao e** - e nao deveria ser - automatico. Jogar um MP3 e receber o chart
certo exige uma decisao musical que a maquina nao tem como tomar por voce. O
que da para fazer e tornar a escolha barata e informada:

```
$ yargen inspect "Beat.wav"

  stem         ativo   volume  onsets/s   tonal    brilho
  ------------------------------------------------------------
  bass          93%      6%       1.7   100%       78Hz
  drums         86%     49%       5.0    43%     4287Hz
  vocals        22%     45%       0.9   100%      424Hz
  guitar         0%      0%       0.0     0%        0Hz   (so vazamento)

recomendado: --genre trap
    - nenhum lead melodico; a bateria conduz (86% do tempo, 5.0 onsets/s)
    - guitarra ausente (0% do tempo)
    - a trilha de 5 trastes vai charteiar a BATIDA
```

Ele mede, recomenda e **diz por que** - para voce discordar com conhecimento de
causa. Um classificador de genero opaco seria pior aqui mesmo se acertasse mais.

Quando o perfil quase acerta, `--set-track` e o escape sem precisar de um
perfil novo:

```bash
yargen chart musica.ogg --genre rock --set-track guitar.source=other
yargen chart musica.ogg --genre pop  --set-track guitar.strategy=percussive
```

### Por que a bateria nao usa altura

Nao existe "a altura de um bumbo". Mas bateria tem uma escala perceptual
estavel e obvia: **brilho**. Bumbo e escuro, caixa e media, chimbal e
brilhante - e isso mapeia direto em grave->agudo no braco, a mesma intuicao que
faz o mapeamento por altura funcionar para guitarra. O mapper nao distingue os
dois casos: ele recebe um escalar por nota e distribui em cinco faixas.

Duas diferencas importam:

- As faixas sao **globais**, nao moveis. Um kit tem quatro ou cinco sons e eles
  nao mudam: o bumbo precisa ser verde no compasso 1 e no compasso 80.
- Os onsets sao detectados **por banda de frequencia**, independentemente. Numa
  bateria, tocar duas pecas ao mesmo tempo e a regra: o chimbal marca todas as
  colcheias e o bumbo cai por cima. Um detector de banda larga funde os dois e
  o brilho medido vira uma mistura que nao corresponde a peca nenhuma - o bumbo
  sumia e o chart virava uma parede de chimbal. Por banda, os dois viram um
  acorde, que e exatamente o que se quer sentir.

Medido contra as 168 pancadas conhecidas de um beat sintetico:
`P=0.788 R=0.839 F1=0.813`, usando os cinco trastes e 22 acordes.

## Uso

```bash
# tudo de uma vez
yargen chart musica.ogg \
  --out "Artista - Musica/" \
  --genre trap \
  --instruments guitar,bass,vocals \
  --difficulties all \
  --bpm 174 \            # opcional, sobrepõe a detecção
  --offset-ms 0 \
  --keep-ir

# ou em dois passos — é assim que se ajusta o mapper
yargen analyze musica.ogg --out ir.json     # o passo caro: roda uma vez
yargen build ir.json --set mapper.max_jump=3 --out "teste/"   # segundos
yargen build ir.json --set mapper.band_window_sec=12 --out "teste2/"

# régua objetiva contra um chart humano
yargen validate "teste/notes.mid" "chart-da-comunidade/notes.mid"

yargen inspect musica.ogg    # mede os stems e recomenda um genero
yargen genres                # lista os perfis e o roteamento de cada um
yargen config                # imprime todos os parâmetros com seus valores
yargen cache list|clear      # cache de stems do Demucs
```

Qualquer parâmetro pode ser sobreposto sem editar arquivo:
`--set mapper.max_jump=3`, `--set difficulty.easy.max_notes_per_second=2.5`.
Um nome de parâmetro errado falha alto em vez de ser ignorado em silêncio.

## Como está organizado

O princípio central é **separar análise, charting e escrita** por uma
representação intermediária serializável.

```
áudio
  ├─► [stems]   Demucs → guitar/bass/drums/vocals   (cache por hash do arquivo)
  ├─► [tempo]   grade de beats, BPM, compasso
  ├─► [onsets]  por stem → (tempo, força)
  ├─► [pitch]   por stem → contorno f0
  ▼
AnalysisIR (JSON) ──► [mapper] ──► ChartIR (JSON) ──► [writer] ──► notes.mid
                                                                   song.ini
```

Por que a IR importa na prática: `analyze` leva minutos (Demucs, pyin,
whisper), `build` leva segundos. Ajustar o mapper é editar um número e rodar
`build` de novo — sem reprocessar áudio. E o mapper, sendo função pura sobre a
IR, é testado com JSON fixo, sem DSP.

```
yargen/
├── config.py            ★ TODOS os parâmetros, em um arquivo só
├── profiles.py          ★ roteamento fonte → trilha por gênero
├── audio/loader.py      carga, resample, normalização
├── audio/stems.py       Demucs + cache em ~/.cache/yargen
├── analysis/tempo.py    beat tracking, grade, BPM
├── analysis/onsets.py   onsets, banda larga e por banda de frequência
├── analysis/pitch.py    pyin / contorno f0
├── analysis/timbre.py   brilho por onset (o "tom" do conteúdo percussivo)
├── analysis/survey.py   medição dos stems + recomendação de gênero
├── chart/ir.py          IRs + TempoMap
├── chart/mapper.py      ★ núcleo: pitch+onset → trastes
├── chart/density.py     teto de notas/segundo (compartilhado)
├── chart/difficulty.py  redução Expert → Hard/Medium/Easy
├── chart/vocals.py      alinhamento letra + pitch
├── chart/writer_mid.py  mido
├── meta/song_ini.py
├── validate/compare.py  precisão/recall contra chart humano
├── pipeline.py
└── cli.py
```

## O mapper

É onde está o valor do projeto, e é onde vai acontecer 80% da iteração.
Estágios, na ordem em que rodam:

1. **Quantizar** à grade. Se a posição quantizada cair a mais de 90ms do onset
   real, a nota fica onde estava — válvula de segurança contra grade errada
   (rubato, gravação ao vivo) puxar tudo para o lugar errado.
2. **Agrupar acordes**: onsets a menos de ~30ms viram nota simultânea. Onsets
   fracos não entram em acorde, senão todo vazamento de bateria vira acorde.
3. **Cortar densidade** — antes de atribuir trastes, senão o limite de salto é
   calculado sobre notas que vão ser jogadas fora.
4. **Tom → traste**: 5 faixas por percentil. O "tom" é altura (melódico) ou
   brilho (percussivo); as faixas são móveis (8s) ou globais conforme o caso.
   Âncora (tom repetido → mesmo traste) preserva riffs.
5. **Limitar saltos**: entre notas a menos de 150ms, no máximo 2 trastes.
6. **Sustains**, **HOPO**, **Star Power**.

Todos os números vivem em `config.py`, com um comentário dizendo o que muda
quando você mexe.

### Dois trade-offs que foram medidos, não chutados

**Janela de percentil (`mapper.band_window_sec`, padrão 8s).** A janela é
móvel porque o registro da música muda entre seções. Mas ela se auto-centra:
numa corrida ascendente longa, cada nota fica no meio da própria janela e o
solo inteiro sai amarelo. Janela longa demais vira global e para de acompanhar
as seções. Medido em `tests/test_mapper.py`:

| janela | corrida cromática de 2 oitavas | seções em registros diferentes |
|---|---|---|
| 2–4s | `GGRRRYYYYYYYYYYYYYBBBOOO` (empilha no meio) | cada seção usa as 5 cores |
| **8s** | `GGGGRRRRRRYYYYBBBBBOOOOO` (varre o braço) | ainda acompanha |
| 20s+ | varre o braço | grave fica só no grave |

**Teto de densidade.** O limite é um número inteiro de notas por janela, então
uma janela curta não representa um teto fracionário: 1.5 notas/s numa janela
de 1s vira limite 1, e o Easy sai 40% mais esparso do que a config pede.
`chart/density.py` alarga a janela até o teto ser representável.

## Formato de saída

Referência: **[TheNathannator/GuitarGame_ChartFormats](https://github.com/TheNathannator/GuitarGame_ChartFormats)**,
que é a documentação que o próprio YARG usa. Pontos que valem registro:

- MIDI **tipo 1**, ticks-por-semínima. Tipo 0/2 e SMPTE não carregam.
- Dificuldades são oitavas: Easy 60, Medium 72, Hard 84, Expert 96. Dentro de
  cada uma o layout é igual: `base+0..4` = cores, `base+5` = force HOPO,
  `base+6` = force strum, `base-1` = nota aberta.
- **HOPO é automático no jogo**: vira HOPO se estiver a no máximo
  `(resolution/3)+1` ticks da anterior, não for a mesma casa e não for acorde.
  Por isso o writer calcula o estado natural e só grava 101/102 quando o
  desejado diverge — gravar marcador em tudo é ruído e muda o hash do chart
  sem mudar como ele se joga.
- Sustain menor que `resolution/3` é descartado pelo jogo.
- Nota aberta sai por **SysEx do Phase Shift** (o método recomendado pela doc);
  o marcador por nota é mais novo e exige `[ENHANCED_OPENS]`.
- Vocals: altura em 36–84, letra como meta-evento `lyric`, frases na nota 105.
  Sílaba sem altura confiável vai marcada com `#` (não-tonal) em vez de receber
  um chute — altura errada em vocals é imediatamente frustrante.
- O nome do arquivo de áudio é **reservado** (`song`, `guitar`, `bass`,
  `drums`, `vocals`...). Nome arbitrário simplesmente não carrega, então o
  writer renomeia na cópia.

**Sincronia.** MP3 tem padding do encoder que desloca o áudio alguns
milissegundos. Prefira `.ogg`; se as notas soarem adiantadas ou atrasadas,
ajuste com `--offset-ms`, que é gravado no campo `delay`.

## Validação

```bash
yargen validate gerado/notes.mid referencia/notes.mid --tolerance-ms 50
# P=0.737 R=0.656 F1=0.694  (75/114 geradas, 75/128 de referência, erro médio 5.2ms)
```

Precisão/recall dos onsets contra um chart humano, tolerância ±50ms. Não vai
bater 100% e nem precisa: serve como régua para saber se uma mudança nos
parâmetros melhorou ou piorou. Sem isso, ajuste de heurística é achismo.

Testes: `pytest` (154 testes, ~3s, nenhum toca em áudio real).

## Estado dos marcos

| | | |
|---|---|---|
| M0 | esqueleto de saída | ✅ `notes.mid` + `song.ini` válidos, conferidos contra a doc canônica |
| M1 | grade + onsets | ✅ BPM exato e **erro médio de sincronia de 5ms** em teste fim a fim |
| M2 | stems | ⚠️ caminho de consumo validado; separação **não** executada (ver abaixo) |
| M3 | mapper de pitch | ✅ riff sai reconhecível e estável entre repetições |
| M4 | dificuldades | ✅ tabela de densidade/cores/acordes respeitada |
| M5 | vocals | ⚠️ caminho de altura validado; **faster-whisper não executado** (ver abaixo) |
| M6 | gênero / roteamento | ✅ perfis, mapeamento percussivo, `inspect` com recomendação |

### O que não pôde ser verificado nesta máquina

O ambiente onde isto foi desenvolvido bloqueia por política de rede o download
dos pesos dos modelos (`dl.fbaipublicfiles.com` e o HuggingFace, ambos 403).
Então:

- **Demucs nunca rodou.** O código de separação, o cache e o fallback entre
  modelos não foram exercidos contra o modelo real. O que foi validado é todo o
  caminho de *consumo*: stems reais injetados no cache, seleção da fonte,
  charting a partir deles. Rode uma vez com rede aberta antes de confiar.
- **faster-whisper nunca rodou.** A letra sincronizada não foi testada de
  verdade; só o caminho degradado (segmentação do contorno de f0, sem letra).

Medição com stems injetados, contra a verdade conhecida de uma música
sintética:

| | mixagem completa | stem de guitarra |
|---|---|---|
| Expert | P=0.474 R=0.500 | **P=0.737** R=0.656 |

Metade das notas do Expert charteado da mixagem vinha da bateria. É exatamente
o motivo de o M2 existir.

## Defeitos encontrados por medição

Registrados porque cada um estava silenciosamente errado e nenhum aparecia como
erro — só como chart pior:

- **Acordes nunca eram gerados**, em nenhuma estratégia. `assign_frets`
  reduzia um evento de vários tons à mediana e devolvia um traste só; o
  estágio de acordes recebia sempre uma lista de um elemento.
- **As faixas de percentil eram calculadas sobre a mediana de cada evento**,
  não sobre os tons individuais. Num acorde bumbo+chimbal a mediana cai no
  meio, onde não há peça nenhuma, e a caixa era classificada junto do bumbo.
- **`voiced_prob >= 0.5` apagava linhas de 808 inteiras.** O pyin marcava 91%
  dos frames como vozeados e acertava a altura, mas a probabilidade por frame
  não passava de 0.485 — ela não tem escala universal. `PART BASS` caía
  silenciosamente para trastes aleatórios. Agora confiamos na decisão do pyin.
- **A última nota sempre sustentava**, porque o valor de fallback do intervalo
  era exatamente o limiar de sustain. Inclusive uma pancada de bumbo.
- **Normalizar cada stem apagava o balanço da mixagem** — que é exatamente a
  informação que diz qual instrumento conduz a música.
- **Energia não responde "esta música tem guitarra?".** Com energia bruta, o
  808 de um beat de trap leva 78% e parece que a música é o baixo; com
  ponderação A ele cai para 4% e some. A recomendação passou a usar
  **atividade no tempo**, que não tem esse problema.

## Fora de escopo por enquanto

Bateria, baixo (a trilha existe, mas o mapper não é específico para baixo),
pro-keys, harmonias, interface gráfica.
