# Publicar no Windows Server com IIS

O IIS não executa Python. O caminho padrão é o módulo **HttpPlatformHandler**:
o IIS sobe o uvicorn como processo filho numa porta que ele mesmo escolhe e
encaminha as requisições para lá. O `web.config` desta pasta já faz isso —
falta ajustar três caminhos.

O desenho do servidor ajuda aqui: **nenhuma requisição HTTP espera o chart
ficar pronto**. O upload cria um job e volta na hora; o navegador pergunta o
andamento de segundo em segundo. Por isso não é preciso mexer em timeout de
nada — nem do IIS, nem do navegador — mesmo quando o Demucs leva cinco
minutos.

## 1. Pré-requisitos no servidor

- **Python 3.11 ou mais novo** (marque "Add python.exe to PATH" no instalador).
- **HttpPlatformHandler**: https://www.iis.net/downloads/microsoft/httpplatformhandler
- No Gerenciador do IIS, confirme que o módulo aparece em *Módulos*.

## 2. Instalar a aplicação

```powershell
# como Administrador
mkdir C:\inetpub\yargen
cd C:\inetpub\yargen
git clone https://github.com/lucasFrt/Auto-charter-Guitar-Hero .

py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip

# PyTorch CPU-only: ~200 MB em vez de ~2,5 GB da build com CUDA.
# Num servidor sem placa de vídeo, a versão CUDA é peso morto.
.\.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\pip install -e ".[all,web]"

mkdir C:\yargen\workspace, C:\yargen\torch, C:\yargen\huggingface, C:\inetpub\yargen\logs
```

Teste **fora do IIS** antes de qualquer outra coisa:

```powershell
.\.venv\Scripts\python -m webapp.server --port 8000
# abra http://127.0.0.1:8000 no servidor mesmo
```

Se não funcionar aqui, não vai funcionar no IIS — e o erro é muito mais fácil
de ler assim.

## 3. Configurar o site

Copie `webapp\web.config` para a **raiz do site** (`C:\inetpub\yargen\`) e
ajuste os três caminhos marcados com `<<<`:

| Campo | Valor |
|---|---|
| `processPath` | `C:\inetpub\yargen\.venv\Scripts\python.exe` |
| `stdoutLogFile` | `C:\inetpub\yargen\logs\stdout` |
| `PYTHONPATH` | `C:\inetpub\yargen` |

No Gerenciador do IIS: **Adicionar Site** apontando o caminho físico para
`C:\inetpub\yargen`.

## 4. Permissões (é aqui que costuma quebrar)

A identidade do pool de aplicativos (`IIS AppPool\<nome-do-pool>`) precisa de
**escrita** em:

```powershell
icacls C:\yargen /grant "IIS AppPool\yargen:(OI)(CI)M" /T
icacls C:\inetpub\yargen\logs /grant "IIS AppPool\yargen:(OI)(CI)M" /T
```

## 5. As três armadilhas conhecidas

**Upload falha com 404.13 e nenhuma explicação.** O IIS rejeita corpos acima
de 30 MB *antes* de chegar no Python. O `web.config` já sobe isso para 100 MB
em `maxAllowedContentLength` — mantenha esse valor acima de
`YARGEN_MAX_UPLOAD_MB`.

**A separação de stems falha na primeira execução.** O Demucs baixa os pesos
(~300 MB) para `%USERPROFILE%\.cache`, e a identidade do pool de aplicativos
normalmente não tem perfil de usuário. Por isso o `web.config` define
`TORCH_HOME` e `HF_HOME` apontando para pastas graváveis. O mesmo vale para o
`HF_HOME` do faster-whisper.

**Os modelos precisam de internet na primeira vez.** Demucs (~300 MB) e
faster-whisper (~500 MB no modelo `small`) baixam sob demanda. Se o servidor
não tem saída para a internet, rode uma vez numa máquina que tenha e copie
`C:\yargen\torch` e `C:\yargen\huggingface` para lá.

## 6. Configuração por variável de ambiente

Todas vão no bloco `<environmentVariables>` do `web.config`.

| Variável | Padrão | O que faz |
|---|---|---|
| `YARGEN_WORKSPACE` | `webapp\workspace` | Uploads e charts gerados |
| `YARGEN_MAX_UPLOAD_MB` | `80` | Tamanho máximo do upload |
| `YARGEN_KEEP_JOBS` | `40` | Jobs mantidos antes de podar os antigos |
| `TORCH_HOME` | — | Cache dos pesos do Demucs |
| `HF_HOME` | — | Cache dos modelos do faster-whisper |

## 7. Antes de expor para fora da rede local

Isto **não tem autenticação**. Ele aceita upload de arquivo e gasta CPU por
vários minutos por requisição — exposto na internet aberta, é um alvo fácil de
negação de serviço. Para uso interno em rede local está adequado. Para além
disso, ponha autenticação do Windows no site pelo IIS, ou deixe atrás de VPN.

O servidor escuta em `127.0.0.1` por padrão e é o IIS que fala com o mundo —
mantenha assim. Só use `--host 0.0.0.0` rodando sem o ISS na frente e sabendo
o que está fazendo.
