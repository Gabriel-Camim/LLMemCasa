# LLM caseira / Homemade LLM

[Português](#português) · [English](#english) · [LLMemCasa](../README.md)

## Português

O que acontece entre digitar um prompt e ver a resposta aparecer? Este laboratório de terminal implementa o loop de geração em Python com PyTorch. O tokenizador, a arquitetura e os pesos pré-treinados vêm do [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).

1. **Tokenização e embeddings.** O script aplica o template de conversa do Qwen e mostra os tokens, seus IDs e os primeiros valores de cada vetor de entrada.
2. **Passagem pelo modelo.** O terminal mostra os estados ocultos de cada camada e compara as representações inicial e final do último token do prompt por similaridade de cosseno.
3. **Contexto.** O terminal mostra quantos tokens o prompt ocupa e quantos turnos anteriores cabem. O script inclui a mensagem de sistema, a pergunta atual e os turnos completos mais recentes que couberem.
4. **Geração com KV cache.** A primeira passagem processa o prompt e guarda as chaves e os valores de atenção. A cada passo seguinte, o modelo recebe o token novo e reaproveita o cache. O terminal mostra as probabilidades dos candidatos e o token escolhido.

### Como rodar

Execute os comandos no diretório do repositório. A configuração foi testada com Python 3.12 e PyTorch em CPU. Na primeira execução, o Hugging Face baixa os arquivos do modelo. Nas seguintes, o script pode usar o cache local.

No PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

No Linux ou macOS, crie o ambiente com `python3.12 -m venv .venv` e use `.venv/bin/python` nos comandos. A instalação do PyTorch deve seguir a plataforma. [Instalação do PyTorch](https://pytorch.org/get-started/locally/)

Digite `Quanto é 2 mais 2?` e depois `Some 1 ao resultado anterior.` para ver como o histórico entra no novo prompt. Use `historico` para ver as mensagens em RAM, `limpar` para reiniciar a conversa e `sair` para encerrar.

A janela padrão é de 1024 tokens, com 128 reservados para a resposta e 896 para o prompt. As variáveis no começo de `main.py` ajustam esses limites e controlam a inspeção do modelo. A temperatura de 0,7 muda as probabilidades exibidas. A escolha do token é gulosa (`argmax`), e o terminal mostra os cinco candidatos mais prováveis por padrão.

## English

What happens between typing a prompt and seeing the answer? This terminal lab implements the generation loop in Python with PyTorch. The tokenizer, architecture, and pretrained weights come from [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).

1. **Tokenization and embeddings.** The script applies Qwen's chat template and displays the tokens, their IDs, and the first values of each input vector.
2. **Forward pass.** The terminal shows the hidden states across layers and uses cosine similarity to compare the initial and final representations of the last prompt token.
3. **Context.** The terminal shows how many tokens the prompt uses and how many prior turns fit. The script includes the system message, the current question, and the most recent complete turns that fit.
4. **Generation with a KV cache.** The first pass processes the prompt and caches the attention keys and values. Each subsequent pass receives the new token and reuses the cache. The terminal displays candidate probabilities and the selected token.

### Run it

Run these commands from the repository root. The tested setup used Python 3.12 and PyTorch on CPU. The first run downloads model files from Hugging Face. Later runs can use the local cache.

In PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

On Linux or macOS, create the environment with `python3.12 -m venv .venv` and use `.venv/bin/python` in the commands. Follow the PyTorch installation instructions for your platform. [PyTorch installation](https://pytorch.org/get-started/locally/)

Enter `Quanto é 2 mais 2?` (What is 2 plus 2?) and then `Some 1 ao resultado anterior.` (Add 1 to the previous result.) to see how history becomes part of the next prompt. Use `historico` to view the messages in RAM, `limpar` to reset the conversation, and `sair` to exit.

The default context window holds 1,024 tokens, with 128 reserved for the answer and 896 for the prompt. Variables at the top of `main.py` control these limits and the inspection output. The temperature of 0.7 changes the displayed probabilities. Token selection is greedy (`argmax`), and the terminal shows the five most likely candidates by default.
