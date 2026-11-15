# Huihui abliterado / Abliterated Huihui

[Português](#português) · [English](#english) · [LLMemCasa](../README.md)

## Português

Chat de terminal com o [Huihui-Qwen3.8-27B-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF), quantizado em GGUF `UD-IQ2_S`. O cliente em Python inicia o `llama.cpp` em `127.0.0.1:8089` e exibe a resposta em streaming. O histórico fica em RAM durante a sessão.

### Como rodar no Windows x64

1. Baixe o [pacote Windows x64 CPU do llama.cpp b12397](https://github.com/ggml-org/llama.cpp/releases/download/b12397/llama-b12397-bin-win-cpu-x64.zip).
2. Extraia o ZIP para `llama_cpp_cpu` na raiz do repositório. O executável deve ficar em `llama_cpp_cpu/llama-server.exe`, junto com as DLLs do pacote.
3. Na raiz do repositório, execute no PowerShell com Python 3.12:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r huihui\requirements.txt
   .\.venv\Scripts\python.exe huihui\main.py
   ```

Na primeira execução, o script baixa o GGUF de cerca de 8,42 GB para `huihui/modelos/`. O servidor usa seis threads de CPU, zero camadas na GPU e uma janela de 1024 tokens, com até 512 tokens de geração. A seleção do histórico usa uma estimativa conservadora, e o servidor faz a tokenização efetiva.

Use `historico` para ver a conversa, `limpar` para reiniciá-la e `sair` para encerrar o chat e o servidor. O log do servidor fica em `huihui/llama_server.log`.

## English

A terminal chat with [Huihui-Qwen3.8-27B-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF) quantized as `UD-IQ2_S` GGUF. The Python client starts `llama.cpp` at `127.0.0.1:8089` and streams the response. Conversation history stays in RAM for the session.

### Run it on Windows x64

1. Download the [llama.cpp b12397 Windows x64 CPU package](https://github.com/ggml-org/llama.cpp/releases/download/b12397/llama-b12397-bin-win-cpu-x64.zip).
2. Extract the archive to `llama_cpp_cpu` at the repository root. Put `llama-server.exe` and the DLLs from the package directly in that folder.
3. From the repository root, run these commands in PowerShell with Python 3.12:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r huihui\requirements.txt
   .\.venv\Scripts\python.exe huihui\main.py
   ```

On the first run, the script downloads the roughly 8.42 GB GGUF to `huihui/modelos/`. The server uses six CPU threads, zero GPU layers, and a 1,024-token context window with up to 512 generation tokens. History selection uses a conservative estimate, and the server performs the actual tokenization.

Use `historico` to view the conversation, `limpar` to reset it, and `sair` to stop the chat and server. The server writes its log to `huihui/llama_server.log`.
