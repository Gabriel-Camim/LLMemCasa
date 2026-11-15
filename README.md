# LLMemCasa — Local LLMs on CPU

[Português](#português) · [English](#english)

## Português

Dois experimentos de inferência local **só na CPU**. Na LLM caseira, o terminal mostra como um token é escolhido. No Huihui, o terminal recebe a resposta em streaming de um modelo de 27 bilhões de parâmetros quantizado em GGUF.

| Módulo | O que faz |
| --- | --- |
| [LLM caseira](docs/llm-caseira.md#português) · `main.py` | Loop de geração implementado em Python com PyTorch e pesos pré-treinados do Qwen2.5-1.5B-Instruct. Mostra tokenização, embeddings, estados ocultos, KV cache e os tokens mais prováveis a cada passo. |
| [Huihui abliterado](huihui/README.md#português) · `huihui/main.py` | Chat com o Huihui-Qwen3.8-27B-abliterated em GGUF `UD-IQ2_S`. O `llama.cpp` roda como servidor local e o cliente em Python exibe a resposta à medida que ela chega. |

Clone o repositório e siga o guia do módulo que você quer executar.

```sh
git clone https://github.com/Gabriel-Camim/LLMemCasa.git
cd LLMemCasa
```

### Hardware

Os dois módulos foram executados no Windows x64 em um **Intel Core i7-8750H (2,20 GHz, 6 núcleos, 12 threads)** com **16 GB de RAM**.

A **NVIDIA GeForce GTX 1050 Ti (4 GB de VRAM, Pascal, `sm_61`)** ficou fora da inferência. Ao tentar colocar uma camada do Huihui na GPU, o pacote CUDA do `llama.cpp` b12387 retornou `no kernel image is available for execution on the device`. O Huihui roda com a build CPU b12397 e zero camadas na GPU (`-ngl 0`). [Arquitetura da GTX 1050 Ti na tabela da NVIDIA](https://developer.nvidia.com/cuda/gpus/legacy)

## English

Two **CPU-only** local inference experiments. The homemade module shows how each token is selected in the terminal. The Huihui module streams output from a quantized 27-billion-parameter GGUF model.

| Module | What it does |
| --- | --- |
| [Homemade LLM](docs/llm-caseira.md#english) · `main.py` | A Python generation loop using PyTorch and pretrained Qwen2.5-1.5B-Instruct weights. It displays tokenization, embeddings, hidden states, the KV cache, and the most likely tokens at each step. |
| [Abliterated Huihui](huihui/README.md#english) · `huihui/main.py` | Chat with Huihui-Qwen3.8-27B-abliterated in `UD-IQ2_S` GGUF format. The Python client streams responses from a local `llama.cpp` server. |

Clone the repository and follow the guide for the module you want to run.

```sh
git clone https://github.com/Gabriel-Camim/LLMemCasa.git
cd LLMemCasa
```

### Hardware

Both modules ran on Windows x64 with an **Intel Core i7-8750H (2.20 GHz, 6 cores, 12 threads)** and **16 GB of RAM**.

The **NVIDIA GeForce GTX 1050 Ti (4 GB VRAM, Pascal, `sm_61`)** was unused for inference. Attempting to offload one Huihui layer to the GPU with the `llama.cpp` b12387 CUDA package returned `no kernel image is available for execution on the device`. Huihui runs with the b12397 CPU build and zero GPU layers (`-ngl 0`). [NVIDIA's GPU compute capability table](https://developer.nvidia.com/cuda/gpus/legacy)
