# Criando uma LLM do zero para fins didáticos

O que acontece entre digitar um prompt e ver a resposta aparecer? O **LLMemCasa** é um laboratório de terminal para acompanhar como um modelo de linguagem gera texto, token por token.

O projeto usa o **Qwen2.5-1.5B-Instruct** pré-treinado pelo Qwen. O tokenizador, a arquitetura e os pesos vêm do modelo. O código implementa o loop de geração em Python com PyTorch para mostrar como os logits viram probabilidades e como a escolha de um token influencia a próxima.

## O que aparece no terminal

1. **Tokenização.** O script aplica o template de conversa do Qwen e exibe os tokens do prompt e seus IDs.
2. **Embeddings.** O script consulta o vetor de entrada de cada token e imprime as dimensões do tensor e os primeiros valores de cada vetor.
3. **Passagem pelo modelo.** O script mostra os estados ocultos e usa a similaridade de cosseno para comparar a representação inicial e final do último token do prompt.
4. **Contexto da conversa.** O terminal mostra o orçamento do prompt, quantos tokens estão em uso e quantos turnos anteriores cabem. O script mantém a mensagem de sistema, a pergunta atual e os turnos completos mais recentes que cabem no orçamento.
5. **Geração com KV cache.** A primeira passagem processa o prompt inteiro e guarda as chaves e os valores de atenção. A cada passo seguinte, o script reaproveita esse cache e envia ao modelo só o token novo. O terminal mostra os candidatos mais prováveis e o token escolhido.

## Como rodar

O projeto foi testado com Python 3.12 em CPU. Na primeira execução, o Hugging Face baixa e guarda em cache os arquivos do modelo. Depois disso, é possível executar com os arquivos em cache.

```sh
git clone https://github.com/Gabriel-Camim/LLMemCasa.git
cd LLMemCasa
python -m venv .venv
```

No Windows, execute estes comandos no PowerShell.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

No Linux ou macOS, use o Python do ambiente virtual.

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py
```

## Exemplo de execução

Digite `Quanto e 2 mais 2?` e depois `Some 1 ao resultado anterior.` para acompanhar como o histórico entra no novo prompt. As mensagens ficam em memória durante a sessão.

- `historico` exibe as mensagens da sessão.
- `limpar` reinicia a conversa com a mensagem de sistema.
- `sair` encerra o programa.

## Configuração do experimento

Por padrão, a janela de contexto é de `1024` tokens, com `128` reservados para a resposta e `896` disponíveis para o prompt. As variáveis no início de `main.py` permitem ajustar esses limites e ligar ou desligar a inspeção e a lista de candidatos.

A temperatura inicial é `0.7` e ajusta as probabilidades exibidas. O script usa `argmax` para escolher o token de maior probabilidade e mostra, por padrão, os `5` candidatos mais prováveis a cada passo.
