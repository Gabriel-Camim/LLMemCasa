# Criando uma LLM do zero para fins didáticos

O que acontece entre digitar um prompt e ver a resposta aparecer? O **LLMemCasa** é um laboratório de terminal para acompanhar como um modelo de linguagem gera texto, token por token.

O projeto usa o **Qwen2.5-1.5B-Instruct** pré-treinado pelo Qwen. O tokenizador, a arquitetura e os pesos vêm do modelo. O código implementa o loop de geração em Python com PyTorch para mostrar como os logits viram probabilidades e como a escolha de um token influencia a próxima.

## O que aparece no terminal

1. **Tokenização.** O script aplica o template de conversa do Qwen e exibe os tokens do prompt, seus IDs e o texto reconstruído.
2. **Embeddings.** O script consulta o vetor de entrada de cada token e imprime as dimensões do tensor e os primeiros valores de cada vetor.
3. **Passagem pelo modelo.** O script mostra os estados ocultos e usa a similaridade de cosseno para comparar a representação inicial e final do último token do prompt.
4. **Geração.** A cada passo, o script aplica temperatura e softmax aos logits, exibe os cinco tokens mais prováveis e mostra o token escolhido. A resposta parcial volta para a sequência de entrada do passo seguinte.

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

Digite `Quanto e 2 mais 2?` para acompanhar a geração. Em uma execução local, o texto final foi `2 + 2 é igual a 4.` Com a amostragem ativada, o resultado pode variar. Digite `sair` para encerrar.

## Configuração do experimento

As variáveis no início de `main.py` definem a temperatura, o limite de novos tokens e quantos candidatos aparecem na saída. A configuração inicial usa temperatura `0.2`, gera até `15` tokens e exibe os `5` candidatos mais prováveis por passo. Com temperatura `0`, o script escolhe sempre o token de maior pontuação. A lista de candidatos serve para inspeção. Com temperatura positiva, a amostragem considera a distribuição de todo o vocabulário.
