
import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM
)

# CONFIGURAÇÃO INICIAL

modelo_id = "Qwen/Qwen2.5-1.5B-Instruct"

temperatura = 0.2
max_new_tokens = 15
top_k_visualizacao = 5

print("Carregando tokenizador...")
tokenizer = AutoTokenizer.from_pretrained(modelo_id)

print("Carregando configuração...")
config = AutoConfig.from_pretrained(modelo_id)

print("Carregando pesos do modelo...")
model = AutoModelForCausalLM.from_pretrained(
    modelo_id,
    dtype=torch.float32
)

model.eval()

# INFORMAÇÕES DA ARQUITETURA

print("\n--- CONFIGURAÇÃO DO MODELO ---")
print("Dimensão oculta:", config.hidden_size)
print("Camadas:", config.num_hidden_layers)
print("Cabeças de atenção:", config.num_attention_heads)
print("Cabeças KV:", config.num_key_value_heads)
print("Vocabulário:", config.vocab_size)

print("\n--- HARDWARE ---")
print("PyTorch:", torch.__version__)
print("CUDA disponível:", torch.cuda.is_available())

print("\n--- CONFIGURAÇÃO DA GERAÇÃO ---")
print("Temperatura:", temperatura)
print("Máximo de novos tokens:", max_new_tokens)

# LOOP INTERATIVO

while True:

    texto = input("\nDigite um texto (ou 'sair'): ")

    if texto.lower() == "sair":
        break

    if not texto.strip():
        continue

    # CONSTRUÇÃO DO PROMPT CONVERSACIONAL

    mensagens = [
        {
            "role": "system",
            "content": (
                    """
                    Responda diretamente às perguntas do usuário.
                    Quando receber uma frase incompleta, pergunte se ele
                    deseja completá-la ou, se a intenção estiver clara,
                    continue a frase.

                    Não invente fatos quando não souber a resposta.
                    Quando precisar de informações externas, informe
                    que não possui acesso a elas.

                    Evite respostas excessivamente longas.
                    """
            )
        },
        {
            "role": "user",
            "content": texto
        }
    ]

    prompt = tokenizer.apply_chat_template(
        mensagens,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False
    )

    ids = inputs["input_ids"][0].tolist()

    # TOKENIZAÇÃO

    # Tokens do prompt completo, incluindo os marcadores
    # especiais de system, user e assistant.
    tokens = tokenizer.convert_ids_to_tokens(ids)

    print("\n--- TOKENIZAÇÃO ---")

    for token, token_id in zip(tokens, ids):
        print(f"{token!r:25} -> ID: {token_id}")

    print("\nQuantidade de tokens:", len(ids))

    print("\n--- PROMPT RECONSTRUÍDO ---")
    print(tokenizer.decode(ids))

    # EMBEDDING LOOKUP

    with torch.inference_mode():
        embeddings = model.get_input_embeddings()(
            inputs["input_ids"]
        )

    print("\n--- EMBEDDINGS ---")
    print("Shape:", tuple(embeddings.shape))

    for i, token in enumerate(tokens):

        print(f"\nToken {i}: {token!r}")
        print("Primeiros 10 valores:")
        print(embeddings[0, i, :10])

    # FORWARD PASS

    print("\n--- FORWARD PASS ---")

    with torch.inference_mode():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True
        )

    hidden_states = outputs.hidden_states

    print("Quantidade de estados:", len(hidden_states))

    for i, estado in enumerate(hidden_states):

        # Último token da sequência
        vetor = estado[0, -1, :]

        print(f"\nEstado {i:02d}")
        print("Shape:", tuple(estado.shape))
        print("Primeiros 5 valores:", vetor[:5].tolist())

    # COMPARAÇÃO DE REPRESENTAÇÕES

    # Comparar o mesmo token no começo e no final
    # do processamento pelas camadas.
    indice_token = -1

    inicial = hidden_states[0][0, indice_token, :]
    final = hidden_states[-1][0, indice_token, :]

    similaridade = torch.nn.functional.cosine_similarity(
        inicial.unsqueeze(0).float(),
        final.unsqueeze(0).float()
    )

    print("\n--- COMPARAÇÃO ---")
    print("Token analisado:", repr(tokens[indice_token]))

    print("Primeiros 10 valores iniciais:")
    print(inicial[:10])

    print("Primeiros 10 valores finais:")
    print(final[:10])

    print("Similaridade de cosseno:", similaridade.item())

    # GERAÇÃO AUTORREGRESSIVA

    print("\n--- GERAÇÃO ---")

    ids_gerados = inputs["input_ids"].clone()

    # O primeiro forward pass já foi realizado acima.
    # Reutilizamos seus logits para gerar o primeiro token.
    logits_atuais = outputs.logits[0, -1, :]

    # IDs que pertencem somente à resposta do assistente
    ids_resposta = []

    # Qwen usa im_end para encerrar uma mensagem.
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    ids_fim = {
        token_id
        for token_id in [tokenizer.eos_token_id, im_end_id]
        if token_id is not None
    }

    for passo in range(max_new_tokens):

        # LOGITS

        print(f"\n--- PASSO {passo + 1} ---")
        print("Shape dos logits:", tuple(logits_atuais.shape))

        # TEMPERATURA E SOFTMAX

        if temperatura > 0:

            logits_ajustados = logits_atuais / temperatura

            probabilidades = torch.softmax(
                logits_ajustados,
                dim=-1
            )

            # Amostragem probabilística
            proximo_id = torch.multinomial(
                probabilidades,
                num_samples=1
            )

        else:

            # Temperatura 0: geração determinística
            probabilidades = torch.softmax(
                logits_atuais,
                dim=-1
            )

            proximo_id = torch.argmax(
                logits_atuais,
                dim=-1,
                keepdim=True
            )

        # TOP 5 TOKENS MAIS PROVÁVEIS

        top_probs, top_ids = torch.topk(
            probabilidades,
            k=top_k_visualizacao
        )

        print("\nTop 5 probabilidades:")

        for posicao, (prob, token_id) in enumerate(
            zip(top_probs, top_ids),
            start=1
        ):

            token = tokenizer.decode([token_id.item()])

            print(
                f"{posicao}. {token!r:15} "
                f"ID: {token_id.item():6} "
                f"Probabilidade: {prob.item() * 100:.2f}%"
            )

        # TOKEN SELECIONADO

        token_id_escolhido = proximo_id.item()

        token_escolhido = tokenizer.decode(
            [token_id_escolhido]
        )

        prob_escolhida = probabilidades[
            token_id_escolhido
        ].item()

        print("\nToken escolhido:", repr(token_escolhido))
        print("ID:", token_id_escolhido)
        print(
            "Probabilidade:",
            f"{prob_escolhida * 100:.2f}%"
        )

        # VERIFICAR FIM DA RESPOSTA

        if token_id_escolhido in ids_fim:
            print("\nFim da resposta detectado.")
            break

        # ADICIONAR TOKEN À RESPOSTA

        ids_resposta.append(token_id_escolhido)

        # ADICIONAR TOKEN À SEQUÊNCIA COMPLETA

        ids_gerados = torch.cat(
            [
                ids_gerados,
                proximo_id.reshape(1, 1)
            ],
            dim=1
        )

        # EXIBIR RESPOSTA PARCIAL

        resposta_parcial = tokenizer.decode(
            ids_resposta,
            skip_special_tokens=True
        )

        print("Resposta parcial:", resposta_parcial)

        # EVITAR FORWARD PASS DESNECESSÁRIO
        # NO ÚLTIMO PASSO

        if passo == max_new_tokens - 1:
            break

        # PRÓXIMO FORWARD PASS

        with torch.inference_mode():
            saida = model(
                input_ids=ids_gerados,
                use_cache=False,
                return_dict=True
            )

        logits_atuais = saida.logits[0, -1, :]

    # RESULTADO FINAL

    print("\n--- TEXTO FINAL ---")

    resposta_final = tokenizer.decode(
        ids_resposta,
        skip_special_tokens=True
    )

    print(resposta_final)
