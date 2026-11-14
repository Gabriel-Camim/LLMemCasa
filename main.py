import traceback
import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM
)

# ==========================================
# CONFIGURAÇÃO
# ==========================================

modelo_id = "Qwen/Qwen2.5-1.5B-Instruct"

temperatura = 0.7
max_new_tokens = 128
janela_contexto = 1024
top_k_visualizacao = 5

mostrar_inspecao = True
mostrar_top5 = True

system_msg = (
    "Você é um assistente de IA chamado LLMemCasa. "
    "Responda em português brasileiro, de maneira "
    "clara, objetiva e tecnicamente precisa. "
    "Considere as mensagens anteriores da conversa."
)

if janela_contexto <= max_new_tokens:
    raise ValueError(
        "A janela de contexto precisa ser maior "
        "que max_new_tokens."
    )

# ==========================================
# CARREGAMENTO
# ==========================================

print("Carregando tokenizador...")
tokenizer = AutoTokenizer.from_pretrained(modelo_id)

print("Carregando configuração...")
config = AutoConfig.from_pretrained(modelo_id)

print("Carregando pesos...")
model = AutoModelForCausalLM.from_pretrained(
    modelo_id,
    dtype=torch.float32
)

model.eval()

print("\n--- ARQUITETURA ---")
print("Dimensão oculta:", config.hidden_size)
print("Camadas:", config.num_hidden_layers)
print("Cabeças de atenção:", config.num_attention_heads)
print("Cabeças KV:", config.num_key_value_heads)
print("Vocabulário:", config.vocab_size)

print("\n--- HARDWARE ---")
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

print("\n--- CONTEXTO ---")
print("Janela:", janela_contexto)
print("Reserva de geração:", max_new_tokens)
print(
    "Orçamento do prompt:",
    janela_contexto - max_new_tokens
)

# ==========================================
# HISTÓRICO EM RAM
# ==========================================

historico = [
    {
        "role": "system",
        "content": system_msg
    }
]

# ==========================================
# FUNÇÕES DE CONTEXTO
# ==========================================

def tokenizar_mensagens(mensagens):

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

    return inputs["input_ids"]



def preparar_contexto(historico):

    # Espaço reservado para a geração.
    limite_prompt = janela_contexto - max_new_tokens

    if limite_prompt <= 0:
        raise ValueError(
            "A janela de contexto deve ser maior "
            "que max_new_tokens."
        )

    # A primeira mensagem é o system prompt.
    system = historico[0]

    # A última mensagem é a pergunta atual.
    atual = historico[-1]

    # Começamos com as mensagens obrigatórias.
    selecionadas = [system, atual]

    # Verificar se o mínimo obrigatório cabe.
    ids = tokenizar_mensagens(selecionadas)

    if ids.shape[1] > limite_prompt:
        raise ValueError(
            f"System + mensagem atual ocupam "
            f"{ids.shape[1]} tokens, mas o limite "
            f"é {limite_prompt}."
        )

    # Mensagens anteriores, sem system e sem
    # a pergunta atual.
    anteriores = historico[1:-1]

    # O histórico deve conter turnos completos:
    # user, assistant, user, assistant...
    if len(anteriores) % 2 != 0:
        raise ValueError(
            "Histórico inconsistente: existe "
            "um turno anterior incompleto."
        )

    # Agrupar em pares user + assistant.
    turnos = []

    for i in range(0, len(anteriores), 2):

        usuario = anteriores[i]
        assistente = anteriores[i + 1]

        if (
            usuario["role"] != "user"
            or assistente["role"] != "assistant"
        ):
            raise ValueError(
                "Ordem inválida no histórico."
            )

        turnos.append([usuario, assistente])

    # Inserir turnos do mais recente
    # para o mais antigo.
    turnos_selecionados = []

    for turno in reversed(turnos):

        candidato_turnos = [
            turno,
            *turnos_selecionados
        ]

        candidato = [system]

        for par in candidato_turnos:
            candidato.extend(par)

        candidato.append(atual)

        ids_teste = tokenizar_mensagens(
            candidato
        )

        if ids_teste.shape[1] <= limite_prompt:
            turnos_selecionados = candidato_turnos
        else:
            # Preserva a sequência recente e
            # descarta os turnos mais antigos.
            break

    # Reconstruir o contexto final.
    selecionadas = [system]

    for turno in turnos_selecionados:
        selecionadas.extend(turno)

    selecionadas.append(atual)

    ids = tokenizar_mensagens(selecionadas)

    print("\n--- GERENCIADOR DE CONTEXTO ---")
    print("Tokens utilizados:", ids.shape[1])
    print("Orçamento do prompt:", limite_prompt)
    print("Turnos anteriores disponíveis:", len(turnos))
    print("Turnos anteriores incluídos:", len(turnos_selecionados))
    print(
        "Turnos descartados:",
        len(turnos) - len(turnos_selecionados)
    )

    return ids, selecionadas

# ==========================================
# INSPEÇÃO DO FORWARD PASS
# ==========================================

def inspecionar_forward(ids, outputs):

    tokens_ids = ids[0].tolist()

    tokens = tokenizer.convert_ids_to_tokens(
        tokens_ids
    )

    print("\n--- TOKENIZAÇÃO ---")

    for token, token_id in zip(
        tokens,
        tokens_ids
    ):
        print(
            f"{token!r:25} -> ID: {token_id}"
        )

    with torch.inference_mode():

        embeddings = model.get_input_embeddings()(
            ids
        )

    print("\n--- EMBEDDINGS ---")
    print("Shape:", tuple(embeddings.shape))

    for i, token in enumerate(tokens):

        print(f"\nToken {i}: {token!r}")
        print(
            "Primeiros 10 valores:",
            embeddings[0, i, :10]
        )

    hidden_states = outputs.hidden_states

    print("\n--- FORWARD PASS ---")
    print(
        "Quantidade de estados:",
        len(hidden_states)
    )

    for i, estado in enumerate(hidden_states):

        vetor = estado[0, -1, :]

        print(f"\nEstado {i:02d}")
        print("Shape:", tuple(estado.shape))
        print(
            "Primeiros 5 valores:",
            vetor[:5].tolist()
        )

    inicial = hidden_states[0][0, -1, :]
    final = hidden_states[-1][0, -1, :]

    similaridade = (
        torch.nn.functional.cosine_similarity(
            inicial.unsqueeze(0).float(),
            final.unsqueeze(0).float()
        )
    )

    print("\n--- COMPARAÇÃO ---")
    print("Token:", repr(tokens[-1]))
    print("Inicial:", inicial[:10])
    print("Final:", final[:10])
    print(
        "Similaridade:",
        similaridade.item()
    )


# ==========================================
# GERAÇÃO COM KV CACHE
# ==========================================

def gerar_resposta(ids):

    tamanho_prompt = ids.shape[1]

    attention_mask = torch.ones_like(ids)

    print("\n--- PREFILL ---")
    print("Tokens de entrada:", tamanho_prompt)

    # Primeiro forward:
    # processa todo o prompt e constrói o cache.
    with torch.inference_mode():

        outputs = model(
            input_ids=ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=mostrar_inspecao,
            return_dict=True
        )

    if mostrar_inspecao:
        inspecionar_forward(ids, outputs)

    # Cache das 28 camadas Transformer.
    cache = outputs.past_key_values

    logits_atuais = outputs.logits[0, -1, :]

    ids_resposta = []

    im_end_id = tokenizer.convert_tokens_to_ids(
        "<|im_end|>"
    )

    ids_fim = {
        token_id
        for token_id in [
            tokenizer.eos_token_id,
            im_end_id
        ]
        if isinstance(token_id, int)
        and token_id >= 0
    }

    print("\n--- KV CACHE ---")
    print("Tipo:", type(cache).__name__)

    # Inspecionar tamanho do cache, quando
    # disponível na versão do Transformers.
    if hasattr(cache, "get_seq_length"):
        print(
            "Tokens armazenados:",
            cache.get_seq_length()
        )

    print("\n--- GERAÇÃO ---")

    for passo in range(max_new_tokens):

        # LOGITS -> TEMPERATURA -> SOFTMAX -> ARGMAX

        logits_ajustados = logits_atuais / temperatura

        probabilidades = torch.softmax(
            logits_ajustados,
            dim=-1
        )

        # Escolher o token de maior probabilidade
        proximo_id = torch.argmax(
            probabilidades,
            dim=-1,
            keepdim=True
        )

        # TOP 5

        if mostrar_top5:

            top_probs, top_ids = torch.topk(
                probabilidades,
                k=top_k_visualizacao
            )

            print(
                f"\n--- PASSO {passo + 1}: TOP 5 ---"
            )

            for posicao, (prob, token_id) in enumerate(
                zip(top_probs, top_ids),
                start=1
            ):

                token = tokenizer.decode(
                    [token_id.item()]
                )

                print(
                    f"{posicao}. {token!r:15} "
                    f"ID: {token_id.item():6} "
                    f"Prob: {prob.item() * 100:.2f}%"
                )

        token_id_escolhido = proximo_id.item()

        if token_id_escolhido in ids_fim:
            print("\nFim da resposta detectado.")
            break

        ids_resposta.append(
            token_id_escolhido
        )

        token_escolhido = tokenizer.decode(
            [token_id_escolhido]
        )

        print(
            f"Passo {passo + 1}: "
            f"{token_escolhido!r}"
        )

        # Último passo: não precisamos
        # executar outro forward.
        if passo == max_new_tokens - 1:
            break

        # A máscara deve acompanhar o
        # crescimento da sequência.
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (1, 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
            ],
            dim=1
        )

        # DECODIFICAÇÃO COM CACHE
        #
        # Enviamos SOMENTE o token novo.
        # O cache contém as Keys e Values
        # dos tokens anteriores.

        with torch.inference_mode():

            outputs = model(
                input_ids=proximo_id.reshape(1, 1),
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                return_dict=True
            )

        cache = outputs.past_key_values

        logits_atuais = outputs.logits[0, -1, :]

    resposta = tokenizer.decode(
        ids_resposta,
        skip_special_tokens=True
    )

    return resposta


# ==========================================
# LOOP INTERATIVO
# ==========================================

print("\n--- LLMemCasa iniciado ---")
print("Comandos: sair, limpar, historico")

while True:

    texto = input("\nVocê: ").strip()

    if texto.lower() == "sair":
        break

    if texto.lower() == "limpar":

        historico = [
            {
                "role": "system",
                "content": system_msg
            }
        ]

        print("Histórico limpo.")
        continue

    if texto.lower() == "historico":

        print("\n--- HISTÓRICO EM RAM ---")

        for mensagem in historico:
            print(
                f"{mensagem['role']}: "
                f"{mensagem['content']}"
            )

        continue

    if not texto:
        continue

    # Adicionar mensagem atual
    historico.append({
        "role": "user",
        "content": texto
    })

    try:

        ids, mensagens_contexto = (
            preparar_contexto(historico)
        )

        print("\n--- JANELA DE CONTEXTO ---")
        print(
            "Tokens do prompt:",
            ids.shape[1]
        )
        print(
            "Mensagens utilizadas:",
            len(mensagens_contexto)
        )
        print(
            "Mensagens totais:",
            len(historico)
        )

        resposta = gerar_resposta(ids)


    except Exception as erro:

        historico.pop()

        print("\n--- ERRO DETECTADO ---")

        print("Tipo:", type(erro).__name__)

        print("Representação:", repr(erro))

        traceback.print_exc()

        continue

    # Guardar resposta no histórico
    historico.append({
        "role": "assistant",
        "content": resposta
    })

    print("\n--- RESPOSTA FINAL ---")
    print("LLMemCasa:", resposta)
