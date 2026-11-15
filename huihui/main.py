"""Chat local com Huihui em GGUF via llama.cpp, apenas em CPU."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import enable_progress_bars

os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
enable_progress_bars()


ROOT = Path(__file__).resolve().parent
REPO = 'huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF'
FILENAME = 'Huihui-Qwen3.8-27B-abliterated-UD-IQ2_S.gguf'
MODEL_DIR = ROOT / 'modelos'
SERVER = ROOT.parent / "llama_cpp_cpu" / "llama-server.exe"
HOST = '127.0.0.1'
PORT = 8089
N_CTX = 1024
N_BATCH = 128
N_PREDICT = 512
TEMPERATURE = 0.0  # greedy: sem multinomial
SYSTEM = ('Você é LLMemCasa. Responda em português brasileiro de forma correta, '
          'objetiva e natural. Considere as mensagens anteriores.')
LOG_PATH = ROOT / 'llama_server.log'
URL = f'http://{HOST}:{PORT}'


def get_model() -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / FILENAME

    if path.is_file():
        print(f"Modelo encontrado: {path}")
        print(f"Tamanho: {path.stat().st_size / 1e9:.2f} GB")
        return path

    print(f"Baixando {FILENAME} (~8,42 GB)...", flush=True)

    downloaded = hf_hub_download(
        repo_id=REPO,
        filename=FILENAME,
        local_dir=str(MODEL_DIR),
    )

    result = Path(downloaded)
    print(f"Download concluído: {result}", flush=True)
    return result


def start_server(model_path: Path):
    if not SERVER.is_file():
        raise FileNotFoundError(
            f'llama-server.exe não encontrado: {SERVER}. '
            'Veja huihui/README.md para instalar a versão CPU do llama.cpp.'
        )

    command = [
        str(SERVER),
        '-m', str(model_path),
        '--host', HOST,
        '--port', str(PORT),
        '-c', str(N_CTX),
        '-b', str(N_BATCH),
        '-ngl', '0',
        '-np', '1',
        '-t', '6',
        '--no-ui'
    ]

    print('Iniciando llama.cpp em modo CPU...')
    print('Comando:', ' '.join(command))

    log = LOG_PATH.open(
        'w',
        encoding='utf-8',
        errors='replace'
    )

    proc = subprocess.Popen(
        command,
        cwd=str(SERVER.parent),
        stdout=log,
        stderr=subprocess.STDOUT
    )

    return proc, log


def log_text():
    try:
        return LOG_PATH.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return ''


def wait_ready(proc, timeout=300):
    deadline = time.monotonic() + timeout

    print('Carregando modelo na RAM. Aguarde...')

    while time.monotonic() < deadline:

        if proc.poll() is not None:
            raise RuntimeError(
                f'Servidor encerrou (código {proc.returncode}).\n'
                + log_text()[-5000:]
            )

        try:
            r = requests.get(
                URL + '/health',
                timeout=2
            )

            if r.ok and r.json().get('status') == 'ok':
                print('Modelo carregado com sucesso!')
                print('Inferência configurada para CPU.')
                return

        except (requests.RequestException, ValueError):
            pass

        time.sleep(1)

    raise TimeoutError(
        'Servidor não ficou pronto. Veja ' + str(LOG_PATH)
    )


def make_messages(history):
    # Retém os turnos mais recentes; usa margem conservadora por caracteres.
    # A contagem exata depende do tokenizer e é validada pelo servidor.
    budget = N_CTX - N_PREDICT
    if budget <= 0:
        raise ValueError('N_CTX deve superar N_PREDICT.')
    # Aproximação conservadora; não substitui a tokenização exata.
    estimate = lambda msg: max(8, len(msg['content'].encode('utf-8')) // 2 + 12)
    base = [history[0], history[-1]]
    if sum(map(estimate, base)) > budget:
        raise ValueError('System + pergunta atual excedem o orçamento estimado.')
    prior = history[1:-1]
    if len(prior) % 2:
        raise ValueError('Histórico anterior incompleto.')
    turns = [prior[i:i + 2] for i in range(0, len(prior), 2)]
    kept = []
    used = sum(map(estimate, base))
    for turn in reversed(turns):
        cost = sum(map(estimate, turn))
        if used + cost > budget:
            break
        kept.insert(0, turn)
        used += cost
    messages = [history[0]]
    for turn in kept:
        messages.extend(turn)
    messages.append(history[-1])
    print(f'Contexto: {len(kept)}/{len(turns)} turnos anteriores; '
          f'orçamento estimado {used}/{budget} tokens.')
    return messages


def generate(messages):
    payload = {
        'model': 'local',
        'messages': messages,
        'temperature': TEMPERATURE,
        'max_tokens': N_PREDICT,
        'stream': True,
    }

    answer_parts = []
    reasoning_parts = []

    print('\nLLMemCasa: ', end='', flush=True)

    with requests.post(
        URL + '/v1/chat/completions',
        json=payload,
        stream=True,
        timeout=(10, 1200)
    ) as r:

        r.raise_for_status()

        for raw_line in r.iter_lines():
            if not raw_line:
                continue

            line = raw_line.decode('utf-8')
            if not line:
                continue

            if not line.startswith('data: '):
                continue

            raw = line[6:].strip()

            if raw == '[DONE]':
                break

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            choices = data.get('choices') or []
            if not choices:
                continue

            delta = choices[0].get('delta') or {}

            # Tokens de raciocínio, se enviados separadamente
            reasoning = (
                delta.get('reasoning_content')
                or delta.get('reasoning')
                or ''
            )

            # Tokens de raciocínio
            if reasoning:
                if not reasoning_parts:
                    print('\n[REASONING]\n', flush=True)

                reasoning_parts.append(reasoning)
                print(reasoning, end='', flush=True)

            # Tokens da resposta final
            content = delta.get('content') or ''

            if content:
                if not answer_parts:
                    print('\n[RESPOSTA]\n', flush=True)

                answer_parts.append(content)
                print(content, end='', flush=True)

    print()

    answer = ''.join(answer_parts)
    reasoning_text = ''.join(reasoning_parts)

    if reasoning_text:
        print(
            f'\n[Raciocínio recebido: '
            f'{len(reasoning_text)} caracteres]'
        )

    return answer


def main():
    model_path = get_model()
    proc, logfile = start_server(model_path)
    history = [{'role': 'system', 'content': SYSTEM}]
    try:
        wait_ready(proc)
        print('LLMemCasa Huihui iniciado. Comandos: sair, limpar, historico')
        while True:
            prompt = input('\nVocê: ').strip()
            if prompt.lower() == 'sair':
                break
            if prompt.lower() == 'limpar':
                history = [history[0]]
                print('Histórico em RAM limpo.')
                continue
            if prompt.lower() == 'historico':
                for msg in history:
                    print(f"{msg['role']}: {msg['content']}")
                continue
            if not prompt:
                continue
            history.append({'role': 'user', 'content': prompt})
            try:
                messages = make_messages(history)
                answer = generate(messages)
            except Exception as exc:
                history.pop()
                print('Erro:', repr(exc))
                continue
            history.append({'role': 'assistant', 'content': answer})
    finally:
        print('Encerrando servidor local...')
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        logfile.close()


if __name__ == '__main__':
    main()
