"""Treina o classificador de ocupacao de sala e gera o modelo para o ESP32.

Fluxo (o mesmo do Hello World da aula):
  1. carregar o dataset UCI Occupancy Detection;
  2. normalizar as entradas;
  3. treinar uma rede neural pequena no TensorFlow/Keras;
  4. converter para TensorFlow Lite com quantizacao int8;
  5. avaliar o modelo float e o int8 nos dois conjuntos de teste;
  6. gerar main/model.cc (xxd -i) e main/model_params.h para o firmware.

Uso:  python3 treinar_modelo.py
"""

import json
import os
import random
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

AQUI = os.path.dirname(os.path.abspath(__file__))
DADOS = os.path.join(AQUI, "dados")
RESULTADOS = os.path.join(AQUI, "resultados")
MAIN = os.path.join(AQUI, "..", "main")

# Apenas as grandezas que os sensores simulados no Wokwi conseguem medir:
# DHT22 -> temperatura e umidade; LDR (fotoresistor) -> luz em lux.
ENTRADAS = ["Temperature", "Humidity", "Light"]
ALVO = "Occupancy"
SEMENTE = 42
# Peso da regularizacao L2. Sem ela a rede "decora" combinacoes de temperatura
# e umidade dos dias de treino e erra mais nos dias de teste (ver README).
L2 = 0.01


def fixar_sementes():
    """Deixa o treino reproduzivel: mesma semente, mesmos pesos iniciais."""
    os.environ["PYTHONHASHSEED"] = str(SEMENTE)
    random.seed(SEMENTE)
    np.random.seed(SEMENTE)
    tf.random.set_seed(SEMENTE)


def carregar(nome):
    df = pd.read_csv(os.path.join(DADOS, nome))
    x = df[ENTRADAS].to_numpy(dtype=np.float32)
    y = df[ALVO].to_numpy(dtype=np.float32)
    return x, y


def metricas(y_real, prob, limiar=0.5):
    y_pred = (prob >= limiar).astype(int)
    y_real = y_real.astype(int)
    vp = int(np.sum((y_pred == 1) & (y_real == 1)))
    vn = int(np.sum((y_pred == 0) & (y_real == 0)))
    fp = int(np.sum((y_pred == 1) & (y_real == 0)))
    fn = int(np.sum((y_pred == 0) & (y_real == 1)))
    precisao = vp / (vp + fp) if vp + fp else 0.0
    revocacao = vp / (vp + fn) if vp + fn else 0.0
    f1 = 2 * precisao * revocacao / (precisao + revocacao) if precisao + revocacao else 0.0
    return {
        "acuracia": round((vp + vn) / len(y_real), 4),
        "precisao": round(precisao, 4),
        "revocacao": round(revocacao, 4),
        "f1": round(f1, 4),
        "matriz_confusao": {"VP": vp, "VN": vn, "FP": fp, "FN": fn},
    }


def criar_modelo(l2):
    # 3 entradas -> 8 -> 8 -> 1. A saida sigmoid e a probabilidade de "ocupada".
    reg = tf.keras.regularizers.l2(l2) if l2 else None
    return tf.keras.Sequential([
        tf.keras.Input(shape=(len(ENTRADAS),)),
        tf.keras.layers.Dense(8, activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Dense(8, activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])


def treinar(x, y, x_val, y_val, l2):
    modelo = criar_modelo(l2)
    modelo.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                   loss="binary_crossentropy", metrics=["accuracy"])
    hist = modelo.fit(x, y, validation_data=(x_val, y_val),
                      epochs=60, batch_size=64, verbose=0,
                      callbacks=[tf.keras.callbacks.EarlyStopping(
                          monitor="val_loss", patience=10, restore_best_weights=True)])
    return modelo, hist


def converter_int8(modelo, x_rep):
    """Quantizacao pos-treino: pesos, ativacoes, entrada e saida em int8.

    O conversor precisa de um "dataset representativo" para descobrir a faixa
    de valores de cada tensor e calcular escala (scale) e ponto zero (zero_point).
    Usamos todo o treino: com poucas amostras os valores altos de luz ficam de
    fora da faixa e seriam cortados (saturados) no int8.
    """
    def representativo():
        for linha in x_rep:
            yield [linha.reshape(1, -1)]

    conv = tf.lite.TFLiteConverter.from_keras_model(modelo)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representativo
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def prever_tflite(modelo_tflite, x):
    """Roda o .tflite no interpretador do PC, do mesmo jeito que o ESP32 fara."""
    interp = tf.lite.Interpreter(model_content=modelo_tflite)
    interp.allocate_tensors()
    ent = interp.get_input_details()[0]
    sai = interp.get_output_details()[0]
    e_esc, e_zero = ent["quantization"]
    s_esc, s_zero = sai["quantization"]
    saida = np.empty(len(x), dtype=np.float32)
    for i, linha in enumerate(x):
        q = np.clip(np.round(linha / e_esc + e_zero), -128, 127).astype(np.int8)
        interp.set_tensor(ent["index"], q.reshape(1, -1))
        interp.invoke()
        saida[i] = (interp.get_tensor(sai["index"])[0, 0].astype(np.float32) - s_zero) * s_esc
    return saida, (e_esc, e_zero), (s_esc, s_zero)


def gerar_model_cc(caminho_tflite):
    """Converte o .tflite em array C com xxd -i, como mostrado na aula."""
    # "xxd -i < arquivo" imprime so os bytes; o nome do array e escrito aqui
    # (versoes antigas do xxd nao tem a opcao -n para escolher o nome).
    with open(caminho_tflite, "rb") as f:
        bytes_hex = subprocess.run(["xxd", "-i"], stdin=f, check=True,
                                   capture_output=True, text=True).stdout
    tamanho = os.path.getsize(caminho_tflite)
    bruto = ("alignas(8) const unsigned char g_model[] = {\n" + bytes_hex
             + "};\nconst int g_model_len = " + str(tamanho) + ";\n")
    cabecalho = (
        "// Gerado por treino/treinar_modelo.py com: xxd -i < modelo_ocupacao_int8.tflite\n"
        "// Nao edite a mao: rode o script de treino novamente.\n\n"
        '#include "model.h"\n\n'
    )
    with open(os.path.join(MAIN, "model.cc"), "w") as f:
        f.write(cabecalho + bruto)


def gerar_params_h(media, desvio, ent_q, sai_q, limiar):
    linhas = [
        "// Gerado por treino/treinar_modelo.py. Nao edite a mao.",
        "// Constantes de normalizacao calculadas no conjunto de treino e",
        "// parametros de quantizacao do modelo int8.",
        "#pragma once",
        "",
        f"constexpr int kNumEntradas = {len(ENTRADAS)};",
        "// Ordem das entradas: temperatura (C), umidade (%), luz (lux)",
        "constexpr float kMedia[kNumEntradas] = {"
        + ", ".join(f"{v:.6f}f" for v in media) + "};",
        "constexpr float kDesvio[kNumEntradas] = {"
        + ", ".join(f"{v:.6f}f" for v in desvio) + "};",
        f"constexpr float kEntradaEscala = {ent_q[0]:.9g}f;",
        f"constexpr int kEntradaZero = {int(ent_q[1])};",
        f"constexpr float kSaidaEscala = {sai_q[0]:.9g}f;",
        f"constexpr int kSaidaZero = {int(sai_q[1])};",
        f"constexpr float kLimiarOcupada = {limiar:.2f}f;",
        "",
    ]
    with open(os.path.join(MAIN, "model_params.h"), "w") as f:
        f.write("\n".join(linhas))


def grafico_treino(hist):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(hist.history["loss"], label="treino")
    ax[0].plot(hist.history["val_loss"], label="validacao")
    ax[0].set_title("Perda (binary cross-entropy)")
    ax[0].set_xlabel("epoca")
    ax[0].legend()
    ax[1].plot(hist.history["accuracy"], label="treino")
    ax[1].plot(hist.history["val_accuracy"], label="validacao")
    ax[1].set_title("Acuracia")
    ax[1].set_xlabel("epoca")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTADOS, "curvas_treino.png"), dpi=120)
    plt.close(fig)


def grafico_luz(x, y, limiar_luz):
    """Mostra por que a luz separa tao bem as duas classes."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(x[y == 0, 2], bins=60, alpha=0.6, label="vazia")
    ax.hist(x[y == 1, 2], bins=60, alpha=0.6, label="ocupada")
    ax.axvline(limiar_luz, color="k", linestyle="--", label=f"limiar {limiar_luz:.0f} lux")
    ax.set_yscale("log")
    ax.set_xlabel("luz (lux)")
    ax.set_ylabel("amostras (escala log)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTADOS, "distribuicao_luz.png"), dpi=120)
    plt.close(fig)


def main():
    fixar_sementes()
    os.makedirs(RESULTADOS, exist_ok=True)

    # 1. Dados. datatest.txt foi coletado com a porta fechada e
    #    datatest2.txt com a porta aberta (Candanedo & Feldheim, 2016).
    x_tr, y_tr = carregar("datatraining.txt")
    testes = {
        "teste1_porta_fechada": carregar("datatest.txt"),
        "teste2_porta_aberta": carregar("datatest2.txt"),
    }

    # 2. Normalizacao com media e desvio APENAS do treino (evita vazamento).
    media = x_tr.mean(axis=0)
    desvio = x_tr.std(axis=0)

    def norm(x):
        return (x - media) / desvio

    # 3. Treino. 20% do treino (embaralhado) fica para validacao.
    idx = np.random.permutation(len(x_tr))
    corte = int(0.8 * len(idx))
    tr, va = idx[:corte], idx[corte:]
    modelo, hist = treinar(norm(x_tr[tr]), y_tr[tr], norm(x_tr[va]), y_tr[va], L2)
    grafico_treino(hist)
    # Mesma rede sem regularizacao, so para comparacao no relatorio.
    tf.random.set_seed(SEMENTE)
    sem_reg, _ = treinar(norm(x_tr[tr]), y_tr[tr], norm(x_tr[va]), y_tr[va], 0.0)

    # 4. Conversao para int8.
    modelo_tflite = converter_int8(modelo, norm(x_tr))
    caminho_tflite = os.path.join(RESULTADOS, "modelo_ocupacao_int8.tflite")
    with open(caminho_tflite, "wb") as f:
        f.write(modelo_tflite)

    # Linha de base: um simples "luz > limiar", com o limiar escolhido no treino.
    candidatos = np.linspace(0, 800, 801)
    acertos = [np.mean((x_tr[:, 2] > c) == y_tr) for c in candidatos]
    limiar_luz = float(candidatos[int(np.argmax(acertos))])
    grafico_luz(x_tr, y_tr, limiar_luz)

    # 5. Avaliacao.
    relatorio = {
        "dataset": "UCI Occupancy Detection (Candanedo & Feldheim, 2016)",
        "entradas": ENTRADAS,
        "amostras_treino": int(len(x_tr)),
        "proporcao_ocupada_treino": round(float(y_tr.mean()), 4),
        "regularizacao_l2": L2,
        "epocas_treinadas": len(hist.history["loss"]),
        "parametros_modelo": int(modelo.count_params()),
        "tamanho_tflite_int8_bytes": len(modelo_tflite),
        "normalizacao": {"media": media.round(4).tolist(), "desvio": desvio.round(4).tolist()},
        "linha_de_base_luz": {"limiar_lux": limiar_luz},
        "resultados": {},
    }
    ent_q = sai_q = None
    for nome, (x_te, y_te) in testes.items():
        prob_float = modelo.predict(norm(x_te), verbose=0).ravel()
        prob_int8, ent_q, sai_q = prever_tflite(modelo_tflite, norm(x_te))
        relatorio["resultados"][nome] = {
            "amostras": int(len(x_te)),
            "modelo_float32": metricas(y_te, prob_float),
            "modelo_int8": metricas(y_te, prob_int8),
            "modelo_float32_sem_l2": metricas(
                y_te, sem_reg.predict(norm(x_te), verbose=0).ravel()),
            "linha_de_base_luz": metricas(y_te, (x_te[:, 2] > limiar_luz).astype(float)),
            "concordancia_float_int8": round(float(np.mean(
                (prob_float >= 0.5) == (prob_int8 >= 0.5))), 4),
        }

    # Exemplos de referencia para conferir no Wokwi.
    exemplos = np.array([
        [21.0, 25.0, 0.0],     # noite, sala escura
        [21.5, 27.0, 450.0],   # luz de escritorio
        [23.0, 30.0, 600.0],   # escritorio cheio
        [20.5, 20.0, 50.0],    # pouca luz
    ], dtype=np.float32)
    prob_ex, _, _ = prever_tflite(modelo_tflite, norm(exemplos))
    relatorio["exemplos_referencia"] = [
        {"temperatura": float(e[0]), "umidade": float(e[1]), "luz": float(e[2]),
         "prob_ocupada_int8": round(float(p), 3)}
        for e, p in zip(exemplos, prob_ex)
    ]

    with open(os.path.join(RESULTADOS, "metricas.json"), "w") as f:
        json.dump(relatorio, f, indent=2, ensure_ascii=False)

    # 6. Arquivos para o firmware.
    gerar_model_cc(caminho_tflite)
    gerar_params_h(media, desvio, ent_q, sai_q, 0.5)
    print(json.dumps(relatorio, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
