// Gerado por treino/treinar_modelo.py. Nao edite a mao.
// Constantes de normalizacao calculadas no conjunto de treino e
// parametros de quantizacao do modelo int8.
#pragma once

constexpr int kNumEntradas = 3;
// Ordem das entradas: temperatura (C), umidade (%), luz (lux)
constexpr float kMedia[kNumEntradas] = {20.619085f, 25.731508f, 119.519371f};
constexpr float kDesvio[kNumEntradas] = {1.016854f, 5.530871f, 194.743835f};
constexpr float kEntradaEscala = 0.0351035707f;
constexpr int kEntradaZero = -82;
constexpr float kSaidaEscala = 0.00390625f;
constexpr int kSaidaZero = -128;
constexpr float kLimiarOcupada = 0.50f;
