// Detector de ocupacao de sala com TensorFlow Lite Micro.
//
// A cada 2 s: le o DHT22 (temperatura e umidade) e o LDR (luz), normaliza as
// tres grandezas como no treino, quantiza para int8, roda o modelo e acende o
// LED quando a probabilidade de "sala ocupada" passa do limiar.

#include <math.h>

#include "driver/gpio.h"
#include "esp_timer.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "dht22.h"
#include "ldr.h"
#include "main_functions.h"
#include "model.h"
#include "model_params.h"

namespace {
constexpr gpio_num_t kPinoDht = GPIO_NUM_15;
constexpr adc_channel_t kCanalLdr = ADC_CHANNEL_6;  // GPIO34
constexpr gpio_num_t kPinoLed = GPIO_NUM_2;

// Faixa observada no dataset. Fora dela o modelo esta extrapolando.
constexpr float kTempMin = 19.0f, kTempMax = 24.5f;
constexpr float kUmidMin = 16.5f, kUmidMax = 39.5f;
constexpr float kLuzMax = 1700.0f;

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 2000;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// Converte float -> int8 arredondando e limitando a faixa [-128, 127].
int8_t quantizar(float valor) {
  int q = static_cast<int>(lroundf(valor / kEntradaEscala)) + kEntradaZero;
  if (q < -128) q = -128;
  if (q > 127) q = 127;
  return static_cast<int8_t>(q);
}
}  // namespace

void setup() {
  gpio_reset_pin(kPinoLed);
  gpio_set_direction(kPinoLed, GPIO_MODE_OUTPUT);
  if (ldr_iniciar(kCanalLdr) != ESP_OK) {
    MicroPrintf("Falha ao iniciar o ADC do LDR");
    return;
  }

  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Modelo com schema %d, esperado %d", model->version(),
                TFLITE_SCHEMA_VERSION);
    return;
  }

  // So as operacoes que o modelo usa: 3 camadas densas (ReLU fundida) e a
  // sigmoid de saida (LOGISTIC).
  static tflite::MicroMutableOpResolver<2> resolver;
  if (resolver.AddFullyConnected() != kTfLiteOk ||
      resolver.AddLogistic() != kTfLiteOk) {
    return;
  }

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    MicroPrintf("AllocateTensors() falhou");
    interpreter = nullptr;
    return;
  }
  input = interpreter->input(0);
  output = interpreter->output(0);

  MicroPrintf("Detector de ocupacao pronto: modelo %d bytes, arena usada %d de %d bytes",
              g_model_len, static_cast<int>(interpreter->arena_used_bytes()),
              kTensorArenaSize);
}

void loop() {
  if (interpreter == nullptr) {
    return;
  }

  float temperatura, umidade, luz;
  int adc_bruto;
  esp_err_t erro = dht22_ler(kPinoDht, &temperatura, &umidade);
  if (erro != ESP_OK) {
    MicroPrintf("Falha ao ler o DHT22: %s", esp_err_to_name(erro));
    return;
  }
  if (ldr_ler(&adc_bruto, &luz) != ESP_OK) {
    MicroPrintf("Falha ao ler o LDR");
    return;
  }

  // Mesma normalizacao do treino: (valor - media) / desvio.
  const float leituras[kNumEntradas] = {temperatura, umidade, luz};
  for (int i = 0; i < kNumEntradas; i++) {
    input->data.int8[i] = quantizar((leituras[i] - kMedia[i]) / kDesvio[i]);
  }

  int64_t inicio = esp_timer_get_time();
  if (interpreter->Invoke() != kTfLiteOk) {
    MicroPrintf("Invoke falhou");
    return;
  }
  int tempo_us = static_cast<int>(esp_timer_get_time() - inicio);

  float prob = (output->data.int8[0] - kSaidaZero) * kSaidaEscala;
  bool ocupada = prob >= kLimiarOcupada;
  gpio_set_level(kPinoLed, ocupada);

  MicroPrintf("T=%.1f C  UR=%.1f %%  Luz=%.0f lux (ADC %d) -> P(ocupada)=%.2f => %s  [%d us]",
              static_cast<double>(temperatura), static_cast<double>(umidade),
              static_cast<double>(luz), adc_bruto, static_cast<double>(prob),
              ocupada ? "OCUPADA" : "VAZIA", tempo_us);
  if (temperatura < kTempMin || temperatura > kTempMax ||
      umidade < kUmidMin || umidade > kUmidMax || luz > kLuzMax) {
    MicroPrintf("  aviso: leitura fora da faixa do dataset, o modelo esta extrapolando");
  }
}
