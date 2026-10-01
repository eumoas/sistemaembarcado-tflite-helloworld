#pragma once

#include "esp_adc/adc_oneshot.h"
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

// Configura o ADC1 no canal informado (GPIO34 = ADC_CHANNEL_6 no ESP32).
esp_err_t ldr_iniciar(adc_channel_t canal);

// Le o modulo fotoresistor do Wokwi. Devolve o valor bruto do ADC (0..4095)
// e a iluminacao estimada em lux.
esp_err_t ldr_ler(int *bruto, float *lux);

#ifdef __cplusplus
}
#endif
