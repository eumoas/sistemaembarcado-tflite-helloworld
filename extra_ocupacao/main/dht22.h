#pragma once

#include "driver/gpio.h"
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

// Le temperatura (graus C) e umidade relativa (%) de um DHT22 ligado em `pino`.
// O sensor precisa de pelo menos 2 s entre duas leituras.
esp_err_t dht22_ler(gpio_num_t pino, float *temperatura, float *umidade);

#ifdef __cplusplus
}
#endif
