// Leitura do modulo fotoresistor (LDR) do Wokwi pelo ADC do ESP32.
//
// No modulo, o LDR fica em serie com um resistor fixo de 10 kohm e a saida AO
// e a tensao sobre o LDR:  AO = VCC * R_ldr / (R_ldr + 10k).
// Como o ADC mede AO em relacao a VCC, a resistencia sai da proporcao:
//   R_ldr = 10k * bruto / (4095 - bruto)
// e a iluminacao vem da curva do LDR usada pelo Wokwi (GAMMA = 0,7 e
// RL10 = 50 kohm, resistencia a 10 lux):
//   lux = (RL10 * 10^GAMMA / R_ldr)^(1/GAMMA)

#include "ldr.h"

#include <math.h>

static const float kResistorFixo = 10000.0f;
static const float kGamma = 0.7f;
static const float kRl10 = 50000.0f;

static adc_oneshot_unit_handle_t s_adc;
static adc_channel_t s_canal;

esp_err_t ldr_iniciar(adc_channel_t canal)
{
    adc_oneshot_unit_init_cfg_t unidade = {.unit_id = ADC_UNIT_1};
    esp_err_t erro = adc_oneshot_new_unit(&unidade, &s_adc);
    if (erro != ESP_OK) {
        return erro;
    }
    // Atenuacao de 12 dB: o ADC aceita a faixa inteira de 0 a 3,3 V.
    adc_oneshot_chan_cfg_t config = {
        .atten = ADC_ATTEN_DB_12,
        .bitwidth = ADC_BITWIDTH_12,
    };
    s_canal = canal;
    return adc_oneshot_config_channel(s_adc, canal, &config);
}

esp_err_t ldr_ler(int *bruto, float *lux)
{
    esp_err_t erro = adc_oneshot_read(s_adc, s_canal, bruto);
    if (erro != ESP_OK) {
        return erro;
    }
    if (*bruto >= 4095) {      // LDR "infinito": escuro total
        *lux = 0.0f;
        return ESP_OK;
    }
    int b = *bruto < 1 ? 1 : *bruto;
    float resistencia = kResistorFixo * b / (4095.0f - b);
    *lux = powf(kRl10 * powf(10.0f, kGamma) / resistencia, 1.0f / kGamma);
    return ESP_OK;
}
