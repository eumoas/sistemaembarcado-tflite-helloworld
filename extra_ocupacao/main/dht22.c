// Driver minimo do DHT22 (protocolo de um fio, por bit-banging).
//
// 1. O ESP32 segura a linha em 0 por ~1,2 ms e solta (pull-up leva a 1).
// 2. O sensor responde com ~80 us em 0 e ~80 us em 1.
// 3. Vem 40 bits. Cada bit comeca com ~50 us em 0; depois a linha fica em 1
//    por ~27 us (bit 0) ou ~70 us (bit 1).
// 4. Bytes: umidade (2), temperatura (2), checksum (1).

#include "dht22.h"

#include "esp_rom_sys.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"

static portMUX_TYPE s_trava = portMUX_INITIALIZER_UNLOCKED;

// Espera a linha sair de `nivel`. Retorna quanto tempo (us) ela ficou nele,
// ou -1 se passar de `limite_us`.
static int duracao_nivel(gpio_num_t pino, int nivel, int limite_us)
{
    int64_t inicio = esp_timer_get_time();
    while (gpio_get_level(pino) == nivel) {
        if (esp_timer_get_time() - inicio > limite_us) {
            return -1;
        }
    }
    return (int)(esp_timer_get_time() - inicio);
}

esp_err_t dht22_ler(gpio_num_t pino, float *temperatura, float *umidade)
{
    uint8_t dados[5] = {0};

    // Sinal de inicio.
    gpio_set_direction(pino, GPIO_MODE_OUTPUT_OD);
    gpio_set_pull_mode(pino, GPIO_PULLUP_ONLY);
    gpio_set_level(pino, 0);
    esp_rom_delay_us(1200);
    gpio_set_level(pino, 1);
    esp_rom_delay_us(30);
    gpio_set_direction(pino, GPIO_MODE_INPUT);

    // A leitura dos bits depende de microssegundos: nenhuma interrupcao
    // pode atrasar o laco enquanto os 40 bits chegam.
    esp_err_t erro = ESP_OK;
    taskENTER_CRITICAL(&s_trava);
    if (duracao_nivel(pino, 1, 100) < 0 ||   // sensor ainda nao baixou a linha
        duracao_nivel(pino, 0, 100) < 0 ||   // resposta em 0
        duracao_nivel(pino, 1, 100) < 0) {   // resposta em 1
        erro = ESP_ERR_TIMEOUT;
    }
    for (int i = 0; i < 40 && erro == ESP_OK; i++) {
        if (duracao_nivel(pino, 0, 80) < 0) {
            erro = ESP_ERR_TIMEOUT;
            break;
        }
        int alto = duracao_nivel(pino, 1, 100);
        if (alto < 0) {
            erro = ESP_ERR_TIMEOUT;
            break;
        }
        dados[i / 8] <<= 1;
        if (alto > 40) {
            dados[i / 8] |= 1;
        }
    }
    taskEXIT_CRITICAL(&s_trava);

    if (erro != ESP_OK) {
        return erro;
    }
    if ((uint8_t)(dados[0] + dados[1] + dados[2] + dados[3]) != dados[4]) {
        return ESP_ERR_INVALID_CRC;
    }

    *umidade = ((dados[0] << 8) | dados[1]) / 10.0f;
    // Bit mais alto da temperatura e o sinal (temperatura negativa).
    int bruto = ((dados[2] & 0x7F) << 8) | dados[3];
    *temperatura = (dados[2] & 0x80 ? -bruto : bruto) / 10.0f;
    return ESP_OK;
}
