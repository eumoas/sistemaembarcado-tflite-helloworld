#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include "main_functions.h"

extern "C" void app_main(void) {
  setup();
  while (true) {
    loop();

    // O DHT22 so aceita uma nova leitura a cada 2 s.
    vTaskDelay(pdMS_TO_TICKS(2000));
  }
}
