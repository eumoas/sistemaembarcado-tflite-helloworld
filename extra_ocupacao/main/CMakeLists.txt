idf_component_register(
    SRCS main.cc main_functions.cc model.cc dht22.c ldr.c
    PRIV_REQUIRES esp_driver_gpio esp_adc esp_timer
    INCLUDE_DIRS "")
