cmake_minimum_required(VERSION 3.5)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(hello_world)

# O Wokwi nao simula as otimizacoes do ESP-NN. A aula indica comentar a linha
# "target_compile_options(${COMPONENT_LIB} PRIVATE -DESP_NN)" no CMakeLists.txt
# do esp-tflite-micro. Como managed_components/ e baixado novamente pelo
# gerenciador de componentes, removemos a mesma flag aqui, depois que o
# componente foi registrado. Os kernels em esp_nn/*.cc usam "#if ESP_NN" e,
# sem essa definicao, executam a implementacao de referencia do TFLite Micro.
idf_component_get_property(tflite_lib espressif__esp-tflite-micro COMPONENT_LIB)
get_target_property(tflite_opts ${tflite_lib} COMPILE_OPTIONS)
list(REMOVE_ITEM tflite_opts -DESP_NN)
set_property(TARGET ${tflite_lib} PROPERTY COMPILE_OPTIONS ${tflite_opts})
