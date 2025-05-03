project(
  'Neural Synthesis',
  'c',
  version: '0.1',
  default_options: ['warning_level=3', 'c_args=-Wall -O3 -march=native'],
)

cc = meson.get_compiler('c')

raylib = cc.find_library('raylib', required: true)
#pipewire_dep = dependency('libpipewire-0.3')
math_lib = cc.find_library('m', required: true)

src_files = files(
  'main.c',
)

executable(
  'main',
  src_files,
  dependencies: [raylib, math_lib],
)
