cmake_minimum_required(VERSION 3.18)
project(transpose LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# ── target ──────────────────────────────────────────────────────────────────
add_executable(transpose_test
    transpose.cu
)

# ── CUDA arch — adjust to your GPU ──────────────────────────────────────────
# Common values: 70 (V100), 75 (T4), 80 (A100), 86 (RTX 30xx), 89 (RTX 40xx)
set_target_properties(transpose_test PROPERTIES
    CUDA_ARCHITECTURES "80;86;89"
)

target_compile_options(transpose_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -lineinfo
        -Xptxas=-v        # print register/smem usage
    >
)

target_link_libraries(transpose_test PRIVATE m)
