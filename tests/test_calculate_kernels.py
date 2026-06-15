import calculate_kernels


def test_generate_settings_vector_width_activation():
    content = calculate_kernels.generate_settings_for_kernel(5)

    assert "#define KERNEL 5" in content
    assert "typedef float4 floatX;" in content
    assert "#define cl_init_vec(x) (float4)((float)(x))" in content
    assert "#define zeros cl_init_vec(0.0f)" in content


def test_generate_settings_scalar_width_fallback():
    content = calculate_kernels.generate_settings_for_kernel(1)

    assert "#define KERNEL 1" in content
    assert "typedef float floatX;" in content
    assert "#define zeros 0.0f" in content
    assert "cl_init_vec" not in content


def test_generate_settings_kernel_6_specific_defines():
    content = calculate_kernels.generate_settings_for_kernel(6)

    assert "#define TSM 64" in content
    assert "#define TSN 64" in content
    assert "#define TSK 8" in content
    assert "#define WPTM 8" in content
    assert "#define WPTN 4" in content


def test_generate_settings_kernel_11_defines():
    content = calculate_kernels.generate_settings_for_kernel(11)

    assert "#define THREADSX 8" in content
    assert "#define THREADSY 8" in content
    assert "#define RX 8" in content
    assert "#define RY 4" in content


def test_generate_settings_math_macros_presence():
    content = calculate_kernels.generate_settings_for_kernel(1)

    assert "#define MIN(a,b)" in content
    assert "#define MAX(a,b)" in content
    assert "#define CEIL_DIV(x,y)" in content


def test_generate_settings_invalid_kernel_fallback():
    content = calculate_kernels.generate_settings_for_kernel(999)

    assert "#define KERNEL 999" in content
    # По стандарту: TS = 16, WIDTH = 1
    assert "#define TS 16" in content
    assert "typedef float floatX;" in content
