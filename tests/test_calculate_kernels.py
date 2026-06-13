import calculate_kernels


def test_generate_settings_math_formulas():
    # TS=32, WPT=4, TSDK=32, WIDTH=4
    content = calculate_kernels.generate_settings_for_kernel(5)

    # rts = 32 // 4 = 8, lpt = (32 * 4) // 32 = 4
    assert "#define RTS 8" in content
    assert "#define LPT 4" in content
    assert "typedef float4 floatX;" in content
    assert "#define cl_init_vec(x) (float4)((float)(x))" in content


def test_generate_settings_width_1_scalar():
    content = calculate_kernels.generate_settings_for_kernel(1)

    assert "typedef float floatX;" in content
    assert "#define zeros 0.0f" in content
    assert "cl_init_vec" not in content


def test_kernel_configs_exist():
    for k in range(1, 12):
        assert k in calculate_kernels.KERNEL_BASE_CONFIGS
        assert isinstance(calculate_kernels.KERNEL_BASE_CONFIGS[k], dict)
