import subprocess
import time
from pathlib import Path

KERNEL_BASE_CONFIGS = {
    1: {"TS": 16, "WIDTH": 1},
    2: {"TS": 16, "WIDTH": 1},
    3: {"TS": 32, "WPT": 4, "WIDTH": 1},
    4: {"TS": 16, "WIDTH": 2},
    5: {"TS": 32, "WPT": 4, "TSDK": 32, "WIDTH": 4},
    6: {"TSM": 64, "TSN": 64, "TSK": 8, "WPTM": 8, "WPTN": 4, "WIDTH": 4},
    7: {"TSM": 64, "TSN": 64, "TSK": 8, "WPTM": 8, "WPTN": 4, "WIDTH": 2},
    8: {"TSM": 64, "TSN": 64, "TSK": 8, "WPTM": 8, "WPTN": 4, "WIDTH": 4},
    9: {"TSM": 64, "TSN": 64, "TSK": 8, "WPTM": 8, "WPTN": 4, "WIDTH": 4},
    10: {"TSM": 32, "TSN": 32, "TSK": 8, "WPTM": 8, "WPTN": 4, "WIDTH": 2},
    11: {"THREADSX": 8, "THREADSY": 8, "RX": 8, "RY": 4, "WIDTH": 1},
}


def fix_opencl_includes():
    for file_path in Path("src").glob("*.cpp"):
        if not file_path.exists():
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if "<CL/cl.h>" in content:
            content = content.replace("<CL/cl.h>", "<OpenCL/opencl.h>")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)


def generate_settings_for_kernel(kernel_num):
    cfg = KERNEL_BASE_CONFIGS.get(kernel_num, {"TS": 16, "WIDTH": 1})
    width = cfg.get("WIDTH", 1)

    lines = [
        f"#define KERNEL {kernel_num}",
        "",
        "#define MIN(a,b) (((a) > (b)) ? (b) : (a))",
        "#define MAX(a,b) (((a) > (b)) ? (a) : (b))",
        "#define CEIL_DIV(x,y) (((x) + (y) - 1) / (y))",
        "#define MOD2(x,y) ((x) % (y))",
        "#define DIV2(x,y) ((x) / (y))",
        "",
        "#define TRANSPOSEX 16",
        "#define TRANSPOSEY 16",
        "#define PADDINGX 16",
        "#define PADDINGY 16",
        "",
    ]

    for key, value in cfg.items():
        lines.append(f"#define {key} {value}")

    lines.append("")

    if "TS" in cfg and "WPT" in cfg:
        lines.append("#define RTS (TS/WPT)")
    if "TSDK" in cfg and "WPT" in cfg and "TS" in cfg:
        lines.append("#define LPT ((TSDK*WPT)/(TS))")
    if "TSM" in cfg and "WPTM" in cfg:
        lines.append("#define RTSM (TSM/WPTM)")
    if "TSN" in cfg and "WPTN" in cfg:
        lines.append("#define RTSN (TSN/WPTN)")
    if "TSK" in cfg and "WPTM" in cfg and "WPTN" in cfg and "TSN" in cfg:
        lines.append("#define LPTA ((TSK*WPTM*WPTN)/(TSN))")
    if "TSK" in cfg and "WPTM" in cfg and "WPTN" in cfg and "TSM" in cfg:
        lines.append("#define LPTB ((TSK*WPTM*WPTN)/(TSM))")

    if kernel_num == 11:
        lines.extend(["", "#define RK (RY)"])

    lines.extend(["", "#ifdef __OPENCL_VERSION__"])
    if width > 1:
        lines.extend(
            [
                f"  typedef float{width} floatX;",
                "  #undef inline",
                "  #define inline __attribute__((always_inline))",
                f"  #define cl_init_vec(x) (float{width})((float)(x))",
                "  #define zeros cl_init_vec(0.0f)",
            ]
        )
    else:
        lines.extend(["  typedef float floatX;", "  #define zeros 0.0f"])
    lines.extend(["#else", "  typedef float floatX;", "#endif", ""])

    return "\n".join(lines)


def compile_and_run_one(kernel_num, warmup, measure):
    settings_text = generate_settings_for_kernel(kernel_num)

    Path("src").mkdir(exist_ok=True)
    Path("obj").mkdir(exist_ok=True)
    Path("bin").mkdir(exist_ok=True)

    with open("src/settings.h", "w") as f:
        f.write(settings_text)

    fix_opencl_includes()

    compile_flags = [
        "-c",
        "-O3",
        "-Wall",
        "-DCL_SILENCE_DEPRECATION",
        "-DCL_TARGET_OPENCL_VERSION=120",
        "-include",
        "src/settings.h",
    ]

    subprocess.run(
        ["g++"] + compile_flags + ["src/main.cpp", "-o", "obj/main.o"], check=True
    )
    subprocess.run(
        ["g++"] + compile_flags + ["src/clGEMM.cpp", "-o", "obj/clGEMM.o"], check=True
    )

    subprocess.run(
        ["g++", "-c", "-x", "c++", "-", "-o", "obj/dummy_clblas.o"],
        input=b"void libclblas(float*, float*, float*, int, int, int, int) {}\n",
        check=True,
    )

    link_cmd = [
        "g++",
        "-O3",
        "-Wall",
        "obj/main.o",
        "obj/clGEMM.o",
        "obj/dummy_clblas.o",
        "-framework",
        "OpenCL",
        "-o",
        "bin/myGEMM",
    ]
    subprocess.run(link_cmd, check=True)

    log_dir = Path(f"results/kernel_{kernel_num}")
    log_dir.mkdir(parents=True, exist_ok=True)

    for _ in range(warmup):
        subprocess.run(
            ["./bin/myGEMM"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    for i in range(1, measure + 1):
        log_file = log_dir / f"run_{i}.log"
        with open(log_file, "w") as f:
            subprocess.run(["./bin/myGEMM"], stdout=f, stderr=subprocess.STDOUT)
        print(f" Kernel {kernel_num}: run {i}/{measure} is done", end="\r")
    print(f"\n Kernel {kernel_num} finished")


def main():
    warmup = 15
    measure = 50

    for kernel in range(1, 12):
        print(f"Start kernel {kernel}")
        try:
            compile_and_run_one(kernel, warmup, measure)
        except subprocess.CalledProcessError:
            print(f"Error compiling or running kernel {kernel}")
        time.sleep(3)

    print("All kernels finished.")


if __name__ == "__main__":
    main()
