import subprocess
import os
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

def generate_settings_for_kernel(kernel_num):
    cfg = KERNEL_BASE_CONFIGS.get(kernel_num, {"TS": 16, "WIDTH": 1})

    ts = cfg.get("TS", 32)
    wpt = cfg.get("WPT", 8)
    width = cfg.get("WIDTH", 4)
    tsdk = cfg.get("TSDK", 16)
    tsm = cfg.get("TSM", 128)
    tsn = cfg.get("TSN", 128)
    tsk = cfg.get("TSK", 16)
    wptm = cfg.get("WPTM", 8)
    wptn = cfg.get("WPTN", 8)

    rts = ts // wpt
    lpt = (tsdk * wpt) // ts
    rtsm = tsm // wptm
    rtsn = tsn // wptn
    lpta = (tsk * wptm * wptn) // tsn
    lptb = (tsk * wptm * wptn) // tsm

    vector_fix = ""
    if width > 1:
        vector_fix = f"""
#ifdef __OPENCL_VERSION__
  #undef inline
  #define inline __attribute__((always_inline))
  #define cl_init_vec(x) (float{width})((float)(x))
  #define zeros cl_init_vec(0.0f)
#endif
"""
    else:
        vector_fix = """
#ifdef __OPENCL_VERSION__
  #define zeros 0.0f
#endif
"""

    content = f"""// AUTO-GENERATED FOR KERNEL {kernel_num}
#define KERNEL {kernel_num}

// Constants for kernels 1 -- 5
#define TS {ts}

// Constants for kernels 3, 5
#define WPT {wpt}
#define RTS {rts}

// Constants for kernels 4, 7 -- 10
#define WIDTH {width}

// Constants for kernel 5
#define TSDK {tsdk}
#define LPT {lpt}

// Constants for kernels 6 -- 10
#define TSM {tsm}
#define TSN {tsn}
#define TSK {tsk}
#define WPTM {wptm}
#define WPTN {wptn}
#define RTSM {rtsm}
#define RTSN {rtsn}
#define LPTA {lpta}
#define LPTB {lptb}

// Constants for kernel 11
#define THREADSX 8
#define THREADSY 8
#define RX 8
#define RY 4
#define RK 4

// Supporting kernels
#define TRANSPOSEX 16
#define TRANSPOSEY 16
#define PADDINGX 16
#define PADDINGY 16

// Macros
#define MIN(a,b) (((a) > (b)) ? (b) : (a))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define CEIL_DIV(x,y) (((x) + (y) - 1) / (y))
#define MOD2(x,y) ((x) % (y))
#define DIV2(x,y) ((x) / (y))

#ifdef __OPENCL_VERSION__
  typedef float{"" if width==1 else width} floatX;
#else
  typedef float floatX;
#endif

{vector_fix}
"""
    return content

def compile_and_run_one(kernel_num, warmup, measure):
    settings_text = generate_settings_for_kernel(kernel_num)

    Path("src").mkdir(exist_ok=True)
    Path("obj").mkdir(exist_ok=True)
    Path("bin").mkdir(exist_ok=True)

    with open("src/settings.h", "w") as f:
        f.write(settings_text)

    compile_flags = [
        "-c", "-O3", "-Wall",
        "-I/System/Library/Frameworks/OpenCL.framework/Headers",
        "-include", "src/settings.h"
    ]

    subprocess.run(["g++"] + compile_flags + ["src/main.cpp", "-o", "obj/main.o"], check=True)
    subprocess.run(["g++"] + compile_flags + ["src/clGEMM.cpp", "-o", "obj/clGEMM.o"], check=True)

    dummy_obj = Path("obj/dummy_libclblas.o")
    if not dummy_obj.exists():
        with open("dummy.cpp", "w") as f:
            f.write('void libclblas(float*, float*, float*, int, int, int, int) {}\n')
        subprocess.run(["g++", "-c", "dummy.cpp", "-o", str(dummy_obj)], check=True)
        os.unlink("dummy.cpp")

    link_cmd = [
        "g++", "-O3", "-Wall",
        "obj/main.o", "obj/clGEMM.o", str(dummy_obj),
        "-framework", "OpenCL", "-o", "bin/myGEMM"
    ]
    subprocess.run(link_cmd, check=True)

    log_dir = Path(f"results/kernel_{kernel_num}")
    log_dir.mkdir(parents=True, exist_ok=True)

    for _ in range(warmup):
        subprocess.run(["./bin/myGEMM"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for i in range(1, measure + 1):
        log_file = log_dir / f"run_{i}.log"
        with open(log_file, "w") as f:
            subprocess.run(["./bin/myGEMM"], stdout=f, stderr=subprocess.STDOUT)
        print(f" Kernel {kernel_num}: run {i}/{measure} is done", end="\r")
    print(f"\n Kernel {kernel_num} finished")

def main():
    warmup = 15
    measure = 50

    for kernel in range(2):
        print(f"Start kernel {kernel} ===")
        try:
            compile_and_run_one(kernel, warmup, measure)
        except subprocess.CalledProcessError:
            print("Warning")
        time.sleep(1)

    print("All kernels finished.")

if __name__ == "__main__":
    main()
