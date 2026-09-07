
#include "fgds_utils.h"

int main(int argc, char **argv){   
    BenchmarkOpts opts;

    if (!parseOpts(argc, argv, opts)){
        exit(EXIT_FAILURE);
    }

    switch (opts.xfer_mode) {
        case GPUD_MODE_FGDS:
            run_fgds(opts);
            break;
        case GPUD_MODE_GDS:
            run_gds(opts);
            break;
        case GPUD_MODE_POSIX:
            run_posix(opts);
            break;
        default:
            pr_error("Unsupport xfer mode");
            printHelp(argv[0]);
            exit(EXIT_FAILURE);
    }


    return 0;
}