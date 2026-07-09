# Vitis HLS script for radar 2D CNN accelerator.
# Usage from hls folder:
#   RADAR_CNN_HLS_TESTBENCH=tb_real_samples.cpp vitis_hls -f run_hls.tcl
#     -> real-sample C simulation only
#   vitis_hls -f run_hls.tcl
#     -> default testbench, synthesis, and IP export

open_project radar_cnn_hls_prj
set_top radar_cnn_accel

add_files radar_cnn_hls.cpp
add_files radar_cnn_hls.h
add_files weights.h
if {[info exists ::env(RADAR_CNN_HLS_TESTBENCH)]} {
    set tb_file $::env(RADAR_CNN_HLS_TESTBENCH)
} else {
    set tb_file "tb_radar_cnn_hls.cpp"
}
add_files -tb $tb_file

if {[info exists ::env(RADAR_CNN_HLS_MODE)]} {
    set run_mode $::env(RADAR_CNN_HLS_MODE)
} elseif {[info exists ::env(RADAR_CNN_HLS_TESTBENCH)]} {
    set run_mode "csim"
} else {
    set run_mode "synth"
}

open_solution "solution1" -flow_target vivado

# KV260/K26 SOM part. Adjust if your board flow uses a platform instead.
set_part {xck26-sfvc784-2LV-c}
create_clock -period 5.0 -name default

if {$run_mode eq "csim" || $run_mode eq "all"} {
    if {$tb_file eq "tb_real_samples.cpp"} {
        set sample_file [file normalize "../reports/fixed_point_validation/input_samples.txt"]
        csim_design -argv $sample_file
    } else {
        csim_design
    }
}

if {$run_mode eq "synth" || $run_mode eq "all"} {
    csynth_design
    # cosim_design is optional and can be slow.
    # cosim_design
    export_design -format ip_catalog
}

exit
