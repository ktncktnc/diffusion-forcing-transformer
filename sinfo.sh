#!/bin/bash

# Fetch node information
node_info=$(scontrol show nodes)

# Process each node
echo "$node_info" | awk '
BEGIN {
    COL_WIDTH_NODE = 10
    COL_WIDTH_GPU = 3
    COL_WIDTH_CPU = 7
    COL_WIDTH_MEM = 9
    COL_WIDTH_STATE = 11

    FS="\n";
    RS="\n\n";
    printf "%"COL_WIDTH_NODE"s %"COL_WIDTH_GPU"s %"COL_WIDTH_CPU"s %"COL_WIDTH_MEM"s %"COL_WIDTH_STATE"s\n", "Node", "GPU", "CPU", "Mem (G)", "State";
}
{
    node_name = "";
    gpu_alloc = "";
    cpu_alloc = "";
    mem_alloc = "";
    gpu_total = "";
    cpu_total = "";
    mem_total = "";
    state = "";

    for (i = 1; i <= NF; i++) {
        if ($i ~ /^NodeName=/) {
            split($i, node_parts, " ");
            for (j in node_parts) {
                if (node_parts[j] ~ /^NodeName=/) {
                    split(node_parts[j], name_parts, "=");
                    node_name = name_parts[2];
                }
            }
        }
        else if ($i ~ /CfgTRES=/) {
            sub("   CfgTRES=", "", $i);
            split($i, tres_parts, ",");
            for (j in tres_parts) {
                if (tres_parts[j] ~ /^gres\/gpu=/) {
                    split(tres_parts[j], gpu_parts, "=");
                    gpu_total = gpu_parts[2];
                }
                else if (tres_parts[j] ~ /^cpu=/) {
                    split(tres_parts[j], cpu_parts, "=");
                    cpu_total = cpu_parts[2];
                }
                else if (tres_parts[j] ~ /^mem=/) {
                    split(tres_parts[j], mem_parts, "=");
                    if (mem_parts[2] ~ /M$/) {
                        gsub(/[A-Z]/, "", mem_parts[2]);
                        mem_total = mem_parts[2] / 1024;  # Convert from MB to GB
                    } else if (mem_parts[2] ~ /G$/) {
                        gsub(/[A-Z]/, "", mem_parts[2]);
                        mem_total = mem_parts[2];  # Already in GB
                    }
                    mem_total = int(mem_total + 0.5)
                }
            }
        }
        else if ($i ~ /AllocTRES=/) {
            sub("   AllocTRES=", "", $i);
            split($i, tres_alloc_parts, ",");
            for (j in tres_alloc_parts) {
                if (tres_alloc_parts[j] ~ /^gres\/gpu=/) {
                    split(tres_alloc_parts[j], alloc_gpu, "=");
                    gpu_alloc = alloc_gpu[2];
                }
                else if (tres_alloc_parts[j] ~ /^cpu=/) {
                    split(tres_alloc_parts[j], alloc_cpu, "=");
                    cpu_alloc = alloc_cpu[2];
                }
                else if (tres_alloc_parts[j] ~ /^mem=/) {
                    split(tres_alloc_parts[j], mem_alloc_info, "=");
                    if (mem_alloc_info[2] ~ /M$/) {
                        gsub(/[A-Z]/, "", mem_alloc_info[2]);
                        mem_alloc = mem_alloc_info[2] / 1024;  # Convert from MB to GB
                    } else if (mem_alloc_info[2] ~ /G$/) {
                        gsub(/[A-Z]/, "", mem_alloc_info[2]);
                        mem_alloc = mem_alloc_info[2];  # Already in GB
                    }
                    mem_alloc = int(mem_alloc + 0.5)
                }
            }
        }
        else if ($i ~ /State=/) {
            split($i, state_line_parts, " ");
            for (j in state_line_parts) {
                if (state_line_parts[j] ~ /^State=/) {
                    split(state_line_parts[j], state_parts, "=");
                    state = tolower(state_parts[2]);
                    state = substr(state, 1, COL_WIDTH_STATE)
                }
            }
        }
    }

    if (node_name != "") {
        printf "%"COL_WIDTH_NODE"s %"COL_WIDTH_GPU"s %"COL_WIDTH_CPU"s %"COL_WIDTH_MEM"s %"COL_WIDTH_STATE"s\n", node_name, gpu_alloc "/" gpu_total, cpu_alloc "/" cpu_total, mem_alloc "/" mem_total, state;
    }
}'