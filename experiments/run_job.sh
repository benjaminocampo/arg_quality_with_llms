#!/bin/bash

# set -e → exit immediately if any command fails.
# set -u → exit if you try to use an undefined variable.
# set -o pipefail → exit if any command in a pipeline fails.
# This makes the script safer.
set -euo pipefail


# Checks that the user passed three argument (the task script file,
# the experiment config file, and the LLM config file).
# Otherwise, prints usage and exits.
if [ $# -lt 4]; then
  echo "Usage: $0 <llm_config.yaml> <dim_config.yaml> <prompt_config.yaml> <run_config.yaml>"
  exit 1
fi

# If Slurm is available, also ensure logs exists
if command -v sbatch &>/dev/null; then
  mkdir -p logs
fi

ENV_NAME="arg_quality_with_llms_venv"

LLM_CONFIG=$1
DIM_CONFIG=$2
PROMPT_CONFIG=$3
RUN_CONFIG=$4

# Takes config file
# Small inline Python helper to extract YAML values
get_yaml() {
python3 <<EOF
import sys, yaml
config_file = "$1"
with open(config_file) as f:
    data = yaml.safe_load(f)
value = ${2}
if value is None:
    sys.exit(1)
print(value)
EOF
}

# Extract top-level values
LLM_NAME=$(get_yaml "$LLM_CONFIG" "data['name']")
PARTITION=$(get_yaml "$LLM_CONFIG" "data['slurm_params']['partition']")
TIME=$(get_yaml "$LLM_CONFIG" "data['slurm_params']['time']")
GRES=$(get_yaml "$LLM_CONFIG" "data['slurm_params']['gres']")
MEM=$(get_yaml "$LLM_CONFIG" "data['slurm_params']['mem']")
PROMPT_TYPE=$(get_yaml "$PROMPT_CONFIG" "data['type']")
DIM_NAME=$(get_yaml "$DIM_CONFIG" "data['name']")
RUN_ID=$(get_yaml "$RUN_CONFIG" "data['run_id']")

JOB_NAME="${LLM_NAME}_${DIM_NAME}_${PROMPT_TYPE}_${RUN_ID}"

# Create sbatch script
SBATCH_SCRIPT=$(mktemp)

cat <<EOF > "$SBATCH_SCRIPT"
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME}
#SBATCH --gres=${GRES}
#SBATCH --mem=${MEM}
#SBATCH --output=./logs/${JOB_NAME}.out
#SBATCH --error=./logs/${JOB_NAME}.err

source ../${ENV_NAME}/bin/activate
python predict.py \\
    llm=${LLM_NAME} \\
    dim=${DIM_NAME} \\
    prompt=${PROMPT_TYPE} \\
    run=${RUN_ID}
EOF

# If Slurm is not available, just print the job script
if ! command -v sbatch &>/dev/null; then
  echo "Script running locally (no batched)"
  bash "$SBATCH_SCRIPT"
else
  sbatch "$SBATCH_SCRIPT"
  echo "Submitted job with config $LLM_CONFIG, $DIM_CONFIG, $PROMPT_CONFIG, and $RUN_CONFIG"
fi
