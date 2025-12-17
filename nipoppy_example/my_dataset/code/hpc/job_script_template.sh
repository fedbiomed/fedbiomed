#!/bin/bash

{#-
# This is a template for generating a job script that will be run on an HPC
# cluster. It is written using the Jinja templating language (see
# https://jinja.palletsprojects.com for more information).

# All variables starting with the "NIPOPPY_" prefix are set internally by
# Nipoppy and cannot be changed. Other (optional) variables can be defined in a
# pipeline's HPC config file (i.e., hpc.json). Additional variables can also be
# defined in the HPC config file for further customization.

# Lines surrounded by { # and # } (without spaces) are Jinja comments and will
# not be included in the final job script.
#}

{#-
# ----------------------------
# JOB SCHEDULER CONFIGURATIONS
# ----------------------------
# Below sections are for the Slurm and SGE job schedulers respectively.
# Depending on the value of the --hpc argument, only one of these will be used.
# Existing lines should not be modified unless you know what you are doing.
# New lines can be added to hardcode extra settings that are to be constant for
# every HPC job (no matter which pipeline).
#}
{%- if NIPOPPY_HPC == 'slurm' %}
{% set NIPOPPY_ARRAY_VAR = 'SLURM_ARRAY_TASK_ID' %}
# ===== Slurm configs =====
#SBATCH --job-name={{ NIPOPPY_JOB_NAME }}
#SBATCH --output={{ NIPOPPY_DPATH_LOGS }}/%x-%A_%a.out
#SBATCH --array=1-{{ NIPOPPY_COMMANDS | length }}
{%- if ARRAY_CONCURRENCY_LIMIT -%}
%{{ ARRAY_CONCURRENCY_LIMIT }}
{%- endif %}
{% if TIME -%}
#SBATCH --time={{ TIME }}
{%- endif -%}
{% if MEMORY %}
#SBATCH --mem={{ MEMORY }}
{%- endif -%}
{% if CORES %}
#SBATCH --cpus-per-task={{ CORES }}
{%- endif -%}
{% if ACCOUNT %}
#SBATCH --account={{ ACCOUNT }}
{%- endif %}
{% if PARTITION %}
#SBATCH --partition={{ PARTITION }}
{%- endif %}

{%- elif NIPOPPY_HPC == 'sge' %}
{% set NIPOPPY_ARRAY_VAR = 'SGE_TASK_ID' %}
# ===== SGE configs =====
#$ -N {{ NIPOPPY_JOB_NAME }}
#$ -o {{ NIPOPPY_DPATH_LOGS }}/$JOB_NAME_$JOB_ID_$TASK_ID.out
#$ -j y
#$ -t 1-{{ NIPOPPY_COMMANDS | length }}
{% if ARRAY_CONCURRENCY_LIMIT -%}
#$ -tc {{ ARRAY_CONCURRENCY_LIMIT }}
{%- endif -%}
{% if TIME %}
#$ -l h_rt={{ TIME }}
{%- endif -%}
{% if MEMORY %}
#$ -l h_vmem={{ MEMORY }}
{%- endif -%}
{% if ACCOUNT %}
#$ -q {{ ACCOUNT }}
{%- endif %}
{% endif %}

# for custom scripting
DPATH_ROOT="{{ NIPOPPY_DPATH_ROOT }}"
PIPELINE_NAME="{{ NIPOPPY_PIPELINE_NAME }}"
PIPELINE_VERSION="{{ NIPOPPY_PIPELINE_VERSION }}"
PIPELINE_STEP="{{ NIPOPPY_PIPELINE_STEP }}"
PARTICIPANT_IDS=({% for participant_id in NIPOPPY_PARTICIPANT_IDS %} "{{ participant_id }}"{% endfor %} )
SESSION_IDS=({% for session_id in NIPOPPY_SESSION_IDS %} "{{ session_id }}"{% endfor %} )
{#
# -------------------
# START OF JOB SCRIPT
# -------------------
# Below lines should not be modified unless you know what you are doing.
#}
{% if NIPOPPY_HPC_PREAMBLE_STRINGS -%}
# HPC_PREAMBLE from global config file
{% for NIPOPPY_HPC_PREAMBLE_STRING in NIPOPPY_HPC_PREAMBLE_STRINGS -%}
{{ NIPOPPY_HPC_PREAMBLE_STRING }}
{% endfor %}
{%- endif %}
# Nipoppy-generated list of commands to be run in job array
COMMANDS=( \
{% for command in NIPOPPY_COMMANDS -%}
    "{{ command }}" \
{% endfor -%}
)

# get command from list
# note that COMMANDS is zero-indexed (bash array)
# but the job array is one-indexed for compatibility with SGE
I_JOB=$(({{NIPOPPY_ARRAY_VAR}}-1))
COMMAND=${COMMANDS[$I_JOB]}

# for custom scripting
PARTICIPANT_ID=${PARTICIPANT_IDS[$I_JOB]}
SESSION_ID=${SESSION_IDS[$I_JOB]}

# print/run command
echo $COMMAND
eval $COMMAND
