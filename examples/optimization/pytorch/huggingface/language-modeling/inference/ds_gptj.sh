export OMP_NUM_THREADS=104
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so

export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"

#numactl -C 0-103 python run_ds_gptj.py --precision bf16 --max-new-tokens 32 &
#numactl -m 0 -C 0-51 python run_ds_gptj.py --precision bf16 --max-new-tokens 32 &
#numactl -m 1 -C 52-103 python run_ds_gptj.py --precision bf16 --max-new-tokens 32 &
deepspeed --num_gpus 2 run_ds_gptj.py --precision bf16 --max-new-tokens 32
