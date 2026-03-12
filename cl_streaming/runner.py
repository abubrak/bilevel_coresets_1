import subprocess as sp
from multiprocessing.dummy import Pool
import itertools
import sys, os
import argparse
import random

# 导入 GPU 配置检测（适配 Colab 单 GPU 环境）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpu_config import detect_gpu_config, get_cuda_visible_devices

# 自动检测 GPU 配置
gpu_config = detect_gpu_config()
NR_GPUS = gpu_config['NR_GPUS']
NR_PROCESSES = gpu_config['NR_PROCESSES']

print(f"检测到 GPU 数量: {NR_GPUS}, 进程数: {NR_PROCESSES}")

cnt = -1

def cl_call_script(args):
    global cnt
    seed, dataset, method, buffer_size, beta = args
    crt_env = os.environ.copy()
    crt_env['OMP_NUM_THREADS'] = '1'
    crt_env['MKL_NUM_THREADS'] = '1'
    crt_env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    cnt += 1
    gpu_id = get_cuda_visible_devices(cnt, NR_GPUS)
    if gpu_id != '':
        crt_env['CUDA_VISIBLE_DEVICES'] = gpu_id
    else:
        # CPU 模式，不设置 CUDA_VISIBLE_DEVICES
        pass
    print(args)
    sp.call([sys.executable, 'cl.py', '--seed', str(seed), '--dataset', dataset, '--method', method,
             '--buffer_size', str(buffer_size), '--beta', str(beta)], env=crt_env)


def streaming_call_script(args):
    global cnt
    seed, dataset, method, buffer_size, beta, nr_slots = args
    crt_env = os.environ.copy()
    crt_env['OMP_NUM_THREADS'] = '1'
    crt_env['MKL_NUM_THREADS'] = '1'
    crt_env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    cnt += 1
    gpu_id = get_cuda_visible_devices(cnt, NR_GPUS)
    if gpu_id != '':
        crt_env['CUDA_VISIBLE_DEVICES'] = gpu_id
    else:
        # CPU 模式，不设置 CUDA_VISIBLE_DEVICES
        pass
    print(args)
    sp.call([sys.executable, 'streaming.py', '--seed', str(seed), '--dataset', dataset, '--method', method,
             '--buffer_size', str(buffer_size), '--beta', str(beta), '--nr_slots', str(nr_slots)], env=crt_env)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runner')
    parser.add_argument('--exp', default='cl', choices=['cl', 'streaming', 'imbalanced_streaming'])
    args = parser.parse_args()
    exp = args.exp
    pool = Pool(NR_PROCESSES)

    seeds = range(5)
    betas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    if exp == 'cl':
        buffer_size = [100]
        datasets = ['permmnist', 'splitmnist']
        methods = ['uniform', 'coreset',
                   'kmeans_features', 'kcenter_features', 'kmeans_grads',
                   'kmeans_embedding', 'kcenter_embedding', 'kcenter_grads',
                   'entropy', 'hardest', 'frcl', 'icarl', 'grad_matching']
        args = list(itertools.product(seeds, datasets, methods, buffer_size, betas))
        random.shuffle(args)
        pool.map(cl_call_script, args)
        pool.close()
        pool.join()
    elif exp == 'streaming':
        buffer_size = [100]
        nr_slots = [10]
        datasets = ['permmnist', 'splitmnist']
        methods = ['reservoir', 'coreset']
        args = list(itertools.product(seeds, datasets, methods, buffer_size, betas, nr_slots))
        random.shuffle(args)
        pool.map(streaming_call_script, args)
        pool.close()
        pool.join()
    elif exp == 'imbalanced_streaming':
        buffer_size = [100]
        nr_slots = [1]
        datasets = ['splitmnistimbalanced']
        methods = ['reservoir', 'cbrs', 'coreset']
        args = list(itertools.product(seeds, datasets, methods, buffer_size, betas, nr_slots))
        random.shuffle(args)
        pool.map(streaming_call_script, args)
        pool.close()
        pool.join()
    else:
        raise Exception('Unknown experiment')
