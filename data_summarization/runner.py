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


def call_script(args):
    global cnt
    exp, method, coreset_size, seed = args
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
    sp.call([sys.executable, '{}.py'.format(exp), '--seed', str(seed), '--method', method,
             '--coreset_size', str(coreset_size)], env=crt_env)

def krr_cifar_call_script(args):
    method, seed, = args
    crt_env = os.environ.copy()
    crt_env['OMP_NUM_THREADS'] = '1'
    crt_env['MKL_NUM_THREADS'] = '1'
    sp.call([sys.executable, 'krr_cifar.py', '--seed', str(seed),
             '--method', method], env=crt_env)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runner')
    parser.add_argument('--exp', default='cnn_mnist', choices=['cnn_mnist', 'resnet_cifar', 'krr_cifar'])
    args = parser.parse_args()
    exp = args.exp
    pool = Pool(NR_PROCESSES)

    seeds = range(5)
    methods = ['uniform', 'coreset']
    if exp == 'cnn_mnist':
        coreset_sizes = range(75, 251, 25)
        args = list(itertools.product([exp], methods, coreset_sizes, seeds))
        random.shuffle(args)
        pool.map(call_script, args)
        pool.close()
        pool.join()
    elif exp == 'resnet_cifar':
        coreset_sizes = [210]
        args = list(itertools.product([exp], methods, coreset_sizes, seeds))
        random.shuffle(args)
        pool.map(call_script, args)
        pool.close()
        pool.join()
    elif exp == 'krr_cifar':
        pool = Pool(5)
        methods = ['uniform', 'uniform_weights_opt', 'coreset']
        coreset_args = list(itertools.product(methods, seeds))
        pool.map(krr_cifar_call_script, coreset_args)
        pool.close()
        pool.join()
    else:
        raise Exception('Unknown experiment')
