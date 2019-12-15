# Default libralies
import itertools
import time
from datetime import datetime
import sys
from copy import deepcopy
import multiprocessing as mp
import multiprocessing.pool
import functools
# Thrid party
import math
import numpy as np


def grid_samples(params_conf, condition=None):
    """Grid Sampling

    Args:
        params_conf (list(dict) or dict): sample configuration
        contion: function object, optional
            Filter of sampling

    Returns:
        samples (list(dict))
    """
    domains = list()
    names = []
    for conf in params_conf:
        names.append(conf['name'])
        domain = conf["domain"]
        type_ = conf.get("type", "grid")
        if type_ in ["integer", "continuous"]:
            if conf.get("scale", None) == 'log':
                domain = np.logspace(np.log10(domain[0]), np.log10(domain[1]), conf["num_grid"])
            else:
                domain = np.linspace(domain[0], domain[1], conf["num_grid"])
            if type_ == "integer":
                domain = domain.astype(int)
        elif type_ == "fixed":
            domain = [domain, ]
        domains.append(list(domain))

    patterns = itertools.product(*domains)
    samples = list()
    for params_val in patterns:
        params_dict = dict()
        for name, val in zip(names, params_val):
            params_dict[name] = val
        if condition is None or condition(params_dict):
            samples.append(params_dict)
    return samples


def expand_call(kwargs):
    """Execute function from dictionary input"""
    func = kwargs['func']
    del kwargs['func']
    out = func(**kwargs)
    return out


def report_progress(job_idx, num_jobs, time0, task):
    """Report progress to system output"""
    msg = [float(job_idx) / num_jobs, (time.time() - time0) / 60.]
    msg.append(msg[1] * (1 / msg[0] - 1))
    time_stamp = str(datetime.fromtimestamp(time.time()))
    msg_ = time_stamp + ' ' + str(
        round(msg[0] * 100, 2)) + '% ' + task + ' done after ' + \
           str(round(msg[1], 2)) + ' minutes. Remaining ' + str(
        round(msg[2], 2)) + ' minutes.'
    if job_idx < num_jobs:
        sys.stderr.write(msg_ + '\r')
    else:
        sys.stderr.write(msg_ + '\n')


def process_jobs(jobs, task=None, num_threads=24, use_thread=False):
    """Execute parallelized jobs

    Parameters
    ----------
    jobs: list(dict)
        Each element contains `function` and its parameters
    task: str, optional
        The name of task. If not specified, function name is used
    num_threads, int, optional
        The number of threads for parallelization. if not feeded, use the maximum
        number of process
    use_thread: bool, defulat False
        If True, use multi process. If False, use multi thread.
        Use True, if the multiprocessing exceeds the memory limit

    Returns
    -------
    List: each element is results of each part
    """
    if task is None:
        task = jobs[0]['func'].__name__
    if num_threads is None:
        num_threads = mp.cpu_count()
    out = []
    if num_threads > 1:
        if use_thread:
            pool = mp.pool.ThreadPool(processes=num_threads)
        else:
            pool = mp.Pool(processes=num_threads)
        outputs = pool.imap_unordered(expand_call, jobs)
        time0 = time.time()
        # Execute programs here
        for i, out_ in enumerate(outputs, 1):
            out.append(out_)
            report_progress(i, len(jobs), time0, task)
        pool.close()
        pool.join()
    else:
        for job in jobs:
            job = deepcopy(job)
            func = job['func']
            del job['func']
            out_ = func(**job)
            out.append(out_)
    return out


def multi_grid_execution(params_conf, task_func, default_params=None,
                         condition=None, num_threads=None):
    """Execute parallelized jobs based on configuration

    Parameters
    ----------
    params_conf: list(dict)
        Configuration to sample parameters with grid samples
    task_func: function
        The function object
    default_params: dict, optional
        default parameters for paralleizied job
    condition: function, optional
        Filter for sampling
    num_threads, int, optional
        The number of threads for parallelization. if not feeded, use the maximum
        number of process

    Returns
    -------
    List: each element is results of each part
    """
    samples = grid_samples(params_conf, condition=condition)
    if default_params is None:
        default_params = dict()
    jobs = []
    for sample in samples:
        sample['func'] = functools.partial(task_func,**default_params)
        jobs.append(sample)
    output = process_jobs(jobs, num_threads=num_threads, task='task')
    return output