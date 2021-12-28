from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm


def parallel_map(array, worker, const_args=None, n_jobs=16, use_kwargs=False, front_num=3, drop_none=False):
    """
        A parallel version of the map function with a progress bar. Adopted from
            http://danshiebler.com/2016-09-14-parallel-progress-bar/

        Args:
            array (array-like): A list to iterate over
            worker (function): A python function to apply to the elements of array
            const_args (dict, default=None): Constant arguments, shared between all processes
            n_jobs (int, default=16): The number of jobs to launch
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of
                keyword arguments to function (if True than **list[n] is passed)
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job
            drop_none (boolean, default=False): Whether to drop None values from the list of results or not
        Returns:
            [worker(list[0], **const_args), worker(list[1], **const_args), ...]
    """
    # Replace None with empty dict
    const_args = dict() if const_args is None else const_args
    # We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [worker(**a, **const_args) if use_kwargs else worker(a, **const_args) for a in array[:front_num]]
    else:
        front = []
    # If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs == 1:
        return front + [worker(**a, **const_args) if use_kwargs else
                        worker(a, **const_args) for a in tqdm(array[front_num:])]
    # Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        # Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(worker, **a, **const_args) for a in array[front_num:]]
        else:
            futures = [pool.submit(worker, a, **const_args) for a in array[front_num:]]
        tqdm_kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True,
            'ncols': 80
        }
        # Print out the progress as tasks complete
        for _ in tqdm(as_completed(futures), **tqdm_kwargs):
            pass
    out = []
    # Get the results from the futures.
    for i, future in enumerate(futures):
        try:
            out.append(future.result())
        except Exception as e:
            print(f"Caught {str(e)} on {i}-th input.")
            out.append(None)

    if drop_none:
        return [v for v in front+out if v is not None]
    else:
        return front + out
