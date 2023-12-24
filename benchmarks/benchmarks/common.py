import json
import os
import inspect


def get_params():
    path = os.path.split(__file__)[0]
    fn = os.path.join(path, '..', 'params.json')
    with open(fn, 'r') as f:
        params = json.loads(f.read())

    return (params['sizes'], params['nthreads'])


class BenchMark:
    pass
