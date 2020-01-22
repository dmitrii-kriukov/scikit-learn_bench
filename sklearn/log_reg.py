# Copyright (C) 2018-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import parse_args, time_mean_min, print_header, print_row, size_str, convert_data
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser(description='scikit-learn logistic '
                                             'regression benchmark')
parser.add_argument('-x', '--filex', '--fileX', type=argparse.FileType('r'),
                    required=True,
                    help='Input file with features, in NPY format')
parser.add_argument('-y', '--filey', '--fileY', type=argparse.FileType('r'),
                    required=True,
                    help='Input file with labels, in NPY format')
parser.add_argument('--no-fit-intercept', dest='fit_intercept',
                    action='store_false', default=True,
                    help="Don't fit intercept")
parser.add_argument('--multiclass', default='auto',
                    choices=('auto', 'ovr', 'multinomial'),
                    help="How to treat multi class data. "
                         "'auto' picks 'ovr' for binary classification, and "
                         "'multinomial' otherwise.")
parser.add_argument('--solver', default='lbfgs',
                    choices=('lbfgs', 'newton-cg', 'saga'),
                    help='Solver to use.')
parser.add_argument('--maxiter', type=int, default=100,
                    help='Maximum iterations for the iterative solver')
parser.add_argument('-C', dest='C', type=float, default=1.0,
                    help='Regularization parameter')
parser.add_argument('--tol', type=float, default=None,
                    help="Tolerance for solver. If solver == 'newton-cg', "
                         "then the default is 1e-3. Otherwise, the default "
                         "is 1e-10.")
params = parse_args(parser, loop_types=('fit', 'predict'))

# Load generated data
X = convert_data(np.load(params.filex.name), params.dtype, params.data_order, params.data_format)
y = convert_data(np.load(params.filey.name), params.dtype, params.data_order, params.data_format)

params.n_classes = len(np.unique(y))

if params.multiclass == 'auto':
    params.multiclass = 'ovr' if params.n_classes == 2 else 'multinomial'

if not params.tol:
    params.tol = 1e-3 if params.solver == 'newton-cg' else 1e-10

# Create our classifier object
clf = LogisticRegression(penalty='l2', C=params.C, n_jobs=params.n_jobs,
                         fit_intercept=params.fit_intercept,
                         verbose=params.verbose,
                         tol=params.tol, max_iter=params.maxiter,
                         solver=params.solver, multi_class=params.multiclass)

columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'solver', 'C', 'multiclass', 'n_classes', 'accuracy', 'time')
params.size = size_str(X.shape)

# Time fit and predict
fit_time, _ = time_mean_min(clf.fit, X, y,
                            outer_loops=params.fit_outer_loops,
                            inner_loops=params.fit_inner_loops,
                            goal_outer_loops=params.fit_goal,
                            time_limit=params.fit_time_limit,
                            verbose=params.verbose)

predict_time, y_pred = time_mean_min(clf.predict, X,
                                     outer_loops=params.predict_outer_loops,
                                     inner_loops=params.predict_inner_loops,
                                     goal_outer_loops=params.predict_goal,
                                     time_limit=params.predict_time_limit,
                                     verbose=params.verbose)
acc = 100 * accuracy_score(y_pred, y)

if params.output_format == "csv":
    print_header(columns, params)
    print_row(columns, params, function='LogReg.fit', time=fit_time)
    print_row(columns, params, function='LogReg.predict', time=predict_time,
        accuracy=acc)
    if params.verbose:
        print()
        print("@ Number of iterations: {}".format(clf.n_iter_))
        print("@ fit coefficients:")
        print("@ {}".format(clf.coef_.tolist()))
        print("@ fit intercept:")
        print("@ {}".format(clf.intercept_.tolist()))

elif params.output_format == "json":
    import json

    res = {
        "lib": "sklearn",
        "algorithm": "logistic_regression",
        "stage": "training",
        "data_format": params.data_format,
        "data_type": str(params.dtype),
        "data_order": params.data_order,
        "rows": X.shape[0],
        "columns": X.shape[1],
        "classes": params.n_classes,
        "time[s]": fit_time,
        "accuracy[%]": acc,
        "algorithm_paramaters": dict(clf.get_params())
    }

    print(json.dumps(res, indent=4))

    res.update({
        "stage": "prediction",
        "time[s]": predict_time,
        "accuracy[%]": acc
    })

    print(json.dumps(res, indent=4))
