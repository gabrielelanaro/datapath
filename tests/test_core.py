import numpy as np
import os

from trails.core import DataCache, make_path
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression


def make_data_set():
    X = np.arange(20)[np.newaxis].T
    y = np.arange(20)
    return X, y


def split(prev):
    X, y = prev
    return train_test_split(X, y)


def square(prev):
    X, y = prev

    return X**2, y


def linear_fit(dataset, C=1.0):
    X_train, X_test, y_train, y_test = dataset
    lr = LinearRegression(C)
    lr.fit(X_train, y_train)

    return lr, lr.score(X_test, y_test)

import py


@py.test.fixture()
def datacache(request):
    dc = DataCache('./dc')

    def teardown():
        py.path.local('./dc').remove()
    request.addfinalizer(teardown)

    return dc


def test_example1(datacache):
    dataset = datacache.step(make_data_set).step(split).checkpoint()
    dataset.record(linear_fit)

    dataset2 = datacache.step(make_data_set).step(
        square).step(split).checkpoint()
    
    dataset2.record(linear_fit)

    # We need to check if hash was created
    assert os.path.exists('./dc/' + make_path(dataset.trail) + '.hash')
    
    print(datacache.summary())


def test_hash_changed(datacache):
    global make_data_set
    make_data_set_old = make_data_set

    hash_1 = datacache.step(make_data_set).step(split).hash()
    hash_2 = datacache.step(make_data_set).step(square).step(split).hash()
    hash_3 = datacache.step(make_data_set).hash()

    assert hash_1 != hash_2

    # A function was rewritten in a different way
    def make_data_set():
        X = np.arange(10)[np.newaxis].T
        y = np.arange(10)
        return X, y

    # A function was rewritten
    hash_4 = datacache.step(make_data_set).hash()
    assert hash_3 != hash_4

    # The code of a dependency changed, we're supposed to
    # update our cache
    hash_5 = datacache.step(make_data_set).step(split).hash()
    assert hash_1 != hash_5

    make_data_set = make_data_set_old


def test_cache_func_changed(datacache):
    global make_data_set
    make_data_set_old = make_data_set
    step = datacache.step(make_data_set).step(square).checkpoint()
    step_x = step.get()[0]

    # A function was rewritten in a different way
    def make_data_set():
        X = np.arange(10)[np.newaxis].T
        y = np.arange(10)
        return X, y

    step_mod = datacache.step(make_data_set).step(square).checkpoint()
    step_mod_x = step_mod.get()[0]
    assert not np.array_equal(step_x, step_mod_x)
    make_data_set = make_data_set_old


def test_new_datacache(datacache):
    datacache.step(make_data_set).step(square).checkpoint()
    datacache_new = DataCache('dc')
    datacache_new.step(make_data_set).step(square).checkpoint()


def make_data_set_p(size):
    return np.arange(size)


def power(step, power):
    return step ** power


def test_unpacking(datacache):
    X, y = datacache.step(make_data_set) # Unfortunately he has to run
    X_train, X_test, y_train, y_test = datacache.step(train_test_split, X, y)

    print(X_train.get())


# def test_parameterized(datacache):
#     val = datacache.step(make_data_set_p, 10).step(power, 3).record()
#     assert np.array_equal(val, np.arange(10) ** 3)
#
#     val = datacache.step(make_data_set_p, 20).step(power, 3).record()
#     assert np.array_equal(val, np.arange(10) ** 3)
