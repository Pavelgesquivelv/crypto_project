import duckdb
import numpy
import pandas
import polars
import pyarrow


def test_core_dependencies_import() -> None:
    assert polars.__version__
    assert pandas.__version__
    assert numpy.__version__
    assert pyarrow.__version__
    assert duckdb.__version__
