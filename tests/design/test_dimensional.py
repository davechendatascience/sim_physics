"""Every registered equation is dimensionally homogeneous."""

import pytest

from design.dimensions import DimensionError, check_homogeneous
from tests.design.conftest import CLAIMS, ids

EQUATIONS = CLAIMS["equations"]


@pytest.mark.parametrize("eq", EQUATIONS, ids=ids(EQUATIONS))
def test_equation_is_homogeneous(eq, tag):
    tag(eq["id"], eq["doc"], "dimensional_homogeneity")
    check_homogeneous(eq["expr"], eq["dims"])


# Negative controls: the checker itself must reject known-bad physics, or a
# passing suite above means nothing.
BAD = [
    ("hoop stress missing thickness", "Eq(sigma, p*R)",
     {"sigma": "M L^-1 T^-2", "p": "M L^-1 T^-2", "R": "L"}),
    ("log of a length", "Eq(b, kappa*log(d))", {"b": "M L^2 T^-2", "kappa": "M L^2 T^-2", "d": "L"}),
    ("adding force to pressure", "F + p", {"F": "M L T^-2", "p": "M L^-1 T^-2"}),
    ("dimensioned exponent", "x**t", {"x": "1", "t": "T"}),
]


@pytest.mark.parametrize("name,expr,dims", BAD, ids=[b[0] for b in BAD])
def test_checker_rejects_bad_physics(name, expr, dims, tag):
    tag(f"CHK-dimensional:{name}", "08-design-validation.md", "checker_negative_control")
    with pytest.raises(DimensionError):
        check_homogeneous(expr, dims)
