"""Dimensional analysis over SymPy expressions.

Every equation in the design is registered with the physical dimension of each
symbol. `check_homogeneous` walks the expression tree and fails when two terms
of a sum, or two sides of a relation, carry different dimensions, when a
transcendental function receives a dimensioned argument, or when a dimensioned
quantity is raised to a non-numeric power.
"""

from __future__ import annotations

from fractions import Fraction

import sympy as sp

BASE = ("M", "L", "T", "Θ", "N")   # mass, length, time, temperature, amount

Dim = tuple[Fraction, ...]
DIMENSIONLESS: Dim = tuple(Fraction(0) for _ in BASE)


class DimensionError(ValueError):
    pass


def parse_dim(spec: str) -> Dim:
    """'M L^-1 T^-2' -> exponent tuple. '1' is dimensionless."""
    exps = dict.fromkeys(BASE, Fraction(0))
    for token in spec.split():
        if token == "1":
            continue
        base, _, exp = token.partition("^")
        if base not in exps:
            raise DimensionError(f"unknown base dimension {base!r} in {spec!r}")
        exps[base] += Fraction(exp) if exp else Fraction(1)
    return tuple(exps[b] for b in BASE)


def render(d: Dim) -> str:
    parts = [f"{b}^{e}" if e != 1 else b for b, e in zip(BASE, d) if e != 0]
    return " ".join(parts) or "1"


def _add(a: Dim, b: Dim) -> Dim:
    return tuple(x + y for x, y in zip(a, b))


def _scale(a: Dim, k: Fraction) -> Dim:
    return tuple(x * k for x in a)


def dimension_of(expr: sp.Basic, dims: dict[str, Dim]) -> Dim:
    if isinstance(expr, sp.Symbol):
        if expr.name not in dims:
            raise DimensionError(f"symbol {expr.name!r} has no declared dimension")
        return dims[expr.name]
    if expr.is_Number or expr in (sp.pi, sp.E):
        return DIMENSIONLESS
    if isinstance(expr, sp.Add):
        terms = [dimension_of(a, dims) for a in expr.args]
        for t in terms[1:]:
            if t != terms[0]:
                raise DimensionError(
                    f"sum mixes [{render(terms[0])}] and [{render(t)}] in {expr}")
        return terms[0]
    if isinstance(expr, sp.Mul):
        out = DIMENSIONLESS
        for a in expr.args:
            out = _add(out, dimension_of(a, dims))
        return out
    if isinstance(expr, sp.Pow):
        base = dimension_of(expr.base, dims)
        if base == DIMENSIONLESS:
            if dimension_of(expr.exp, dims) != DIMENSIONLESS:
                raise DimensionError(f"dimensioned exponent in {expr}")
            return DIMENSIONLESS
        if not expr.exp.is_Rational:
            raise DimensionError(f"dimensioned base raised to non-numeric power in {expr}")
        return _scale(base, Fraction(int(expr.exp.p), int(expr.exp.q)))
    if isinstance(expr, (sp.Abs, sp.Max, sp.Min)):
        return dimension_of(sp.Add(*expr.args, evaluate=False), dims) if len(expr.args) > 1 \
            else dimension_of(expr.args[0], dims)
    if isinstance(expr, sp.Function):
        for a in expr.args:
            if dimension_of(a, dims) != DIMENSIONLESS:
                raise DimensionError(f"{expr.func.__name__} of a dimensioned argument in {expr}")
        return DIMENSIONLESS
    if isinstance(expr, sp.Integral):
        out = dimension_of(expr.function, dims)
        for var, *_ in expr.limits:
            out = _add(out, dimension_of(var, dims))
        return out
    if isinstance(expr, sp.Derivative):
        out = dimension_of(expr.expr, dims)
        for var, count in expr.variable_count:
            out = _add(out, _scale(dimension_of(var, dims), Fraction(-int(count))))
        return out
    raise DimensionError(f"cannot analyse {type(expr).__name__}: {expr}")


def check_homogeneous(source: str, symbol_dims: dict[str, str], expect: str | None = None) -> Dim:
    """Check an equation/inequality (or bare expression) and return its dimension."""
    dims = {name: parse_dim(spec) for name, spec in symbol_dims.items()}
    local = {name: sp.Symbol(name, positive=True) for name in dims}
    expr = sp.sympify(source, locals=local, evaluate=False)
    if isinstance(expr, sp.core.relational.Relational):
        lhs, rhs = dimension_of(expr.lhs, dims), dimension_of(expr.rhs, dims)
        if lhs != rhs:
            raise DimensionError(f"sides differ: [{render(lhs)}] vs [{render(rhs)}]")
        result = lhs
    else:
        result = dimension_of(expr, dims)
    if expect is not None and result != parse_dim(expect):
        raise DimensionError(f"expected [{expect}], got [{render(result)}]")
    return result
