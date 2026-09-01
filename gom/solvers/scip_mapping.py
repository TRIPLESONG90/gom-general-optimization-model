from __future__ import annotations

from typing import Any, Mapping


def build_transformed_variable_map(
    model: Any,
    original_variables: Mapping[str, Any],
) -> dict[str, str]:
    """Map SCIP transformed-variable names back to Optimization IR variable ids.

    SCIP solves in transformed space after presolve. LP branching candidates are
    therefore transformed variables (commonly named ``t_<original>``), while GOM
    is trained and featurized in the original Optimization IR namespace.

    Ambiguous many-to-one transformed mappings are deliberately omitted rather
    than assigning a wrong supervision target.
    """
    mapped: dict[str, str | None] = {}
    for original_id, original_var in original_variables.items():
        try:
            transformed = model.getTransformedVar(original_var)
        except Exception:
            transformed = None
        if transformed is None:
            continue
        name = str(transformed.name)
        if name in mapped and mapped[name] != original_id:
            mapped[name] = None
        else:
            mapped[name] = original_id
    return {name: original_id for name, original_id in mapped.items() if original_id is not None}


def resolve_original_variable_id(
    candidate: Any,
    transformed_map: Mapping[str, str],
    original_variables: Mapping[str, Any],
) -> str | None:
    name = str(candidate.name)
    if name in transformed_map:
        return transformed_map[name]
    if name in original_variables:
        return name

    # Compatibility fallback for SCIP's default transformed naming. Explicit
    # getTransformedVar mappings above always take precedence.
    if name.startswith("t_"):
        original_name = name[2:]
        if original_name in original_variables:
            return original_name
    return None
