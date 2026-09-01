from gom.solvers.scip_mapping import build_transformed_variable_map, resolve_original_variable_id


class FakeVar:
    def __init__(self, name):
        self.name = name


class FakeModel:
    def getTransformedVar(self, var):
        return FakeVar(f"t_{var.name}")


def test_transformed_variables_resolve_to_original_ir_ids():
    originals = {"x0": FakeVar("x0"), "x1": FakeVar("x1")}
    mapping = build_transformed_variable_map(FakeModel(), originals)
    assert mapping == {"t_x0": "x0", "t_x1": "x1"}
    assert resolve_original_variable_id(FakeVar("t_x1"), mapping, originals) == "x1"


def test_unknown_transformed_variable_is_not_mislabeled():
    originals = {"x0": FakeVar("x0")}
    mapping = build_transformed_variable_map(FakeModel(), originals)
    assert resolve_original_variable_id(FakeVar("t_missing"), mapping, originals) is None


def test_prefix_fallback_only_accepts_known_original_id():
    originals = {"x0": FakeVar("x0")}
    assert resolve_original_variable_id(FakeVar("t_x0"), {}, originals) == "x0"
    assert resolve_original_variable_id(FakeVar("t_x999"), {}, originals) is None
