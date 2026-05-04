import pandas as pd

import pipeline


def test_pipeline_main_runs_steps(monkeypatch) -> None:
    calls: list[str] = []

    def record(name: str):
        def _inner(*_args, **_kwargs) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame] | None:
            calls.append(name)
            if name == "engineer":
                return pd.DataFrame({"FTR": [0], "ELO_DIFF": [1.0]})
            if name == "preprocess":
                df = pd.DataFrame({"FTR": [0], "ELO_DIFF": [1.0]})
                return df, df
            return None
        return _inner

    monkeypatch.setattr(pipeline._collect, "main", record("collect"))
    monkeypatch.setattr(pipeline._validate, "main", record("validate"))
    monkeypatch.setattr(pipeline._engineer, "main", record("engineer"))
    monkeypatch.setattr(pipeline._preprocess, "main", record("preprocess"))
    monkeypatch.setattr(pipeline._visualize, "main", record("visualize"))
    monkeypatch.setattr(pipeline._select, "main", record("select"))
    monkeypatch.setattr(pipeline._train, "main", record("train"))
    monkeypatch.setattr(pipeline.pd, "read_csv", lambda _path: pd.DataFrame({"FTR": [0]}))

    pipeline.main()

    assert calls == [
        "collect",
        "validate",
        "engineer",
        "preprocess",
        "visualize",
        "select",
        "train",
    ]
