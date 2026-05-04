from pathlib import Path

import pandas as pd

import visualize


def test_visualize_writes_pdf(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "FTR": [0, 1, 2, 0, 1, 2],
            "ELO_DIFF": [1, 2, 3, 4, 5, 6],
            "IMP_H_PROB_ODDS": [0.4, 0.5, 0.6, 0.45, 0.55, 0.65],
            "HPTS_FORM": [2, 1, 3, 2, 4, 3],
        }
    )

    output_path = tmp_path / "visualizations.pdf"
    result = visualize.main(df, output_path=str(output_path))

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
