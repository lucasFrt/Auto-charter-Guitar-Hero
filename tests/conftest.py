import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yargen.chart.ir import AnalysisIR  # noqa: E402

DATA = Path(__file__).parent / "data"


@pytest.fixture
def riff_ir() -> AnalysisIR:
    """IR fixa em JSON: dois compassos de um riff, sem nenhum DSP envolvido."""
    return AnalysisIR.load(DATA / "riff.analysis.json")
