from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_dashboard_runs_and_switches_models():
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    at.multiselect[0].set_value(["LEAR (Lasso, one model per hour)"]).run()
    assert not at.exception
    assert at.dataframe
