"""Smoke tests for the Streamlit dashboard: every tab renders and reacts to the controls."""

import os

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.join(os.path.dirname(__file__), "..", "app.py")


@pytest.fixture(scope="module")
def app():
    return AppTest.from_file(APP, default_timeout=180).run()


def test_every_tab_renders_without_errors(app):
    assert not app.exception
    assert len(app.tabs) == 7
    assert {m.label for m in app.metric} >= {"Class average", "Attendance link", "Early-warning accuracy"}


def test_dashboard_reacts_to_settings(app):
    pass_mark = next(s for s in app.sidebar.slider if s.label == "Pass mark")
    pass_mark.set_value(50).run()
    assert not app.exception
    app.selectbox[0].select_index(25).run()  # a different student in the explorer
    assert not app.exception
    assert app.metric[-4].value == "46.7"  # Sneha's overall average
