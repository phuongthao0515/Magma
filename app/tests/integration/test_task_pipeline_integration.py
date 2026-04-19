import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from src.domains.task.dao import ActionDAO, TaskCreateDAO, TaskProcessDAO
from src.domains.task.service import TaskService

pytestmark = pytest.mark.integration

INTEGRATION_FIXTURES_DIR = (
    Path(__file__).resolve().parents[1] / "fixtures" / "integration"
)
SCENARIO_MANIFEST_PATH = Path(
    os.environ.get(
        "FULL_APP_INTEGRATION_SCENARIOS",
        str(INTEGRATION_FIXTURES_DIR / "scenarios.json"),
    )
)


@dataclass
class StepExpectation:
    image: str
    expected_status: str | None = None
    expected_action_type: str | None = None
    expected_parameters: dict[str, Any] | None = None
    expected_pyautogui_call: str | None = None
    coordinate_tolerance: int = 0


@dataclass
class IntegrationScenario:
    prompt: str
    steps: list[StepExpectation]
    max_steps: int | None = None


def _require_real_model_and_manifest() -> None:
    if os.environ.get("FULL_APP_USE_REAL_MODEL_TESTS") != "1":
        pytest.skip(
            "Set FULL_APP_USE_REAL_MODEL_TESTS=1 to disable the unit-test model stub "
            "and run real image-based integration tests."
        )

    if not SCENARIO_MANIFEST_PATH.exists():
        pytest.skip(
            "Integration scenario manifest not found. "
            f"Create {SCENARIO_MANIFEST_PATH} from the example template."
        )


def _load_manifest() -> dict[str, Any]:
    _require_real_model_and_manifest()
    return json.loads(SCENARIO_MANIFEST_PATH.read_text(encoding="utf-8"))


def _load_scenario(name: str) -> IntegrationScenario:
    manifest = _load_manifest()
    if name not in manifest:
        pytest.skip(f"Scenario '{name}' is missing from {SCENARIO_MANIFEST_PATH}")

    raw_scenario = manifest[name]
    steps = [StepExpectation(**step) for step in raw_scenario["steps"]]
    return IntegrationScenario(
        prompt=raw_scenario["prompt"],
        steps=steps,
        max_steps=raw_scenario.get("max_steps"),
    )


def _image_to_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def _render_pyautogui_call(action: ActionDAO) -> str:
    params = action.parameters
    action_type = action.action_type.value

    if action_type == "click":
        return (
            "pyautogui.click("
            f"x={params.x}, y={params.y}, button={params.button!r}, clicks={params.clicks})"
        )
    if action_type == "double_click":
        return f"pyautogui.doubleClick(x={params.x}, y={params.y})"
    if action_type == "right_click":
        return f"pyautogui.rightClick(x={params.x}, y={params.y})"
    if action_type == "type":
        return f"pyautogui.write({params.text!r}, interval=0.05)"
    if action_type == "done":
        return "done"
    return f"unsupported:{action_type}"


def _assert_expected_parameters(
    action: ActionDAO, expected_parameters: dict[str, Any], coordinate_tolerance: int
) -> None:
    for key, expected_value in expected_parameters.items():
        actual_value = getattr(action.parameters, key)
        if (
            key in {"x", "y"}
            and expected_value is not None
            and actual_value is not None
            and coordinate_tolerance > 0
        ):
            assert abs(actual_value - expected_value) <= coordinate_tolerance
            continue
        assert actual_value == expected_value


def _run_scenario(scenario_name: str, expected_step_count: int) -> None:
    scenario = _load_scenario(scenario_name)
    assert len(scenario.steps) == expected_step_count

    task = TaskService.create_task(TaskCreateDAO(prompt=scenario.prompt))
    task.max_steps = scenario.max_steps or max(task.max_steps, len(scenario.steps))

    for step_index, step in enumerate(scenario.steps):
        image_path = (SCENARIO_MANIFEST_PATH.parent / step.image).resolve()
        if not image_path.exists():
            pytest.skip(f"Missing integration image: {image_path}")

        response = TaskService.process_screenshot(
            TaskProcessDAO(
                task_id=task.id,
                screenshot_base64=_image_to_base64(image_path),
                step=step_index,
            )
        )

        if step.expected_status is not None:
            assert response.status.value == step.expected_status

        if step.expected_action_type is not None:
            assert response.action.action_type.value == step.expected_action_type

        if step.expected_parameters is not None:
            _assert_expected_parameters(
                response.action,
                step.expected_parameters,
                step.coordinate_tolerance,
            )

        if step.expected_pyautogui_call is not None:
            assert _render_pyautogui_call(response.action) == step.expected_pyautogui_call

    assert len(task.actions_history) == expected_step_count
    assert task.current_step == expected_step_count


class TestTaskPipelineIntegration:
    def test_one_step_prompt_with_one_image(self):
        _run_scenario("one_step", expected_step_count=1)

    def test_two_step_prompt_with_two_images(self):
        _run_scenario("two_step", expected_step_count=2)

    def test_font_size_two_step_prompt_with_two_images(self):
        _run_scenario("two_step_font_size", expected_step_count=2)
