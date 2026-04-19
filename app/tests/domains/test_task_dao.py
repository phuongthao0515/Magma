from datetime import datetime

import pytest
from pydantic import ValidationError

from src.domains.task.dao import (
    ActionDAO,
    ActionParametersDAO,
    ActionType,
    SuccessResponseDAO,
    TaskCreateDAO,
    TaskDAO,
    TaskProcessDAO,
    TaskStatus,
)


class TestTaskDAO:
    def test_task_create_dao_requires_non_empty_prompt(self):
        with pytest.raises(ValidationError):
            TaskCreateDAO(prompt="")

    def test_task_process_dao_uses_default_step(self):
        payload = TaskProcessDAO(task_id="task-1", screenshot_base64="ZmFrZQ==")

        assert payload.step == 0

    def test_task_dao_uses_expected_defaults(self):
        task = TaskDAO(prompt="Open the settings page")

        assert task.id
        assert task.status == TaskStatus.PENDING
        assert task.current_step == 0
        assert task.max_steps == 2
        assert task.actions_history == []
        assert isinstance(task.created_at, datetime)

    def test_action_dao_default_parameters_are_not_shared_between_instances(self):
        first_action = ActionDAO(action_type=ActionType.CLICK)
        second_action = ActionDAO(action_type=ActionType.CLICK)

        first_action.parameters.x = 100

        assert first_action.parameters.x == 100
        assert second_action.parameters.x is None

    def test_action_parameters_dao_uses_expected_defaults(self):
        params = ActionParametersDAO()

        assert params.button == "left"
        assert params.clicks == 1
        assert params.text is None
        assert params.keys is None

    def test_success_response_dao_uses_standard_envelope_defaults(self):
        response = SuccessResponseDAO(data={"ok": True})

        assert response.data == {"ok": True}
        assert response.api_version == "v1.0"
        assert response.errors is None
