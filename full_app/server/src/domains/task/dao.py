from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"


class ActionType(StrEnum):
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    TYPE = "type"
    SELECT = "select"
    HOTKEY = "hotkey"
    SCROLL = "scroll"
    MOVE = "move"
    DRAG = "drag"
    DONE = "done"


# --- Request DAOs ---


class TaskCreateDAO(BaseModel):
    prompt: str = Field(..., min_length=1, description="User instruction for the agent")


class TaskProcessDAO(BaseModel):
    task_id: str = Field(..., description="Task ID")
    screenshot_base64: str = Field(..., description="Base64-encoded screenshot image")
    step: int = Field(default=0, description="Current step number in the loop")


# --- Response DAOs ---


class ActionParametersDAO(BaseModel):
    x: int | None = None
    y: int | None = None
    button: str = "left"
    text: str | None = None
    keys: list[str] | None = None
    dx: int | None = None
    dy: int | None = None
    clicks: int = 1


class ActionDAO(BaseModel):
    action_type: ActionType
    parameters: ActionParametersDAO = Field(default_factory=ActionParametersDAO)
    description: str = ""


class TaskDAO(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str
    status: TaskStatus = TaskStatus.PENDING
    current_step: int = 0
    max_steps: int = 20
    actions_history: list[ActionDAO] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"from_attributes": True}


class TaskProcessResponseDAO(BaseModel):
    task_id: str
    action: ActionDAO
    status: TaskStatus
    step: int
    message: str = ""


class SuccessResponseDAO(BaseModel):
    data: Any
    api_version: str = "v1.0"
    errors: None = None


class FailureResponseDAO(BaseModel):
    data: dict[str, Any] | None = None
    api_version: str = "v1.0"
    errors: dict[str, Any] | None = None
