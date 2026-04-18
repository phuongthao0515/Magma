export type TaskStatus = "pending" | "in_progress" | "done" | "failed" | "cancelled";

export type ActionType =
  | "click"
  | "double_click"
  | "right_click"
  | "type"
  | "hotkey"
  | "scroll"
  | "move"
  | "drag"
  | "done";

export interface ActionParameters {
  x?: number;
  y?: number;
  button?: string;
  text?: string;
  keys?: string[];
  dx?: number;
  dy?: number;
  clicks?: number;
}

export interface Action {
  action_type: ActionType;
  parameters: ActionParameters;
  description: string;
}

export interface Task {
  id: string;
  prompt: string;
  status: TaskStatus;
  current_step: number;
  max_steps: number;
  actions_history: Action[];
  created_at: string;
}

export interface TaskProcessRequest {
  task_id: string;
  screenshot_base64: string;
  step: number;
}

export interface TaskProcessResponse {
  task_id: string;
  action: Action;
  status: TaskStatus;
  step: number;
  message: string;
}
