import { useMutation, useQueryClient } from "@tanstack/react-query";
import { createTask, deleteTask, processScreenshot } from "./task";
import type { TaskProcessRequest } from "../types/task";
import { TASK_QUERY_KEYS } from "./task.query";

export const useCreateTaskMutation = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (prompt: string) => createTask(prompt),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: TASK_QUERY_KEYS.all });
    },
  });
};

export const useProcessScreenshotMutation = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (payload: TaskProcessRequest) => processScreenshot(payload),
    onSuccess: (data) => {
      queryClient.invalidateQueries({
        queryKey: TASK_QUERY_KEYS.detail(data.task_id),
      });
    },
  });
};

export const useDeleteTaskMutation = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (taskId: string) => deleteTask(taskId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: TASK_QUERY_KEYS.all });
    },
  });
};
