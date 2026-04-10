import { useQuery } from "@tanstack/react-query";
import { getTask, listTasks } from "./task";
import type { Task } from "../types/task";

export const TASK_QUERY_KEYS = {
  all: ["tasks"] as const,
  detail: (id: string) => ["tasks", id] as const,
};

export const useTasksQuery = (enabled = true) => {
  return useQuery<Task[], Error>({
    queryKey: TASK_QUERY_KEYS.all,
    queryFn: listTasks,
    staleTime: 0,
    enabled,
  });
};

export const useTaskQuery = (taskId: string, enabled = true) => {
  return useQuery<Task, Error>({
    queryKey: TASK_QUERY_KEYS.detail(taskId),
    queryFn: () => getTask(taskId),
    staleTime: 0,
    enabled: enabled && !!taskId,
  });
};
