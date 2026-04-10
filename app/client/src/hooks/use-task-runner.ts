import { useCallback, useEffect, useRef } from "react";
import { useStore } from "@tanstack/react-store";
import { taskStore } from "../stores/task";
import { createTask, getTask } from "../services/task";
import { useQueryClient } from "@tanstack/react-query";
import { TASK_QUERY_KEYS } from "../services/task.query";
import type { StepLog } from "../stores/task";
import type { Task } from "../types/task";

const POLL_INTERVAL = 2000;

export const useTaskRunner = () => {
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const lastStepRef = useRef(0);
  const queryClient = useQueryClient();

  const store = useStore(taskStore);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const pollTaskProgress = useCallback(
    async (taskId: string) => {
      try {
        const task: Task = await getTask(taskId);

        // Build new step logs from actions_history that we haven't seen yet
        const newActions = task.actions_history.slice(lastStepRef.current);
        if (newActions.length > 0) {
          const newLogs: StepLog[] = newActions.map((action, i) => ({
            step: lastStepRef.current + i + 1,
            action,
            status: task.status,
            message: `Step ${lastStepRef.current + i + 1} executed`,
            timestamp: new Date().toISOString(),
          }));

          lastStepRef.current = task.actions_history.length;

          taskStore.setState((s) => ({
            ...s,
            stepLogs: [...s.stepLogs, ...newLogs],
          }));
        }

        // Update status
        if (task.status === "done" || task.status === "failed") {
          stopPolling();
          taskStore.setState((s) => ({
            ...s,
            isRunning: false,
            finalStatus: task.status,
          }));
          queryClient.invalidateQueries({ queryKey: TASK_QUERY_KEYS.all });
        }
      } catch {
        // Silently retry on next interval
      }
    },
    [stopPolling, queryClient]
  );

  const startTask = useCallback(
    async (prompt: string) => {
      if (!prompt.trim()) return;

      // Create task on backend (status: pending)
      const task = await createTask(prompt);

      lastStepRef.current = 0;
      taskStore.setState((s) => ({
        ...s,
        activeTaskId: task.id,
        isRunning: true,
        stepLogs: [],
        finalStatus: null,
      }));

      // Start polling for progress updates from the agent
      // First poll fires immediately so we don't miss fast tasks
      stopPolling();
      pollTaskProgress(task.id);
      pollRef.current = setInterval(() => {
        pollTaskProgress(task.id);
      }, POLL_INTERVAL);
    },
    [pollTaskProgress, stopPolling]
  );

  const stopTask = useCallback(() => {
    stopPolling();
    taskStore.setState((s) => ({ ...s, isRunning: false }));
  }, [stopPolling]);

  // Cleanup on unmount
  useEffect(() => {
    return () => stopPolling();
  }, [stopPolling]);

  return {
    ...store,
    startTask,
    stopTask,
  };
};
