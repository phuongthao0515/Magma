import type { FC } from "react";
import { Tag } from "antd";
import type { TaskStatus } from "../../types/task";

const STATUS_COLORS: Record<TaskStatus, string> = {
  pending: "warning",
  in_progress: "processing",
  done: "success",
  failed: "error",
  cancelled: "warning",
};

interface StatusBadgeProps {
  status: TaskStatus;
}

export const StatusBadge: FC<StatusBadgeProps> = ({ status }) => {
  return <Tag color={STATUS_COLORS[status] || "default"}>{status}</Tag>;
};
