export interface ApiResult<T = unknown> {
  data: T;
  api_version: string;
  errors: ApiErrors | null;
  success: boolean;
}

export interface ApiErrors {
  msg: string[];
  code: string;
}

export enum ErrorCode {
  NETWORK_ERROR = "NETWORK_ERROR",
  UNAUTHORIZED = "UNAUTHORIZED",
  NOT_FOUND = "NOT_FOUND",
  SERVER_ERROR = "SERVER_ERROR",
}
