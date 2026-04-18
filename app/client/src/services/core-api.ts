import axios, { type AxiosRequestConfig, type AxiosResponse } from "axios";
import type { ApiResult } from "../types/api";

const axiosInstance = axios.create({
  timeout: 30000,
  headers: {
    "Content-Type": "application/json",
  },
});

// Request interceptor
axiosInstance.interceptors.request.use(
  (config) => {
    // Use Vite proxy in dev (/api → backend), or window.config in production
    if (!config.baseURL) {
      config.baseURL = "";
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor
axiosInstance.interceptors.response.use(
  (response: AxiosResponse) => {
    return response;
  },
  (error) => {
    if (error.response) {
      return Promise.resolve({
        data: {
          data: null,
          api_version: "v1.0",
          errors: {
            msg: [error.response.data?.detail || "Request failed"],
            code: String(error.response.status),
          },
          success: false,
        },
      });
    }
    return Promise.resolve({
      data: {
        data: null,
        api_version: "v1.0",
        errors: { msg: ["Network error"], code: "NETWORK_ERROR" },
        success: false,
      },
    });
  }
);

export const coreApi = {
  get: async <T>(url: string, config?: AxiosRequestConfig): Promise<ApiResult<T>> => {
    const response = await axiosInstance.get<ApiResult<T>>(url, config);
    return { ...response.data, success: !response.data.errors };
  },

  post: async <T>(
    url: string,
    data?: unknown,
    config?: AxiosRequestConfig
  ): Promise<ApiResult<T>> => {
    const response = await axiosInstance.post<ApiResult<T>>(url, data, config);
    return { ...response.data, success: !response.data.errors };
  },

  patch: async <T>(
    url: string,
    data?: unknown,
    config?: AxiosRequestConfig
  ): Promise<ApiResult<T>> => {
    const response = await axiosInstance.patch<ApiResult<T>>(url, data, config);
    return { ...response.data, success: !response.data.errors };
  },

  delete: async <T>(url: string, config?: AxiosRequestConfig): Promise<ApiResult<T>> => {
    const response = await axiosInstance.delete<ApiResult<T>>(url, config);
    return { ...response.data, success: !response.data.errors };
  },
};
