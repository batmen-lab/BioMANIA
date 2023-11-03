
export interface OpenAIModel {
  id: string;
  name: string;
  maxLength: number; // maximum length of a message
  tokenLimit: number;
}

export interface ToolLLaMAModel {
  id: string;
  name: string;
  method: string;
  maxLength: 8000;
}

export enum ToolLLaMAMethodID {
  DFS = "No specified"
}

export const fallbackMethodID = ToolLLaMAMethodID.DFS;

export const ToolLLaMAMethods: Record<ToolLLaMAMethodID, ToolLLaMAModel> = {
  [ToolLLaMAMethodID.DFS]: {
    id: ToolLLaMAMethodID.DFS,
    name: "DFS without filter",
    method: ToolLLaMAMethodID.DFS,
    maxLength: 8000,
  }
}
