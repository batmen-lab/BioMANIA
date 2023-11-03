import { ToolLLaMAModel } from './toolllama';

export interface BaseUsage {
    block_id: string;
    occurence : number;
    type: "tool" | "llm" | "recommendation" | "root";
    ongoing: boolean;
    depth: number;
    children: BaseUsage[];
    parent: BaseUsage | null;
}
export interface IntermediateMessage {
    role: Role;
    content: string;
}
export interface LLMUsage extends BaseUsage {
    type: "llm";
    block_id: string;
    occurence : number;
    messages: IntermediateMessage[];
    response: string;
}
export interface ToolUsage extends BaseUsage {
    type: "tool";
    block_id: string;
    occurence : number;
    Parameters: string;
    status: number;
    api_name: string;
    api_description: string;
    api_calling: string;
    task: string;
    task_title: string;
    composite_code: string
}
export interface ToolParameters {
  options: any;
  prompt: any;
  required: any;
  type: any;
}
export interface Tool {
  name: string;
  description: string;
  parameters: ToolParameters;
}
export interface ToolRecommendation extends BaseUsage {
  type: "recommendation";
  occurence : number;
  block_id: string;
  recommendations: Tool[];
}
export interface FileObject {
  id: string;
  data: File | string;
  type: 'file' | 'url';
  filename: string;
}
export interface Message {
  role: Role;
  content: string;
  tools: ToolUsage[] | LLMUsage[];
  recommendations: ToolRecommendation[];
  files: FileObject[] | null;
  conversation_started?: boolean;
}
export type Role = 'assistant' | 'user';
export interface ChatBody {
  method: ToolLLaMAModel;
  messages: Message[];
  top_k: number;
  Lib: string;
  files: { id: string; data: string; type:string; filename:string; }[];
  new_lib_github_url: string;
  new_lib_doc_url: string;
  api_html: string;
  lib_alias: string;
  conversation_started: boolean;
}
export interface Conversation {
  id: string;
  name: string;
  messages: Message[];
  method: ToolLLaMAModel;
  top_k: number;
  folderId: string | null;
  Lib: string;
  files: { id: string; data: string; type:string; filename:string; }[];
  new_lib_github_url: string;
  new_lib_doc_url: string;
  api_html: string;
  lib_alias: string;
  conversation_started: boolean;
}
