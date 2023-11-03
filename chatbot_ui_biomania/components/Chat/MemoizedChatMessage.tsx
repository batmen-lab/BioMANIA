import { FC, memo } from "react";
import { ChatMessage, Props } from "./ChatMessage";
import {LLMUsage, Role, ToolUsage} from "@/types/chat";

// export interface BaseUsage {
//   block_id: string;
//   occurence : number;
//   type: "tool" | "llm" | "recommendation";
//   ongoing: boolean;
//   depth: number;
// }
//
// export interface IntermediateMessage {
//   role: Role;
//   content: string;
// }
// export interface LLMUsage extends BaseUsage {
//   type: "llm";
//   block_id: string;
//   occurence : number;
//   messages: IntermediateMessage[];
//   response: string;
// }
//
//
// export interface ToolUsage extends BaseUsage {
//   type: "tool";
//   block_id: string;
//   occurence : number;
//   action: string;
//   status: number;
//   tool_name: string;
//   tool_description: string;
//   tool_input: string;
//   output: string;
// }
//
// export interface ToolParameters {
//   options: any;
//   prompt: any;
//   required: any;
//   type: any;
// }
// export interface Tool {
//   name: string;
//   description: string;
//   parameters: ToolParameters;
// }
// export interface ToolRecommendation extends BaseUsage {
//   type: "recommendation";
//   occurence : number;
//   block_id: string;
//   recommendations: Tool[];
// }
//
// export interface Message {
//   role: Role;
//   content: string;
//   tools: ToolUsage[] | LLMUsage[];
//   recommendations: ToolRecommendation[];
// }

const CompareElements = (a: LLMUsage | ToolUsage, b: LLMUsage | ToolUsage) => {
    // check if types are the same
    if (a.type !== b.type) {
        return false;
    }
    if (a.type === "llm") {
        a = a as LLMUsage;
        b = b as LLMUsage;
        if (a.block_id !== b.block_id
          || a.occurence !== b.occurence
          || a.messages !== b.messages
          || a.response !== b.response) {
            return false;
        }
    } else if (a.type === "tool") {
        a = a as ToolUsage;
        b = b as ToolUsage;
        if (a.block_id !== b.block_id
          || a.occurence !== b.occurence
          || a.Parameters !== b.Parameters
          || a.status !== b.status
          || a.api_name !== b.api_name
          || a.api_description !== b.api_description
          || a.task !== b.task
          || a.task_title !== b.task_title
          || a.composite_code!==b.composite_code) {
            return false;
        }
    }
    return true;
}


export const MemoizedChatMessage: FC<Props> = memo(
    ChatMessage,
    (prevProps, nextProps) => {
        return false;
        if (prevProps.message.content !== nextProps.message.content) {
            return false;
        }
        if (prevProps.message.role === "user") {
            return true;
        }
        var prevundefined = prevProps.message.tools === undefined;
        var nextundefined = nextProps.message.tools === undefined;
        if (prevundefined !== nextundefined) {
            return false;
        }
        if (prevundefined === undefined && nextundefined === undefined) {
          return true;
        }

        // now they should both be defined
        if (prevProps.message.tools.length !== nextProps.message.tools.length) {
            return false;
        }

        // for (var i = 0; i < prevProps.message.tools.length; i++) {
        //     if (!CompareElements(prevProps.message.tools[i], nextProps.message.tools[i])) {
        //         return false;
        //     }
        // }

        return true;
    }
);
