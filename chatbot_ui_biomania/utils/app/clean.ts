import { Conversation } from '@/types/chat';
import { ToolLLaMAMethods, ToolLLaMAMethodID } from '@/types/toolllama';

import {DEFAULT_TOP_K } from './const';

export const cleanSelectedConversation = (conversation: Conversation) => {

  let updatedConversation = conversation;

  // check for model on each conversation
  if (!updatedConversation.method) {
    updatedConversation = {
      ...updatedConversation,
      method: updatedConversation.method || ToolLLaMAMethods[ToolLLaMAMethodID.DFS],
    };
  }

  // check for system prompt on each conversation
  // if (!updatedConversation.prompt) {
  //   updatedConversation = {
  //     ...updatedConversation,
  //     prompt: updatedConversation.prompt || DEFAULT_SYSTEM_PROMPT,
  //   };
  // }

  if (!updatedConversation.top_k) {
    updatedConversation = {
      ...updatedConversation,
      top_k: updatedConversation.top_k || DEFAULT_TOP_K,
    };
  }

  if (!updatedConversation.folderId) {
    updatedConversation = {
      ...updatedConversation,
      folderId: updatedConversation.folderId || null,
    };
  }

  if (!updatedConversation.messages) {
    updatedConversation = {
      ...updatedConversation,
      messages: updatedConversation.messages || [],
    };
  }

  return updatedConversation;
};

export const cleanConversationHistory = (history: any[]): Conversation[] => {

  if (!Array.isArray(history)) {
    console.warn('history is not an array. Returning an empty array.');
    return [];
  }

  return history.reduce((acc: any[], conversation) => {
    try {
      if (!conversation.model) {
        conversation.method = ToolLLaMAMethods[ToolLLaMAMethodID.DFS];
      }

      // if (!conversation.prompt) {
      //   conversation.prompt = DEFAULT_SYSTEM_PROMPT;
      // }

      if (!conversation.top_k) {
        conversation.top_k = DEFAULT_TOP_K;
      }

      if (!conversation.folderId) {
        conversation.folderId = null;
      }

      if (!conversation.messages) {
        conversation.messages = [];
      }

      acc.push(conversation);
      return acc;
    } catch (error) {
      console.warn(
        `error while cleaning conversations' history. Removing culprit`,
        error,
      );
    }
    return acc;
  }, []);
};
