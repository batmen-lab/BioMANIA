import { Conversation } from './chat';
import { FolderInterface } from './folder';
import { PluginKey } from './plugin';
import { Prompt } from './prompt';

// keep track of local storage schema
export interface LocalStorage {
  apiKey: string;
  conversationHistory: Conversation[];
  selectedConversation: Conversation;
  theme: 'light' | 'dark';
  folders: FolderInterface[];
  prompts: Prompt[];
  showChatbar: boolean;
  showPromptbar: boolean;
  pluginKeys: PluginKey[];
}
