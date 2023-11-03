import { Conversation } from '@/types/chat';
import {
  ExportFormatV1,
  ExportFormatV2,
  ExportFormatV3,
  ExportFormatV4,
  LatestExportFormat,
  SupportedExportFormats,
} from '@/types/export';
import { FolderInterface } from '@/types/folder';
import { Prompt } from '@/types/prompt';

import { cleanConversationHistory } from './clean';

export function isExportFormatV1(obj: any): obj is ExportFormatV1 {
  return Array.isArray(obj);
}

export function isExportFormatV2(obj: any): obj is ExportFormatV2 {
  return !('version' in obj) && 'folders' in obj && 'history' in obj;
}

export function isExportFormatV3(obj: any): obj is ExportFormatV3 {
  return obj.version === 3;
}

export function isExportFormatV4(obj: any): obj is ExportFormatV4 {
  return obj.version === 4;
}

export const isLatestExportFormat = isExportFormatV4;

export function cleanData(data: SupportedExportFormats): LatestExportFormat {
  if (isExportFormatV1(data)) {
    return {
      version: 4,
      history: cleanConversationHistory(data),
      folders: [],
    };
  }

  if (isExportFormatV2(data)) {
    return {
      version: 4,
      history: cleanConversationHistory(data.history || []),
      folders: (data.folders || []).map((chatFolder) => ({
        id: chatFolder.id.toString(),
        name: chatFolder.name,
        type: 'chat',
      })),
    };
  }

  if (isExportFormatV3(data)) {
    return { ...data, version: 4 };
  }

  if (isExportFormatV4(data)) {
    return data;
  }

  throw new Error('Unsupported data format');
}

function currentDate() {
  const date = new Date();
  const month = date.getMonth() + 1;
  const day = date.getDate();
  return `${month}-${day}`;
}

export const exportData = () => {
  let history = localStorage.getItem('conversationHistory');
  let folders = localStorage.getItem('folders');
  let prompts = localStorage.getItem('prompts');

  if (history) {
    history = JSON.parse(history);
  }

  if (folders) {
    folders = JSON.parse(folders);
  }

  if (prompts) {
    prompts = JSON.parse(prompts);
  }

  const data = {
    version: 4,
    history: history || [],
    folders: folders || [],
    prompts: prompts || [],
  } as LatestExportFormat;

  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: 'application/json',
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.download = `chatbot_ui_history_${currentDate()}.json`;
  link.href = url;
  link.style.display = 'none';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

export const uploadData = (file: File, callback: (data: any) => void) => {
  const reader = new FileReader();
  // 获取文件扩展名
  const fileExtension = file.name.split('.').pop()?.toLowerCase();
  reader.onload = () => {
    let data;
    if (fileExtension === 'json') {
      // JSON 文件处理
      data = JSON.parse(reader.result as string);
    } else if (fileExtension === 'txt') {
      // TXT 文件处理，这里我们直接将内容作为字符串返回
      data = reader.result;
    } else if (fileExtension === 'csv') {
      // CSV 文件处理，你可以根据需要进行更复杂的处理
      data = reader.result;
    } else if (fileExtension === 'xlsx') {
      // XLSX 文件处理，由于 FileReader 不能直接解析 XLSX 文件，你可能需要一个专门的库来处理这种文件格式
      data = reader.result;
    } else {
      // 不支持的文件类型
      throw new Error('Unsupported file type');
    }
    callback(data);
  };
  reader.onerror = () => {
    throw new Error('Error reading file');
  };
  reader.readAsText(file);
};


export const importData = (
  data: SupportedExportFormats,
): LatestExportFormat => {
  const { history, folders } = cleanData(data);

  const oldConversations = localStorage.getItem('conversationHistory');
  const oldConversationsParsed = oldConversations
    ? JSON.parse(oldConversations)
    : [];

  const newHistory: Conversation[] = [
    ...oldConversationsParsed,
    ...history,
  ].filter(
    (conversation, index, self) =>
      index === self.findIndex((c) => c.id === conversation.id),
  );
  localStorage.setItem('conversationHistory', JSON.stringify(newHistory));
  if (newHistory.length > 0) {
    localStorage.setItem(
      'selectedConversation',
      JSON.stringify(newHistory[newHistory.length - 1]),
    );
  } else {
    localStorage.removeItem('selectedConversation');
  }

  const oldFolders = localStorage.getItem('folders');
  const oldFoldersParsed = oldFolders ? JSON.parse(oldFolders) : [];
  const newFolders: FolderInterface[] = [
    ...oldFoldersParsed,
    ...folders,
  ].filter(
    (folder, index, self) =>
      index === self.findIndex((f) => f.id === folder.id),
  );
  localStorage.setItem('folders', JSON.stringify(newFolders));

  // const oldPrompts = localStorage.getItem('prompts');
  // const oldPromptsParsed = oldPrompts ? JSON.parse(oldPrompts) : [];
  // const newPrompts: Prompt[] = [...oldPromptsParsed, ...prompts].filter(
  //   (prompt, index, self) =>
  //     index === self.findIndex((p) => p.id === prompt.id),
  // );
  // localStorage.setItem('prompts', JSON.stringify(newPrompts));

  return {
    version: 4,
    history: newHistory,
    folders: newFolders,
    // prompts: newPrompts,
  };
};
