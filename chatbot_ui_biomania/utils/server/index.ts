import {Message} from '@/types/chat';

export class OpenAIError extends Error {
  type: string;
  param: string;
  code: string;

  constructor(message: string, type: string, param: string, code: string) {
    super(message);
    this.name = 'OpenAIError';
    this.type = type;
    this.param = param;
    this.code = code;
  }
}

export const url = process.env.BACKEND_URL || "http://localhost:5000"
const streamUrl = url + '/stream';

export const BioMANIAStream = async (
  method: string,
  messages: Message[],
  top_k: number,
  Lib: string,
  files: { id: string; data: string, type:string, filename:string }[],
  new_lib_github_url: string,
  new_lib_doc_url: string,
  api_html: string,
  lib_alias: string,
  conversation_started: boolean,
  session_id:string,
  optionalParams:string,
) => {
  // streamed response
  const response = await fetch(streamUrl, {
    method: 'POST',
    body: JSON.stringify({
      text: messages[messages.length-1].content,
      top_k: top_k,
      method: method,
      Lib: Lib,
      files: files,
      new_lib_github_url: new_lib_github_url,
      new_lib_doc_url: new_lib_doc_url,
      api_html: api_html,
      lib_alias: lib_alias,
      conversation_started: conversation_started,
      session_id:session_id,
      optionalParams:optionalParams,
    }),
    headers: {
      'Content-Type': 'application/json'
    }
  });
  //@ts-ignore
  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  const convertToReadableStream = async (reader: ReadableStreamDefaultReader<Uint8Array>) => {
    const stream = new ReadableStream({
      async start(controller) {
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) {
              controller.close();
              return;
            }
            controller.enqueue(value);
          }
        } finally {
          reader.releaseLock();
        }
      },
    });

    return stream;
  };

  return await convertToReadableStream(reader);
};
