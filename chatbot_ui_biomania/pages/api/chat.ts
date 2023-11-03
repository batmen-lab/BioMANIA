// import { DEFAULT_SYSTEM_PROMPT, DEFAULT_TEMPERATURE } from '@/utils/app/const';
import { OpenAIError, BioMANIAStream } from '@/utils/server';

import { ChatBody, Message } from '@/types/chat';

export const config = {
  runtime: 'edge',
};
const handler = async (req: Request): Promise<Response> => {
  try {
    const { top_k, method, messages, files, Lib, new_lib_github_url, new_lib_doc_url, conversation_started, api_html,lib_alias } = (await req.json()) as ChatBody;

    let messagesToSend: Message[] = [];

    for (let i = messages.length - 1; i >= 0; i--) {
      const message = messages[i];
      messagesToSend = [message, ...messagesToSend];
    }
    const stream = await BioMANIAStream(method.method, messagesToSend, top_k, Lib, files, new_lib_github_url, new_lib_doc_url, api_html,lib_alias,conversation_started);

    return new Response(stream);
  } catch (error) {
    console.error(error);
    if (error instanceof OpenAIError) {
      return new Response('Error', { status: 500, statusText: error.message });
    } else {
      return new Response('Error', { status: 500 });
    }
  }
};

export default handler;
