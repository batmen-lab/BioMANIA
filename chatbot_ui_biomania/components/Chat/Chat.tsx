import { IconClearAll, IconSettings } from '@tabler/icons-react';
import {
  MutableRefObject,
  memo,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from 'react';
import toast from 'react-hot-toast';
import { useTranslation } from 'next-i18next';
import { v4 as uuidv4 } from 'uuid';
import {fileToBase64} from './ChatInput'
import {
  saveConversation,
  saveConversations,
  updateConversation,
} from '@/utils/app/conversation';
import { throttle } from '@/utils/data/throttle';
import { ChatBody, Conversation, Message, FileObject } from '@/types/chat';
import { Plugin } from '@/types/plugin';
import { LibSelect } from './LibSelect';
import HomeContext from '@/pages/api/home/home.context';
import { ChatInput } from './ChatInput';
import { ChatLoader } from './ChatLoader';
import { LibCardSelect } from './LibCardSelect';
import { SystemPrompt } from './SystemPrompt';
import { MemoizedChatMessage } from './MemoizedChatMessage';
import { ToolUsage } from '@/types/chat';
interface Props {
  stopConversationRef: MutableRefObject<boolean>;
}
export const Chat = memo(({ stopConversationRef }: Props) => {
  const { t } = useTranslation('chat');
  const [optionalParams, setOptionalParams] = useState<string>('');
  // From OptionalCard 
  const handleOptionalParamsChange = (newParams: Record<string, any>) => {
  const optionalParamsString = JSON.stringify(newParams);
  setOptionalParams(optionalParamsString);
};
  const {
    state: {
      selectedConversation,
      conversations,
      methods,
      apiKey,
      pluginKeys,
      serverSideApiKeyIsSet,
      messageIsStreaming,
      modelError,
      loading,
    },
    handleUpdateConversation,
    dispatch: homeDispatch,
  } = useContext(HomeContext);
  const [currentMessage, setCurrentMessage] = useState<Message>();
  const [autoScrollEnabled, setAutoScrollEnabled] = useState<boolean>(true);
  const [showSettings, setShowSettings] = useState<boolean>(false);
  const [showScrollDownButton, setShowScrollDownButton] =
    useState<boolean>(false);
  const [attachedFiles, setAttachedFiles] = useState<FileObject[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const toolStart: ToolUsage = {
    type: "tool",
    block_id: "tool-1",
    occurence: 1,
    Parameters: "",
    status: 0,
    api_name: "",
    api_description: "",
    api_calling: "",
    task: "",
    task_title: "",
    composite_code: "",
    ongoing: false,
    depth: 0,
    children: [],
    parent: null
  };
  const toolEnd: ToolUsage = {
    type: "tool",
    block_id: "tool-1",
    occurence: 1,
    Parameters: "",
    status: 0,
    api_name: "",
    api_description: "",
    api_calling: "",
    task: "",
    task_title: "",
    composite_code: "",
    ongoing: false,
    depth: 0,
    children: [],
    parent: null
  };
  const agentAction: ToolUsage = {
    type: "tool",
    block_id: "log-1",
    occurence: 1,
    Parameters: "",
    status: 0,
    api_name: "",
    api_description: "",
    api_calling: "",
    task: "[Would you like to see some examples to learn how to interact with the bot?](https://github.com/batmen-lab/BioMANIA/tree/35da76f7d5cd40b99980654e84000e25ff167cb2/examples)",
    task_title: "Welcome to BioMANIA! How can I help?",
    composite_code: "",
    ongoing: false,
    depth: 0,
    children: [],
    parent: null
  };
  
  const [welcomeMessage] = useState<Message>({
    role: 'assistant',
    content: '',
    tools: [toolStart, toolEnd, agentAction],
    recommendations: [],
    files: [],
    session_id: selectedConversation?.id || 'default-session-id'
  });
  const handleSend = useCallback(
    async (message: Message, deleteCount = 0, plugin: Plugin | null = null) => {
      if (selectedConversation) {
        const isFirstMessageInConversation = selectedConversation.messages.length === 0;
        let updatedMessage = {...message};
        
        if (attachedFiles.length > 0) {
          updatedMessage = {
            ...message,
            files: attachedFiles,
          };
          setAttachedFiles([]); // Clear the attached files after adding them to the message
        }
        let updatedConversation: Conversation;
        if (deleteCount) {
          const updatedMessages = [...selectedConversation.messages];
          for (let i = 0; i < deleteCount; i++) {
            updatedMessages.pop();
          }
          updatedConversation = {
            ...selectedConversation,
            messages: [...updatedMessages, message],
          };
        } else {
          const isNewConversation = selectedConversation.messages.length === 0;
          updatedConversation = {
            ...selectedConversation,
            messages: isNewConversation
              ? [
                  {
                    role: 'assistant',
                    content: '',
                    tools: [toolStart, toolEnd, agentAction],
                    recommendations: [],
                    files: [],
                    session_id: selectedConversation.id 
                  },
                  message
                ]
              : [...selectedConversation.messages, message],
          };
        }
        homeDispatch({
          field: 'selectedConversation',
          value: updatedConversation,
        });
        homeDispatch({ field: 'loading', value: true });
        homeDispatch({ field: 'messageIsStreaming', value: true });
        console.log("Set messageIsStreaming to true");
        const chatBody: ChatBody = {
          method: updatedConversation.method,
          messages: updatedConversation.messages,
          top_k: 1, 
          Lib: updatedConversation.Lib,
          files: updatedConversation.files,
          new_lib_github_url: updatedConversation.new_lib_github_url,
          new_lib_doc_url: updatedConversation.new_lib_doc_url,
          api_html: updatedConversation.api_html,
          lib_alias: updatedConversation.lib_alias,
          conversation_started: isFirstMessageInConversation,
          session_id: updatedConversation.id,
          optionalParams: optionalParams,
        };
        console.log("updatedConversation", updatedConversation)
        const endpoint = "api/chat";
        if (message.files) {
          const filePromises = message.files.map(fileObject => {
            if (typeof fileObject.data === 'string') {
              // if fileObject.file is a string, return it directly
              return Promise.resolve({ id: fileObject.id, data: fileObject.data, type: fileObject.type, filename: fileObject.filename });
            } else {
              // if fileObject.file is a fileObject, use fileToBase64 to convert it
              return fileToBase64(fileObject.data).then(base64 => ({ id: fileObject.id, data: base64, type:fileObject.type, filename: fileObject.filename }));
            }
          });
          try {
            const filesData = await Promise.all(filePromises);
            chatBody.files = filesData;
          } catch (error) {
            console.error("Error converting files to base64", error);
          }
        }
        let body;
        body = JSON.stringify(chatBody);
        const controller = new AbortController();
        const formData = new FormData();
        formData.append('chatBody', JSON.stringify(chatBody));
        const response = await fetch(endpoint, {
          method: 'POST',
          signal: controller.signal,
          body,
        });
        if (!response.ok) {
          homeDispatch({ field: 'loading', value: false });
          homeDispatch({ field: 'messageIsStreaming', value: false });
          toast.error(response.statusText);
          return;
        }
        const data = response.body;
        if (!data) {
          homeDispatch({ field: 'loading', value: false });
          homeDispatch({ field: 'messageIsStreaming', value: false });
          return;
        }
        if (!plugin) {
          if (updatedConversation.messages.length === 1) {
            const { content } = message;
            const customName =
              content.length > 30 ? content.substring(0, 30) + '...' : content;
            updatedConversation = {
              ...updatedConversation,
              name: customName,
            };
          }
          homeDispatch({ field: 'loading', value: false });
          const reader = data.getReader();
          const decoder = new TextDecoder('utf-8');
          let done = false;
          let isFirst = true;
          let text = '';
          let result = '';
          let resultObjs: any = [];
          let error = false;
          while (!done) {
            if (stopConversationRef.current) {
              controller.abort();
              done = true;
              break;
            }
            const { value, done: doneReading } = await reader.read();
            done = doneReading;
            if (error) break;
            if (done) {
              if (resultObjs.length === 0) {
                error = true;
              }
            }
            var decoded = decoder.decode(value);
            result += decoded;
            var resultSplit = result.split('\n');
            resultObjs = resultSplit.map((line) => {
              try {
                var obj = JSON.parse(line);
                if (obj?.method_name === "on_request_end") {
                  var temp = JSON.parse(obj.output);
                  text = temp.final_answer;
                }
                return obj;
              } catch (e) {
                // if the line is the last line, ignore it
                console.error("JSON parsing error:", e);
                console.error("Faulty line:", line);
                if (line === "") {
                  return null;
                }
              }
            });
            //@ts-ignore
            resultObjs = resultObjs.filter((obj) => obj !== null);
            // Now we have a list of resultObjs just like before, turn it into a list of messages
            if (isFirst) {
              isFirst = false;
              const updatedMessages: Message[] = [
                ...updatedConversation.messages,
                { role: 'assistant', content: text, tools: resultObjs, recommendations: [], files:[],session_id: selectedConversation.id },
              ];
              updatedConversation = {...updatedConversation, messages: updatedMessages};
              homeDispatch({field: 'selectedConversation', value: updatedConversation});

            } else {
              // create deep copy of updatedConversation
              const updatedMessages: Message[] = updatedConversation.messages.map((message, index) => message);
              updatedMessages[updatedMessages.length - 1] = {
                ...updatedMessages[updatedMessages.length - 1],
                content: text,
                tools: resultObjs,
              }
              updatedConversation = { ...updatedConversation, messages: updatedMessages};
              homeDispatch({ field: 'selectedConversation', value: updatedConversation});
            }
          }
          console.log("updatedConversation", updatedConversation);
          saveConversation(updatedConversation);
          const updatedConversations: Conversation[] = conversations.map( (conversation) => {
              if (conversation.id === selectedConversation.id) return updatedConversation;
              return conversation;
            },
          );
          if (updatedConversations.length === 0) {
            updatedConversations.push(updatedConversation);
          }
          homeDispatch({ field: 'conversations', value: updatedConversations });
          saveConversations(updatedConversations);
          homeDispatch({ field: 'messageIsStreaming', value: false });
        } else {
          const { answer } = await response.json();
          const updatedMessages: Message[] = [
            ...updatedConversation.messages,
            { role: 'assistant', content: answer, tools: [], recommendations: [], files:[],session_id: selectedConversation.id },
          ];
          updatedConversation = {...updatedConversation, messages: updatedMessages};
          homeDispatch({ field: 'selectedConversation', value: updateConversation});
          saveConversation(updatedConversation);
          const updatedConversations: Conversation[] = conversations.map(
            (conversation) => {
              if (conversation.id === selectedConversation.id) {
                return updatedConversation;
              }
              return conversation;
            },
          );
          if (updatedConversations.length === 0) {
            updatedConversations.push(updatedConversation);
          }
          homeDispatch({ field: 'conversations', value: updatedConversations });
          saveConversations(updatedConversations);
          homeDispatch({ field: 'loading', value: false });
          homeDispatch({ field: 'messageIsStreaming', value: false });
        }
      }
    },
    [
      apiKey,
      conversations,
      pluginKeys,
      selectedConversation,
      stopConversationRef,
      attachedFiles,
      selectedConversation, 
      optionalParams,
    ],
  );
  const handleFileUpload = useCallback(
    (file: File) => {
      const fileExtension = file.name.split(".").pop()?.toLowerCase();
      setAttachedFiles((prevFiles) => [...prevFiles, { id: uuidv4(), data:file, type: 'file' , filename: file.name}]);
       if (selectedConversation) {
        const updatedConversation = {
          ...selectedConversation,
          files: [...(selectedConversation.files || []), { id: uuidv4(), data:file, type: 'file' , filename: file.name}],
        };
        homeDispatch({ field: 'selectedConversation', value: updatedConversation });
      }
    },
    [selectedConversation]
  );
  const scrollToBottom = useCallback(() => {
    if (autoScrollEnabled) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
      textareaRef.current?.focus();
    }
  }, [autoScrollEnabled]);
  const handleScroll = () => {
    if (chatContainerRef.current) {
      const { scrollTop, scrollHeight, clientHeight } =
        chatContainerRef.current;
      const bottomTolerance = 30;

      if (scrollTop + clientHeight < scrollHeight - bottomTolerance) {
        setAutoScrollEnabled(false);
        setShowScrollDownButton(true);
      } else {
        setAutoScrollEnabled(true);
        setShowScrollDownButton(false);
      }
    }
  };
  const handleScrollDown = () => {
    chatContainerRef.current?.scrollTo({
      top: chatContainerRef.current.scrollHeight,
      behavior: 'smooth',
    });
  };
  const handleSettings = () => {
    setShowSettings(!showSettings);
  };
  const onClearAll = () => {
    if (
      confirm(t<string>('Are you sure you want to clear all messages?')) &&
      selectedConversation
    ) {
      handleUpdateConversation(selectedConversation, {
        key: 'messages',
        value: [],
      });
    }
  };
  const scrollDown = () => {
    if (autoScrollEnabled) {
      messagesEndRef.current?.scrollIntoView(true);
    }
  };
  const throttledScrollDown = throttle(scrollDown, 250);
  // useEffect(() => {
  //   console.log('currentMessage', currentMessage);
  //   if (currentMessage) {
  //     handleSend(currentMessage);
  //     homeDispatch({ field: 'currentMessage', value: undefined });
  //   }
  // }, [currentMessage]);
  useEffect(() => {
    throttledScrollDown();
    selectedConversation &&
      setCurrentMessage(
        selectedConversation.messages[selectedConversation.messages.length - 2],
      );
  }, [selectedConversation, throttledScrollDown]);
  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        setAutoScrollEnabled(entry.isIntersecting);
        if (entry.isIntersecting) {
          textareaRef.current?.focus();
        }
      },
      {
        root: null,
        threshold: 0.5,
      },
    );
    const messagesEndElement = messagesEndRef.current;
    if (messagesEndElement) {
      observer.observe(messagesEndElement);
    }
    return () => {
      if (messagesEndElement) {
        observer.unobserve(messagesEndElement);
      }
    };
  }, [messagesEndRef]);
  function displayDefaultScreen(selectedConversation: Conversation) {
    return (
      <>
        <div className="mx-auto flex flex-col space-y-5 md:space-y-10 px-3 pt-5 md:pt-12 sm:max-w-[600px] bg-white">
          <div className="text-center text-3xl font-semibold text-gray-800 dark:text-gray-100">
            BioMANIA UI
          </div>
          <div className="flex h-full flex-col space-y-4 rounded-lg border border-neutral-200 p-4 dark:border-neutral-600 overflow-auto bg-white">
            <LibCardSelect />
            <div className="welcome-message mt-4 bg-white">
              <MemoizedChatMessage
                key="welcomeMessage"
                message={welcomeMessage}
                messageIndex={0}
                onEdit={(editedMessage) => {
                  setCurrentMessage(editedMessage);
                  handleSend(editedMessage, selectedConversation?.messages.length);
                }}
              />
            </div>
          </div>
        </div>
      </>
    );
}
  const reportRef = useRef(null);
  return (
    <div className="relative flex-1 overflow-hidden bg-white dark:bg-[#343541]">
      {
          (
        <>
          <div
            style={{ height: '85vh', overflowX: 'hidden' }}
            ref={chatContainerRef}
            onScroll={handleScroll}
          >
            {selectedConversation?.messages.length === 0 ? (
              <>
              {displayDefaultScreen(selectedConversation)}
              {/*<MemoizedChatMessage
                key="welcomeMessage"
                message={welcomeMessage}
                messageIndex={0}
                onEdit={(editedMessage) => {
                  setCurrentMessage(editedMessage);
                  handleSend(editedMessage, selectedConversation?.messages.length);
                }}
              />*/}
            </>
            ) : (
              <>
                <div className="sticky top-0 z-10 flex justify-center border border-b-neutral-300 bg-neutral-100 py-2 text-sm text-neutral-500 dark:border-none dark:bg-[#444654] dark:text-neutral-200">
                  {t('Top K')}
                  : {selectedConversation?.top_k} |  {t('Lib')}
                  : {selectedConversation?.Lib} |  {/* present curret selected Lib */}
                  <button
                    className="ml-2 cursor-pointer hover:opacity-50"
                    onClick={handleSettings}
                  >
                    <IconSettings size={18} />
                  </button>
                  <button
                    className="ml-2 cursor-pointer hover:opacity-50"
                    onClick={onClearAll}
                  >
                    <IconClearAll size={18} />
                  </button>
                </div>
                {showSettings && (
                  <div className="fixed-top-container">
                    <div className="flex flex-col space-y-10 md:mx-auto md:max-w-xl md:gap-6 md:py-3 md:pt-6 lg:max-w-2xl lg:px-0 xl:max-w-3xl">
                      <div className="flex h-full flex-col space-y-4 border-b border-neutral-200 p-4 dark:border-neutral-600 md:rounded-lg md:border">
                        <LibSelect />
                      </div>
                    </div>
                  </div>
                )}
                <style jsx>{`
                  .fixed-top-container {
                    position: fixed;
                    top: 0%;
                    left: 60%;
                    transform: translate(-50%, -50%);
                    width: auto;
                    z-index: 1000;
                    background-color: rgba(255, 255, 255, 0.8);
                  }
                `}</style>
                {selectedConversation?.messages.map((message, index) => (
                  <MemoizedChatMessage
                    key={index}
                    message={message}
                    messageIndex={index}
                    onEdit={(editedMessage) => {
                      setCurrentMessage(editedMessage);
                      // discard edited message and the ones that come after then resend
                      handleSend(
                        editedMessage,
                        selectedConversation?.messages.length - index,
                      );
                    }
                  }
                  />
                ))}

                {loading && <ChatLoader />}

                <div
                  className="h-[162px] bg-white dark:bg-[#343541]"
                  ref={messagesEndRef}
                />
              </>
            )}
          </div>
          <ChatInput
            stopConversationRef={stopConversationRef}
            textareaRef={textareaRef}
            onSend={(message, plugin) => {
              setCurrentMessage(message);
              handleSend(message, 0, plugin);
            }}
            onScrollDownClick={handleScrollDown}
            onRegenerate={() => {
              if (currentMessage) {
                handleSend(currentMessage, 2, null);
              }
            }}
            showScrollDownButton={showScrollDownButton}
            onUpload={handleFileUpload}
          />
        </>
      )}
    </div>
  );
});
Chat.displayName = 'Chat';
