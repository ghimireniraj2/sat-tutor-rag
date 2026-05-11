import { useChat } from '../hooks/useChat'
import { ModeSelector } from '../components/ModeSelector'
import { ChatHistory } from '../components/ChatHistory'
import { ChatInput } from '../components/ChatInput'

export function ChatPage() {
  const { messages, mode, isLoading, error, submit, clearChat, switchMode }
    = useChat()

  return (
    <div className="flex flex-col h-screen bg-gray-50">

      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-4 py-3
                         flex items-center justify-between shrink-0">
        <div>
          <h1 className="font-semibold text-gray-900">AI SAT Tutor</h1>
          <p className="text-xs text-gray-500">
            Powered by Retrieval-Augmented Generation (RAG) · {mode === 'explain' ? 'Explain mode' : 'Practice mode'}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <ModeSelector mode={mode} onSwitch={switchMode} />
          <button
            onClick={clearChat}
            className="text-xs text-gray-400 hover:text-gray-600 transition-colors"
          >
            Clear
          </button>
        </div>
      </header>

      {/* Error banner */}
      {error && (
        <div className="bg-red-50 border-b border-red-200 px-4 py-2
                        text-sm text-red-700">
          {error}
          <button
            onClick={() => {}}
            className="ml-2 underline"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Messages */}
      <ChatHistory messages={messages} isLoading={isLoading} />

      {/* Input */}
      <ChatInput onSubmit={submit} isLoading={isLoading} mode={mode} />

    </div>
  )
}