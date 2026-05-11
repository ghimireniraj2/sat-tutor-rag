// frontend/src/components/ChatMessage.jsx
import { PracticeCard } from './PracticeCard'
import ReactMarkdown from 'react-markdown'

export function ChatMessage({ message }) {
  const isUser = message.role === 'user'

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div className={`
        max-w-[80%]
        ${isUser ? 'order-2' : 'order-1'}
      `}>
        {isUser ? (
          <div className="bg-gray-900 text-white px-4 py-2.5 rounded-2xl
                          rounded-tr-sm text-sm">
            {message.content}
          </div>
        ) : message.type === 'practice' ? (
          <PracticeCard {...message.content} />
        ) : (
          <div className="bg-white border border-gray-200 px-4 py-3
                          rounded-2xl rounded-tl-sm text-sm text-gray-800
                          leading-relaxed whitespace-pre-wrap !text-left">
              <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
        )}
      </div>
    </div>
  )
}