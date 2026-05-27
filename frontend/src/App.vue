<script setup>
import { computed, ref } from 'vue'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || ''
const videoUrl = ref('')
const question = ref('')
const sessionId = ref('')
const videoId = ref('')
const isCreatingSession = ref(false)
const isSendingMessage = ref(false)
const errorMessage = ref('')
const statusMessage = ref('Paste a YouTube URL to load the transcript.')
const chatMessages = ref([
  {
    role: 'assistant',
    text: 'I am ready when you are. Add a YouTube video, load the transcript, and start asking questions.',
  },
])

const canAskQuestions = computed(() => Boolean(sessionId.value))

function pushMessage(role, text) {
  chatMessages.value.push({
    role,
    text,
  })
}

async function startSession() {
  if (!videoUrl.value.trim()) {
    errorMessage.value = 'Add a YouTube link first.'
    return
  }

  errorMessage.value = ''
  statusMessage.value = 'Loading transcript and building the retrieval chain...'
  isCreatingSession.value = true

  try {
    const response = await fetch(`${API_BASE_URL}/api/sessions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ video_url: videoUrl.value }),
    })

    const data = await response.json()

    if (!response.ok) {
      throw new Error(data.detail || 'Failed to load the video transcript.')
    }

    sessionId.value = data.session_id
    videoId.value = data.video_id
    statusMessage.value = data.message
    chatMessages.value = [
      {
        role: 'assistant',
        text: 'Transcript loaded. Ask anything about the video and I will answer from the captured context.',
      },
    ]
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : 'Something went wrong.'
    statusMessage.value = 'Ready to try another video.'
  } finally {
    isCreatingSession.value = false
  }
}

async function sendQuestion() {
  const cleanQuestion = question.value.trim()

  if (!cleanQuestion || !sessionId.value) {
    return
  }

  const activeQuestion = cleanQuestion
  question.value = ''
  errorMessage.value = ''
  pushMessage('user', activeQuestion)
  isSendingMessage.value = true

  try {
    const response = await fetch(`${API_BASE_URL}/api/sessions/${sessionId.value}/messages`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question: activeQuestion }),
    })

    const data = await response.json()

    if (!response.ok) {
      throw new Error(data.detail || 'Failed to generate an answer.')
    }

    pushMessage('assistant', data.answer)
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : 'Something went wrong.'
  } finally {
    isSendingMessage.value = false
  }
}
</script>

<template>
  <main class="shell">
    <section class="hero">
      <div class="hero-copy">
        <p class="eyebrow">ytChatbot</p>
        <h1>Ask questions about any YouTube video without leaving the page.</h1>
        <p class="lede">
          Load a transcript, then chat with a retrieval-backed assistant that answers only from
          the video context.
        </p>

        <div class="status-card">
          <span class="status-dot"></span>
          <div>
            <strong>{{ statusMessage }}</strong>
            <p v-if="videoId">Active video id: {{ videoId }}</p>
            <p v-else>Waiting for a transcript to be loaded.</p>
          </div>
        </div>
      </div>

      <div class="panel panel-accent">
        <form class="stack" @submit.prevent="startSession">
          <label for="video-url">YouTube link</label>
          <input
            id="video-url"
            v-model="videoUrl"
            type="text"
            placeholder="https://www.youtube.com/watch?v=..."
            autocomplete="off"
          />
          <button class="primary" type="submit" :disabled="isCreatingSession">
            {{ isCreatingSession ? 'Loading transcript...' : 'Load transcript' }}
          </button>
        </form>

        <div class="helper-grid">
          <article>
            <span>01</span>
            <p>Transcript is fetched and translated when needed.</p>
          </article>
          <article>
            <span>02</span>
            <p>Answers are grounded in FAISS retrieval over the transcript.</p>
          </article>
        </div>
      </div>
    </section>

    <section class="chat-layout">
      <div class="panel chat-panel">
        <div class="panel-heading">
          <div>
            <p class="eyebrow">Conversation</p>
            <h2>Chat with the video</h2>
          </div>
          <span class="pill" :class="{ ready: canAskQuestions }">
            {{ canAskQuestions ? 'Ready' : 'Waiting for video' }}
          </span>
        </div>

        <div class="messages">
          <article
            v-for="(message, index) in chatMessages"
            :key="`${message.role}-${index}`"
            class="message"
            :class="message.role"
          >
            <p>{{ message.text }}</p>
          </article>
        </div>

        <form class="composer" @submit.prevent="sendQuestion">
          <textarea
            v-model="question"
            rows="3"
            placeholder="Ask for a summary, key idea, or a specific timestamp-related detail..."
            :disabled="!canAskQuestions || isSendingMessage"
          />
          <button class="secondary" type="submit" :disabled="!canAskQuestions || isSendingMessage">
            {{ isSendingMessage ? 'Thinking...' : 'Send question' }}
          </button>
        </form>

        <p v-if="errorMessage" class="error">{{ errorMessage }}</p>
      </div>

      <aside class="panel side-panel">
        <p class="eyebrow">What this does</p>
        <ul>
          <li>Accepts normal YouTube watch, share, shorts, or youtu.be links.</li>
          <li>Creates a reusable session so follow-up questions stay in context.</li>
          <li>Uses the existing transcript, translation, and LangChain pipeline.</li>
        </ul>

        <div class="mini-card">
          <span>Session</span>
          <strong>{{ sessionId || 'No session yet' }}</strong>
        </div>
      </aside>
    </section>
  </main>
</template>
