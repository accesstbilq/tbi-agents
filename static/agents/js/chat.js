const sessionId =
  'sess-' +
  Math.random().toString(36).slice(2, 10) +
  '-' +
  Date.now().toString(36);
// ---------- Config ----------
const STREAM_URL = '/api/chat'; // endpoint that uses your stream_generator
const messagesEl = document.getElementById('messages');
const usedTokensEl = document.getElementById('usedTokens');
const tokenLimitEl = document.getElementById('tokenLimit');
const usageBarEl = document.getElementById('usageBar');
const usagePercentEl = document.getElementById('usagePercent');
const estimatedCostEl = document.getElementById('estimatedCost');
const connStatusEl = document.getElementById('connStatus');
const stopBtn = document.getElementById('stopBtn');
const clearBtn = document.getElementById('clearBtn');
const emptyState = document.getElementById('emptyState');

let controller = null; // AbortController for the current streaming fetch
let currentAgentMessageId = null;




const chatForm = document.getElementById('chatForm');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');

// Auto-grow helpers
function autoGrow(el) {
    if(el.scrollHeight <= '150'){
        el.style.height = 'auto';
        el.style.height = el.scrollHeight + 'px';
    }
}

function resetAndFocus() {
  messageInput.value = '';
  // reset height to baseline (1 row)
  messageInput.style.height = 'auto';
  messageInput.style.height = '40px'; // baseline, tweak if you want smaller/larger
  messageInput.focus();
}

// Initialize starting height
messageInput.style.height = '40px';
autoGrow(messageInput);

// Listen for input to auto grow
messageInput.addEventListener('input', (e) => {
  autoGrow(e.target);
});

// Key handling: Enter to send, Shift+Enter newline
messageInput.addEventListener('keydown', function (e) {
  if (e.key === 'Enter') {
    if (e.shiftKey) {
        return;
    }
    console.log("messageInput:::::")
    if(emptyState){
      emptyState.remove();
    }
    return startStream(e)
    // Prevent default newline and send the message
  }
});












// ---------- helpers ----------
function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function formatTime(date = new Date()) {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function appendBubble(author, text, meta = '') {
  if (author === 'You') {
    // USER ROW
    const row = document.createElement('div');
    row.className = 'chat-row chat-row-user mb-2';

    const bubble = document.createElement('div');
    bubble.className = 'chat-card-user';
    bubble.innerHTML = text;

    row.appendChild(bubble);
    messagesEl.appendChild(row);
    scrollToBottom();

    return { wrapper: row, body: bubble };
  } else {
    // AGENT ROW (LangChain-style card)
    const row = document.createElement('div');
    row.className = 'chat-row chat-row-agent mb-4';

    // avatar
    const avatar = document.createElement('div');
    avatar.className = 'chat-avatar';
    avatar.innerHTML = '<img src="/static/agents/images/logo.png">'; // you can change to emoji or icon

    // right side (card + footer)
    const right = document.createElement('div');
    right.className = 'flex-1';

    const card = document.createElement('div');
    card.className = 'chat-card msg-agent';

    // header line (Agent steps (1) • Xs)
    const header = document.createElement('div');
    header.className = 'chat-header-line';
    const title = document.createElement('span');
    title.textContent = meta || 'Agent steps (1)';
    const dot = document.createElement('span');
    dot.className = 'dot';
    const time = document.createElement('span');
    time.textContent = formatTime(); // or '4s'
    header.appendChild(title);
    header.appendChild(dot);
    header.appendChild(time);

    // subtitle
    const subtitle = document.createElement('div');
    subtitle.className = 'chat-subtitle';
    subtitle.textContent = '01 Planning next steps...'; // static label, tweak if you like

    // main body (this is what we update while streaming)
    const body = document.createElement('div');
    body.className = 'chat-body';
    body.innerHTML = text;

    card.appendChild(header);
    card.appendChild(subtitle);
    card.appendChild(body);

    // footer
    const footer = document.createElement('div');
    footer.className = 'chat-footer';
    const footerLeft = document.createElement('div');
    footerLeft.className = 'chat-footer-left';

    ['Copy', 'Regenerate', 'Good', 'Bad', 'Feedback'].forEach((label) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'chat-footer-btn';
      btn.textContent = label;
      footerLeft.appendChild(btn);
    });

    const footerRight = document.createElement('div');
    footerRight.textContent = 'Loading Trace…';

    footer.appendChild(footerLeft);
    footer.appendChild(footerRight);

    right.appendChild(card);
    right.appendChild(footer);

    row.appendChild(avatar);
    row.appendChild(right);

    messagesEl.appendChild(row);
    scrollToBottom();

    // return both the wrapper and the body element we will stream into
    return { wrapper: row, body };
  }
}

// SSE-like parser for fetch streaming body (handles "data: {...}\n\n" chunks)
async function streamResponseToEvents(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // split on double-newline boundary for SSE events
    let parts = buffer.split(/\n\n/);
    // keep last partial chunk in buffer
    buffer = parts.pop();

    for (const part of parts) {
      // part contains lines like: data: {"type":"...","..."}
      const lines = part.split(/\n/).filter(Boolean);
      for (const line of lines) {
        // only support "data: " lines
        const m = line.match(/^data:\s*(.*)$/);
        if (!m) continue;
        try {
          const payload = JSON.parse(m[1]);
          onEvent(payload);
        } catch (err) {
          console.warn('Failed parse payload:', m[1], err);
        }
      }
    }
  }

  // if any leftover buffer contains one last event
  if (buffer.trim()) {
    const lines = buffer.split(/\n/).filter(Boolean);
    for (const line of lines) {
      const m = line.match(/^data:\s*(.*)$/);
      if (!m) continue;
      try {
        const payload = JSON.parse(m[1]);
        onEvent(payload);
      } catch (err) {
        console.warn('Failed parse last payload:', m[1], err);
      }
    }
  }
}

// ---------- main streaming logic ----------
async function startStream(e) {

  e.preventDefault();
  const input = document.getElementById('messageInput');
  const model = document.getElementById('modelSelect').value;
  const text = input.value.trim();
  if (!text) return false;

  // add user bubble
  appendBubble('You', escapeHtml(text));
  input.value = '';
  connStatusEl.textContent = 'connecting';
  connStatusEl.classList.remove('text-green-600');
  connStatusEl.classList.add('text-yellow-600');

  // if previous controller exists, abort it
  if (controller) {
    try {
      controller.abort();
    } catch (_) {}
  }
  controller = new AbortController();
  stopBtn.classList.remove('hidden');

    // Create a place-holder for agent streaming text
  const agentBubble = appendBubble('Agent', '');
  const agentWrapper = agentBubble.wrapper;
  const agentBody = agentBubble.body;
  let streamedText = '';

  const formData = new FormData();

  formData.append('user_message', text);
  formData.append('session_id', sessionId);

  // Build request body to match your backend's expected agent_input/config
  // const payload = {
  //   client_text: text,
  //   session_id: sessionId,
  //   config: { model }
  // };

  // fetch (POST) and stream
  try {
    const resp = await fetch(STREAM_URL, {
      method: 'POST',
      body: formData,
      signal: controller.signal
    });

    if (!resp.ok) {
      agentBody.textContent = `Error: ${resp.status} ${resp.statusText}`;
      connStatusEl.textContent = 'error';
      connStatusEl.classList.remove('text-yellow-600');
      connStatusEl.classList.add('text-red-600');
      stopBtn.classList.add('hidden');
      return false;
    }

    connStatusEl.textContent = 'streaming';
    connStatusEl.classList.remove('text-yellow-600');
    connStatusEl.classList.add('text-green-600');
    messageInput.value = '';
    // reset height to baseline (1 row)
    messageInput.style.height = 'auto';
    messageInput.style.height = '40px';

    // parse SSE-like streamed events
    await streamResponseToEvents(resp, (event) => {
      // event is the object you emitted in stream_generator
      const type = event.type || 'message';

      if (type === 'streaming') {
        // live partial message content
        const chunk = event.message || '';
        streamedText += chunk;
        // update agent bubble in place
        agentBody.innerHTML = escapeHtml(streamedText);
      }else if (type === 'usage') {
        // update tokens
        const inT = event.input_tokens || 0;
        const outT = event.output_tokens || 0;
        const total = event.total_tokens || inT + outT;
        updateTokenUsage(
          total,
          parseInt(tokenLimitEl.textContent || '4096'),
          estimateCost(total)
        );
        
      } else if (type === 'error') {
        const errEl = document.createElement('div');
        errEl.className = 'mt-2 text-sm text-red-600';
        errEl.textContent = 'Error: ' + (event.message || 'unknown');
        agentWrapper.appendChild(errEl);
      }
    });

    // streaming finished
    connStatusEl.textContent = 'connected';
    connStatusEl.classList.remove('text-green-600');
    connStatusEl.classList.add('text-green-600'); // keep green
  } catch (err) {
    if (err.name === 'AbortError') {
      // stopped by user
      const brk = document.createElement('div');
      brk.className = 'mt-2 text-xs text-gray-500';
      brk.textContent = 'Streaming stopped.';
      agentWrapper.appendChild(brk);
      connStatusEl.textContent = 'stopped';
      connStatusEl.classList.remove('text-green-600');
      connStatusEl.classList.add('text-yellow-600');
    } else {
      const errEl = document.createElement('div');
      errEl.className = 'mt-2 text-sm text-red-600';
      errEl.textContent = 'Network error: ' + (err.message || err);
      agentWrapper.appendChild(errEl);
      connStatusEl.textContent = 'error';
      connStatusEl.classList.remove('text-yellow-600');
      connStatusEl.classList.add('text-red-600');
    }
  } finally {
    stopBtn.classList.add('hidden');
    controller = null;
    scrollToBottom();
  }

  return false;
}

// ---------- small utilities ----------
clearBtn.addEventListener('click', () => {
  messagesEl.innerHTML = '';
  usedTokensEl.textContent = '0';
  usageBarEl.style.width = '0%';
  usagePercentEl.textContent = '0%';
  estimatedCostEl.textContent = '$0.0000';
});

stopBtn.addEventListener('click', () => {
  if (controller) {
    controller.abort();
  }
});

function updateTokenUsage(used, limit, cost) {
  const pct = Math.min(100, Math.round((used / limit) * 100));
  usedTokensEl.textContent = used;
  tokenLimitEl.textContent = limit;
  usageBarEl.style.width = pct + '%';
  usagePercentEl.textContent = pct + '%';
  estimatedCostEl.textContent = '$' + cost.toFixed(4);
}

// Very small cost estimator placeholder — replace with your pricing formula
function estimateCost(tokens) {
  // example: $0.000002 per token
  return tokens * 0.000002;
}

// Get CSRF token from cookie (Django default)
function getCsrfToken() {
  const name = 'csrftoken';
  const cookies = document.cookie.split(';').map((c) => c.trim());
  for (const c of cookies) {
    if (c.startsWith(name + '=')) return decodeURIComponent(c.split('=')[1]);
  }
  return '';
}

// simple html escaper
function escapeHtml(s) {
  return s
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;')
    .replaceAll('\n', '<br/>');
}
