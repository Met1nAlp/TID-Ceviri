/**
 * DeepSign TID — Frontend
 * Gerçek zamanlı tahmin, cümle oluşturma, durum halkası, animasyonlar.
 */

class TIDApp {
    constructor() {
        this.predictions   = [];
        this.sentence      = [];
        this.selection     = {};
        this.signState     = 'idle';
        this.speechVoices  = [];
        this.preferredVoice = null;

        this.init();
    }

    init() {
        this.bindEvents();
        this.initSpeechVoices();
        this.startPolling();
        this.loadSentence();

        const vid = document.getElementById('videoFeed');
        vid.onload  = () => document.getElementById('videoOverlay').classList.remove('active');
        vid.onerror = () => document.getElementById('videoOverlay').classList.add('active');
    }

    // ── Events ──────────────────────────────────────────────
    bindEvents() {
        document.querySelectorAll('.prediction-item').forEach((item) => {
            item.addEventListener('click', (e) => {
                const idx = parseInt(item.dataset.index, 10);
                if (this.predictions[idx]) {
                    this.addWord(this.predictions[idx].label_tr, e);
                }
            });
        });

        document.getElementById('clearPredictions').addEventListener('click', () => this.clearPredictions());
        document.getElementById('clearSentence').addEventListener('click', () => this.clearSentence());
        document.getElementById('speakSentence').addEventListener('click', (e) => {
            this.speakSentence();
            this.ripple(e.currentTarget, e);
        });
    }

    // ── Speech ──────────────────────────────────────────────
    initSpeechVoices() {
        if (!('speechSynthesis' in window)) return;
        const load = () => {
            this.speechVoices = window.speechSynthesis.getVoices();
            this.preferredVoice = this._pickTurkishVoice(this.speechVoices);
        };
        load();
        window.speechSynthesis.addEventListener('voiceschanged', load);
    }

    _pickTurkishVoice(voices) {
        if (!Array.isArray(voices) || !voices.length) return null;
        const tr = voices.filter(v => v.lang?.toLowerCase().startsWith('tr'));
        if (!tr.length) return null;

        const preferred = [
            'microsoft tolga online (natural)',
            'microsoft emel online (natural)',
            'microsoft ahmet',
            'microsoft filiz',
            'google türkçe',
        ];
        for (const pat of preferred) {
            const m = tr.find(v => v.name.toLowerCase().includes(pat));
            if (m) return m;
        }
        return tr.find(v => v.name.toLowerCase().includes('natural')) ?? tr[0];
    }

    speakSentence() {
        if (!this.sentence?.length) return;
        if (!('speechSynthesis' in window)) { alert('Tarayıcınız seslendirmeyi desteklemiyor.'); return; }

        window.speechSynthesis.cancel();
        const utt = new SpeechSynthesisUtterance(this.sentence.join(' '));
        utt.lang = 'tr-TR'; utt.rate = 0.75; utt.pitch = 0.96; utt.volume = 1.0;

        if (!this.preferredVoice) {
            this.preferredVoice = this._pickTurkishVoice(window.speechSynthesis.getVoices());
        }
        if (this.preferredVoice) { utt.voice = this.preferredVoice; utt.lang = this.preferredVoice.lang || 'tr-TR'; }
        window.speechSynthesis.speak(utt);
    }

    // ── Polling ─────────────────────────────────────────────
    startPolling() {
        // Selection status: tahminler + cümle + seçim durumu
        setInterval(() => this.fetchSelectionStatus(), 200);
        // Debug status: sign state (kayıt/bekliyor) → halka rengi
        setInterval(() => this.fetchDebugStatus(), 250);
    }

    async fetchSelectionStatus() {
        try {
            const res = await fetch('/selection_status');
            const sel = await res.json();
            this.selection = sel || {};

            if (Array.isArray(sel.sentence)) {
                this.sentence = sel.sentence;
                this.renderSentence();
            }
            if (Array.isArray(sel.predictions)) {
                this.predictions = sel.predictions;
                this.renderPredictions();
            }
            this.renderSelectionHighlight();
        } catch { /* sessiz */ }
    }

    async fetchDebugStatus() {
        try {
            const res   = await fetch('/debug_status');
            const debug = await res.json();
            if (debug?.state) this.setSignState(debug.state);
        } catch { /* sessiz */ }
    }

    // ── Sign State Ring ─────────────────────────────────────
    setSignState(state) {
        if (state === this.signState) return;
        this.signState = state;

        const ring  = document.getElementById('statusRing');
        const badge = document.getElementById('videoBadge');
        const dot   = document.getElementById('badgeDot');
        const label = document.getElementById('badgeLabel');

        const cfg = {
            signing:   { text: 'Kayıt',    blink: true },
            selection: { text: 'Seçim',    blink: true },
            idle:      { text: 'Bekliyor', blink: false },
        }[state] ?? { text: 'Bekliyor', blink: false };

        ring.className  = `status-ring ${state}`;
        badge.className = `video-badge ${state}`;
        label.textContent = cfg.text;
        dot.className = 'badge-dot';
    }

    // ── Predictions UI ──────────────────────────────────────
    renderPredictions() {
        const p = this.predictions;
        for (let i = 0; i < 3; i++) {
            document.getElementById(`pred${i+1}Label`).textContent = p[i] ? `${i+1}. ${p[i].label_tr}` : `${i+1}. —`;
            document.getElementById(`pred${i+1}Bar`).style.width   = p[i] ? `${p[i].confidence}%` : '0%';
            document.getElementById(`pred${i+1}Conf`).textContent  = p[i] ? `%${p[i].confidence}` : '%0';
        }
    }

    renderSelectionHighlight() {
        document.querySelectorAll('.prediction-item').forEach(el => el.classList.remove('selection-target'));

        const sel = this.selection || {};
        const active = Boolean(sel.active);
        const digit  = active ? (sel.stable_digit || sel.last_digit_value) : null;

        if (Number.isInteger(digit) && digit >= 1 && digit <= 3) {
            const el = document.querySelector(`.prediction-item[data-index="${digit-1}"]`);
            if (el) el.classList.add('selection-target');
        }
    }

    // ── Sentence UI ─────────────────────────────────────────
    renderSentence() {
        const key = JSON.stringify(this.sentence);
        if (key === this._lastSentenceKey) return;   // değişmediyse yeniden render etme
        this._lastSentenceKey = key;

        const c = document.getElementById('sentenceContainer');
        if (!this.sentence.length) {
            c.innerHTML = '<span class="empty-sentence">Tahminlere tıklayın veya 1 / 2 / 3 gösterin</span>';
            return;
        }
        c.innerHTML = this.sentence.map((w, i) => `
            <div class="sentence-word" onclick="app.removeWord(${i})">
                <span>${w}</span><span class="remove-btn">×</span>
            </div>
        `).join('');
    }

    // ── Server Actions ──────────────────────────────────────
    async addWord(word, event) {
        try {
            const res  = await fetch('/add_word', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ word }),
            });
            const data = await res.json();
            this.sentence = data.sentence;
            this.renderSentence();
            if (event) this.ripple(event.currentTarget, event);
        } catch { /* sessiz */ }
    }

    async removeWord(index) {
        try {
            const res  = await fetch('/remove_word', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ index }),
            });
            const data = await res.json();
            this.sentence = data.sentence;
            this.renderSentence();
        } catch { /* sessiz */ }
    }

    async clearPredictions() {
        try {
            await fetch('/clear_predictions', { method: 'POST' });
            this.predictions = [];
            this.renderPredictions();
            this.selection = {};
            this.renderSelectionHighlight();
        } catch { /* sessiz */ }
    }

    async clearSentence() {
        try {
            const res  = await fetch('/clear_sentence', { method: 'POST' });
            const data = await res.json();
            this.sentence = data.sentence;
            this.renderSentence();
        } catch { /* sessiz */ }
    }

    async loadSentence() {
        try {
            const res  = await fetch('/get_sentence');
            const data = await res.json();
            this.sentence = data.sentence;
            this.renderSentence();
        } catch { /* sessiz */ }
    }

    // ── Ripple animation ────────────────────────────────────
    ripple(el, e) {
        if (!el || !e) return;
        const rect = el.getBoundingClientRect();
        const x = (e.clientX ?? rect.left + rect.width / 2) - rect.left;
        const y = (e.clientY ?? rect.top + rect.height / 2) - rect.top;
        const r = document.createElement('span');
        r.className = 'ripple';
        r.style.cssText = `left:${x}px;top:${y}px`;
        el.appendChild(r);
        setTimeout(() => r.remove(), 700);
    }
}

let app;
document.addEventListener('DOMContentLoaded', () => { app = new TIDApp(); window.app = app; });
