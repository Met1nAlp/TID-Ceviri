/**
 * TID Recognition System - Frontend JavaScript
 * Handles real-time predictions, digit-based selection, sentence building, and UI updates.
 */

class TIDApp {
    constructor() {
        this.predictions = [];
        this.sentence = [];
        this.selection = {};
        this.pollInterval = null;
        this.speechVoices = [];
        this.preferredVoice = null;

        this.init();
    }

    init() {
        this.bindEvents();
        this.initSpeechVoices();
        this.startPolling();
        this.loadSentence();

        const videoFeed = document.getElementById('videoFeed');
        videoFeed.onload = () => {
            document.getElementById('videoOverlay').classList.remove('active');
        };
        videoFeed.onerror = () => {
            document.getElementById('videoOverlay').classList.add('active');
        };
    }

    bindEvents() {
        document.querySelectorAll('.prediction-item').forEach((item) => {
            item.addEventListener('click', () => {
                const index = parseInt(item.dataset.index, 10);
                if (this.predictions[index]) {
                    this.addWord(this.predictions[index].label_tr);
                }
            });
        });

        document.getElementById('clearPredictions').addEventListener('click', () => {
            this.clearPredictions();
        });

        document.getElementById('clearSentence').addEventListener('click', () => {
            this.clearSentence();
        });

        document.getElementById('speakSentence').addEventListener('click', () => {
            this.speakSentence();
        });
    }

    initSpeechVoices() {
        if (!('speechSynthesis' in window)) {
            return;
        }

        const loadVoices = () => {
            this.speechVoices = window.speechSynthesis.getVoices();
            this.preferredVoice = this.selectPreferredTurkishVoice(this.speechVoices);
        };

        loadVoices();
        window.speechSynthesis.addEventListener('voiceschanged', loadVoices);
    }

    selectPreferredTurkishVoice(voices) {
        if (!Array.isArray(voices) || voices.length === 0) {
            return null;
        }

        const preferredPatterns = [
            'microsoft tolga online (natural)',
            'microsoft emel online (natural)',
            'microsoft ahmet',
            'microsoft filiz',
            'google türkçe',
            'google turkce',
            'turkish'
        ];

        const turkishVoices = voices.filter((voice) =>
            typeof voice.lang === 'string' && voice.lang.toLowerCase().startsWith('tr')
        );

        if (turkishVoices.length === 0) {
            return null;
        }

        for (const pattern of preferredPatterns) {
            const matched = turkishVoices.find((voice) =>
                voice.name.toLowerCase().includes(pattern)
            );
            if (matched) {
                return matched;
            }
        }

        const naturalVoice = turkishVoices.find((voice) =>
            voice.name.toLowerCase().includes('natural')
        );
        if (naturalVoice) {
            return naturalVoice;
        }

        return turkishVoices[0];
    }

    startPolling() {
        this.pollInterval = setInterval(() => {
            this.fetchPredictions();
            this.fetchSelectionStatus();
        }, 200);
    }

    async fetchPredictions() {
        try {
            const response = await fetch('/predictions');
            const predictions = await response.json();
            this.predictions = Array.isArray(predictions) ? predictions : [];
            this.updatePredictionsUI();
        } catch (error) {
            console.error('Error fetching predictions:', error);
        }
    }

    async fetchSelectionStatus() {
        try {
            const response = await fetch('/selection_status');
            const selection = await response.json();
            this.selection = selection || {};

            if (Array.isArray(selection.sentence)) {
                this.sentence = selection.sentence;
                this.updateSentenceUI();
            }

            if (Array.isArray(selection.predictions)) {
                this.predictions = selection.predictions;
                this.updatePredictionsUI();
            }

            this.updateSelectionUI();
        } catch (error) {
            console.error('Error fetching selection status:', error);
        }
    }

    updatePredictionsUI() {
        const predictions = this.predictions;

        document.querySelectorAll('.prediction-item').forEach((item) => {
            item.classList.remove('selection-target');
        });

        for (let i = 0; i < 3; i++) {
            const labelEl = document.getElementById(`pred${i + 1}Label`);
            const barEl = document.getElementById(`pred${i + 1}Bar`);
            const confEl = document.getElementById(`pred${i + 1}Conf`);

            if (predictions[i]) {
                labelEl.textContent = `${i + 1}. ${predictions[i].label_tr}`;
                barEl.style.width = `${predictions[i].confidence}%`;
                confEl.textContent = `%${predictions[i].confidence}`;
            } else {
                labelEl.textContent = `${i + 1}. -`;
                barEl.style.width = '0%';
                confEl.textContent = '0%';
            }
        }
    }

    updateSelectionUI() {
        const selection = this.selection || {};
        const active = Boolean(selection.active);
        const selected = selection.last_selected || null;

        let statusText = 'Tahmin geldikten sonra 1, 2 veya 3 goster.';
        if (active) {
            statusText = 'Top-3 adayi secmek icin 1, 2 veya 3 goster.';
        } else if (selection.last_event === 'selected' && selected) {
            statusText = `${selected.digit_value} ile secildi: ${selected.candidate.label_tr}`;
        } else if (selection.last_event === 'timeout') {
            statusText = 'Secim zamani doldu. Yeni tahmini bekliyor.';
        } else if (selection.last_event === 'cancelled') {
            statusText = 'Secim iptal edildi.';
        }

        const detectedParts = [];
        if (selection.last_digit_value) {
            detectedParts.push(String(selection.last_digit_value));
        } else if (selection.last_digit_label === 'other_digit') {
            detectedParts.push('DIGER');
        } else {
            detectedParts.push('-');
        }
        if (typeof selection.last_confidence === 'number' && selection.last_confidence > 0) {
            detectedParts.push(`%${selection.last_confidence.toFixed(1)}`);
        }

        let chosenText = '-';
        if (selected) {
            chosenText = `${selected.digit_value} -> ${selected.candidate.label_tr} (%${selected.confidence})`;
        }

        document.getElementById('selectionStatus').textContent = statusText;
        document.getElementById('selectionCountdown').textContent =
            active ? `${(selection.remaining_ms / 1000).toFixed(1)} sn` : '-';
        document.getElementById('selectionDetected').textContent = detectedParts.join(' | ');
        document.getElementById('selectionChosen').textContent = chosenText;

        document.querySelectorAll('.prediction-item').forEach((item) => {
            item.classList.remove('selection-target');
        });

        const highlightDigit = active ? selection.stable_digit || selection.last_digit_value : null;
        if (Number.isInteger(highlightDigit) && highlightDigit >= 1 && highlightDigit <= 3) {
            const target = document.querySelector(`.prediction-item[data-index="${highlightDigit - 1}"]`);
            if (target) {
                target.classList.add('selection-target');
            }
        }
    }

    async clearPredictions() {
        try {
            await fetch('/clear_predictions', { method: 'POST' });
            this.predictions = [];
            this.updatePredictionsUI();
            this.selection = {};
            this.updateSelectionUI();
        } catch (error) {
            console.error('Error clearing predictions:', error);
        }
    }

    async addWord(word) {
        try {
            const response = await fetch('/add_word', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ word })
            });

            const data = await response.json();
            this.sentence = data.sentence;
            this.updateSentenceUI();
            this.showWordAdded(word);
        } catch (error) {
            console.error('Error adding word:', error);
        }
    }

    async removeWord(index) {
        try {
            const response = await fetch('/remove_word', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ index })
            });

            const data = await response.json();
            this.sentence = data.sentence;
            this.updateSentenceUI();
        } catch (error) {
            console.error('Error removing word:', error);
        }
    }

    async clearSentence() {
        try {
            const response = await fetch('/clear_sentence', {
                method: 'POST'
            });

            const data = await response.json();
            this.sentence = data.sentence;
            this.updateSentenceUI();
        } catch (error) {
            console.error('Error clearing sentence:', error);
        }
    }

    async loadSentence() {
        try {
            const response = await fetch('/get_sentence');
            const data = await response.json();
            this.sentence = data.sentence;
            this.updateSentenceUI();
        } catch (error) {
            console.error('Error loading sentence:', error);
        }
    }

    speakSentence() {
        if (!this.sentence || this.sentence.length === 0) return;

        const textToSpeak = this.sentence.join(' ');

        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();

            const utterance = new SpeechSynthesisUtterance(textToSpeak);
            utterance.lang = 'tr-TR';
            utterance.rate = 0.75;
            utterance.pitch = 0.96;
            utterance.volume = 1.0;

            if (!this.preferredVoice) {
                this.preferredVoice = this.selectPreferredTurkishVoice(
                    window.speechSynthesis.getVoices()
                );
            }

            if (this.preferredVoice) {
                utterance.voice = this.preferredVoice;
                utterance.lang = this.preferredVoice.lang || 'tr-TR';
            }

            window.speechSynthesis.speak(utterance);
        } else {
            console.warn('Tarayici seslendirme ozelligini desteklemiyor.');
            alert('Tarayici seslendirme ozelligini desteklemiyor.');
        }
    }

    updateSentenceUI() {
        const container = document.getElementById('sentenceContainer');

        if (this.sentence.length === 0) {
            container.innerHTML = '<span class="empty-sentence">Kelime eklemek icin tahmine tiklayin veya 1/2/3 gosterin</span>';
            return;
        }

        container.innerHTML = this.sentence.map((word, index) => `
            <div class="sentence-word" onclick="app.removeWord(${index})">
                <span>${word}</span>
                <span class="remove-btn">×</span>
            </div>
        `).join('');
    }

    showWordAdded() {
        const container = document.getElementById('sentenceContainer');
        container.style.animation = 'none';
        container.offsetHeight;
        container.style.animation = 'fadeIn 0.3s ease';
    }
}

let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new TIDApp();
});
