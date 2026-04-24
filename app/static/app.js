/**
 * TID Recognition System - Frontend JavaScript
 * Handles real-time predictions, sentence building, and UI updates
 */

class TIDApp {
    constructor() {
        this.predictions = [];
        this.sentence = [];
        this.debug = {};
        this.pollInterval = null;
        
        this.init();
    }
    
    init() {
        // Bind event listeners
        this.bindEvents();
        
        // Start polling for predictions
        this.startPolling();
        
        // Load initial sentence
        this.loadSentence();
        
        // Hide video overlay when video loads
        const videoFeed = document.getElementById('videoFeed');
        videoFeed.onload = () => {
            document.getElementById('videoOverlay').classList.remove('active');
        };
        videoFeed.onerror = () => {
            document.getElementById('videoOverlay').classList.add('active');
        };
    }
    
    bindEvents() {
        // Prediction items click
        document.querySelectorAll('.prediction-item').forEach((item) => {
            item.addEventListener('click', () => {
                const index = parseInt(item.dataset.index);
                if (this.predictions[index]) {
                    this.addWord(this.predictions[index].label_tr);
                }
            });
        });
        
        // Clear predictions button
        document.getElementById('clearPredictions').addEventListener('click', () => {
            this.clearPredictions();
        });
        
        // Clear sentence button
        document.getElementById('clearSentence').addEventListener('click', () => {
            this.clearSentence();
        });

        // Speak sentence button
        document.getElementById('speakSentence').addEventListener('click', () => {
            this.speakSentence();
        });
    }
    
    startPolling() {
        // Poll for predictions every 200ms
        this.pollInterval = setInterval(() => {
            this.fetchPredictions();
            this.fetchDebugStatus();
        }, 200);
    }
    
    async fetchPredictions() {
        try {
            const response = await fetch('/predictions');
            const predictions = await response.json();
            
            if (predictions && predictions.length > 0) {
                this.predictions = predictions;
                this.updatePredictionsUI();
            }
        } catch (error) {
            console.error('Error fetching predictions:', error);
        }
    }

    async fetchDebugStatus() {
        try {
            const response = await fetch('/debug_status');
            const debug = await response.json();

            if (debug && Object.keys(debug).length > 0) {
                this.debug = debug;
                this.updateDebugUI();
            }
        } catch (error) {
            console.error('Error fetching debug status:', error);
        }
    }
    
    updatePredictionsUI() {
        const predictions = this.predictions;
        
        for (let i = 0; i < 3; i++) {
            const labelEl = document.getElementById(`pred${i + 1}Label`);
            const barEl = document.getElementById(`pred${i + 1}Bar`);
            const confEl = document.getElementById(`pred${i + 1}Conf`);
            
            if (predictions[i]) {
                labelEl.textContent = predictions[i].label_tr.toUpperCase();
                barEl.style.width = `${predictions[i].confidence}%`;
                confEl.textContent = `%${predictions[i].confidence}`;
            } else {
                labelEl.textContent = '-';
                barEl.style.width = '0%';
                confEl.textContent = '0%';
            }
        }
    }

    updateDebugUI() {
        const debug = this.debug || {};
        const lastDecisionParts = [];

        if (debug.last_label && debug.last_label !== '-') {
            lastDecisionParts.push(debug.last_label.toUpperCase());
        }
        if (typeof debug.last_confidence === 'number' && debug.last_confidence > 0) {
            lastDecisionParts.push(`%${debug.last_confidence.toFixed(1)}`);
        }
        if (debug.last_ambiguous) {
            lastDecisionParts.push('ambiguous');
        }
        if (debug.last_low_confidence) {
            lastDecisionParts.push('low_conf');
        }

        document.getElementById('debugState').textContent = debug.state || '-';
        document.getElementById('debugMotion').textContent = this.formatNumber(debug.motion, 4);
        document.getElementById('debugCollected').textContent = this.formatInteger(debug.collected_frames);
        document.getElementById('debugStart').textContent = this.formatInteger(debug.signing_frames);
        document.getElementById('debugIdle').textContent = this.formatInteger(debug.idle_frames);
        document.getElementById('debugCooldown').textContent = this.formatInteger(debug.cooldown_counter);
        document.getElementById('debugHandVisible').textContent = debug.hand_visible ? 'evet' : 'hayir';
        document.getElementById('debugHandFrames').textContent =
            `${this.formatInteger(debug.hand_frames)} / ${this.formatInteger(debug.min_hand_frames)}`;
        document.getElementById('debugLastDecision').textContent =
            lastDecisionParts.length > 0 ? lastDecisionParts.join(' | ') : '-';
        document.getElementById('debugReason').textContent =
            [debug.last_event, debug.last_reason].filter(Boolean).join(' | ') || '-';
    }

    formatNumber(value, decimals = 1) {
        return typeof value === 'number' ? value.toFixed(decimals) : '-';
    }

    formatInteger(value) {
        return Number.isFinite(value) ? String(value) : '0';
    }
    
    clearPredictions() {
        this.predictions = [];
        this.updatePredictionsUI();
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
            
            // Visual feedback
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
        
        // Cümleyi birleştir (kelimeler arası boşluk bırak)
        const textToSpeak = this.sentence.map(w => w.toUpperCase()).join(' ');
        
        // Web Speech API kullan (Tarayıcı destekliyorsa)
        if ('speechSynthesis' in window) {
            // Önceki konuşmaları iptal et
            window.speechSynthesis.cancel();
            
            const utterance = new SpeechSynthesisUtterance(textToSpeak);
            utterance.lang = 'tr-TR';
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            
            window.speechSynthesis.speak(utterance);
        } else {
            console.warn('Tarayıcınız Text-to-Speech (Sese Dönüştürme) özelliğini desteklemiyor.');
            alert('Tarayıcınız seslendirme özelliğini desteklemiyor.');
        }
    }
    
    updateSentenceUI() {
        const container = document.getElementById('sentenceContainer');
        
        if (this.sentence.length === 0) {
            container.innerHTML = '<span class="empty-sentence">Kelime eklemek için yukarıdaki tahminlere tıklayın</span>';
            return;
        }
        
        container.innerHTML = this.sentence.map((word, index) => `
            <div class="sentence-word" onclick="app.removeWord(${index})">
                <span>${word.toUpperCase()}</span>
                <span class="remove-btn">×</span>
            </div>
        `).join('');
    }
    
    showWordAdded(word) {
        // Flash animation on the sentence container
        const container = document.getElementById('sentenceContainer');
        container.style.animation = 'none';
        container.offsetHeight; // Trigger reflow
        container.style.animation = 'fadeIn 0.3s ease';
    }
}

// Initialize app when DOM is loaded
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new TIDApp();
});
