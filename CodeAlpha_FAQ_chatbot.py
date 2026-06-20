# complete_faq_chatbot.py
# A complete FAQ Chatbot with automatic dependency installation

import subprocess
import sys
import importlib
import os

def install_packages():
    """Automatically install required packages"""
    required_packages = {
        'numpy': 'numpy',
        'scikit-learn': 'scikit-learn',
        'nltk': 'nltk',
        'flask': 'flask'
    }
    
    print("📦 Checking and installing required packages...")
    
    for package_name, import_name in required_packages.items():
        try:
            importlib.import_module(import_name)
            print(f"✅ {package_name} is already installed")
        except ImportError:
            print(f"📥 Installing {package_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"✅ {package_name} installed successfully")
    
    print("🎉 All packages are ready!\n")

# Install packages before importing
install_packages()

# Now import all required libraries
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import json
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data (only once)
def download_nltk_data():
    """Download required NLTK data"""
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir, exist_ok=True)
    
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("📥 Downloading NLTK tokenizer data...")
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("📥 Downloading NLTK stopwords data...")
        nltk.download('stopwords', quiet=True)
    
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("📥 Downloading NLTK punkt_tab data...")
        nltk.download('punkt_tab', quiet=True)

download_nltk_data()

class CompleteFAQChatbot:
    """Complete FAQ Chatbot with product and general knowledge"""
    
    def __init__(self, use_web_ui=False):
        """Initialize the chatbot"""
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Load all FAQs
        self.faqs = self.load_all_faqs()
        
        self.questions = [item['question'] for item in self.faqs]
        self.answers = [item['answer'] for item in self.faqs]
        
        # Preprocess all questions
        self.processed_questions = [self.preprocess_text(q) for q in self.questions]
        
        # Initialize TF-IDF vectorizer (using unigrams and bigrams)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.processed_questions)
        
        self.use_web_ui = use_web_ui
    
    def load_all_faqs(self):
        """Load all FAQs (product + general knowledge)"""
        faqs = []
        
        # Product FAQs
        product_faqs = [
            {
                "question": "What is the battery life of the device?",
                "answer": "The Smart Home Assistant has a battery life of up to 12 hours on a full charge with normal usage. Heavy usage (constant voice commands, music) may reduce battery to 8 hours."
            },
            {
                "question": "How do I connect to Wi-Fi?",
                "answer": "To connect to Wi-Fi: Go to Settings > Network > Wi-Fi, select your network, and enter your password. The device supports both 2.4GHz and 5GHz networks."
            },
            {
                "question": "Is the device waterproof?",
                "answer": "Yes, the device has an IP67 rating, making it dust-tight and water-resistant up to 1 meter for 30 minutes. However, it's not designed for swimming or showering."
            },
            {
                "question": "What is your return policy?",
                "answer": "We offer a 30-day money-back guarantee for unused products in original packaging. Contact customer support for return authorization and shipping instructions."
            },
            {
                "question": "How do I update the firmware?",
                "answer": "Firmware updates are automatic when connected to Wi-Fi. You can also check manually in Settings > System > Update. The process takes about 5-10 minutes."
            },
            {
                "question": "Does it work with Alexa or Google Home?",
                "answer": "Yes, the device is compatible with both Amazon Alexa and Google Assistant for seamless voice control. You can link it through the respective apps."
            },
            {
                "question": "What is included in the box?",
                "answer": "The box includes: Smart Home Assistant device, USB-C charging cable, power adapter, quick start guide, warranty card, and a complimentary screen protector."
            },
            {
                "question": "How do I reset the device?",
                "answer": "To factory reset: Press and hold the power button for 10 seconds until the LED flashes red, then release. This will erase all settings and data."
            },
            {
                "question": "What is the screen size?",
                "answer": "The device features a 7-inch HD display with 1024x600 resolution, offering clear and vibrant visuals for all your needs."
            },
            {
                "question": "Does it have a camera?",
                "answer": "Yes, it includes a 5MP front-facing camera for video calls and security monitoring. The camera has a physical privacy shutter."
            }
        ]
        
        # General Knowledge FAQs
        general_faqs = [
            {
                "question": "What is Java?",
                "answer": "Java is a high-level, object-oriented programming language developed by Sun Microsystems in 1995. It follows 'write once, run anywhere' principle using Java Virtual Machine (JVM)."
            },
            {
                "question": "What is Python?",
                "answer": "Python is an interpreted, high-level programming language created by Guido van Rossum in 1991. It's known for its simple, readable syntax and extensive libraries for data science, web development, and AI."
            },
            {
                "question": "What is machine learning?",
                "answer": "Machine Learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to find patterns in data."
            },
            {
                "question": "What is artificial intelligence?",
                "answer": "Artificial Intelligence (AI) is the simulation of human intelligence in machines. It includes learning, reasoning, problem-solving, perception, and language understanding."
            },
            {
                "question": "What is NLP?",
                "answer": "Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human language. Examples include chatbots, translation, and sentiment analysis."
            },
            {
                "question": "What is deep learning?",
                "answer": "Deep Learning is a subset of machine learning using neural networks with multiple layers. It excels at tasks like image recognition, speech recognition, and natural language processing."
            },
            {
                "question": "What is the difference between AI and ML?",
                "answer": "AI is the broader concept of machines being able to perform tasks smartly, while ML is a subset of AI that allows systems to learn from data without explicit programming."
            },
            {
                "question": "What is TensorFlow?",
                "answer": "TensorFlow is an open-source machine learning framework developed by Google. It's widely used for building and training deep learning models."
            },
            {
                "question": "What is PyTorch?",
                "answer": "PyTorch is an open-source machine learning framework developed by Facebook's AI Research lab. It's known for its dynamic computation graphs and Python-friendly interface."
            },
            {
                "question": "How does cosine similarity work?",
                "answer": "Cosine similarity measures the cosine of the angle between two vectors in a multi-dimensional space. Values range from -1 to 1, where 1 means identical vectors."
            }
        ]
        
        faqs.extend(product_faqs)
        faqs.extend(general_faqs)
        
        return faqs
    
    def preprocess_text(self, text):
        """Enhanced preprocessing that keeps important words"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation but keep spaces
        text = re.sub(r'[^\w\s]', '', text)
        
        # Tokenize
        tokens = nltk.word_tokenize(text)
        
        # Important words to always keep
        important_words = {'what', 'how', 'is', 'does', 'can', 'do', 'are', 'which', 
                          'where', 'when', 'why', 'tell', 'explain', 'define'}
        
        processed_tokens = []
        for token in tokens:
            if token in important_words or (token not in self.stop_words and len(token) > 1):
                processed_tokens.append(self.stemmer.stem(token))
        
        return ' '.join(processed_tokens)
    
    def get_response(self, user_question, threshold=0.15):
        """Get best matching answer for user's question"""
        # Preprocess user question
        processed_user = self.preprocess_text(user_question)
        
        # Transform user question
        user_tfidf = self.vectorizer.transform([processed_user])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(user_tfidf, self.tfidf_matrix).flatten()
        
        # Get best match
        best_match_idx = np.argmax(similarities)
        best_score = similarities[best_match_idx]
        
        # Check if match meets threshold
        if best_score >= threshold:
            return {
                "answer": self.answers[best_match_idx],
                "confidence": round(float(best_score), 3),
                "matched_question": self.questions[best_match_idx],
                "success": True
            }
        else:
            # Helpful fallback response
            return {
                "answer": f"I'm not sure about that. I can help with questions about:\n\n" +
                         f"📱 PRODUCT: battery, Wi-Fi, waterproof, reset, camera, screen size\n" +
                         f"💻 PROGRAMMING: Java, Python, TensorFlow, PyTorch\n" +
                         f"🤖 AI/ML: AI, machine learning, deep learning, NLP\n\n" +
                         f"Try rephrasing your question or ask 'faqs' to see all available topics!",
                "confidence": round(float(best_score), 3),
                "success": False
            }
    
    def find_similar_questions(self, user_question, top_k=3):
        """Find top K most similar questions"""
        processed_user = self.preprocess_text(user_question)
        user_tfidf = self.vectorizer.transform([processed_user])
        similarities = cosine_similarity(user_tfidf, self.tfidf_matrix).flatten()
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                "question": self.questions[idx],
                "similarity": round(float(similarities[idx]), 3)
            })
        return results
    
    def add_faq(self, question, answer):
        """Add new FAQ dynamically"""
        self.questions.append(question)
        self.answers.append(answer)
        processed = self.preprocess_text(question)
        self.processed_questions.append(processed)
        
        # Rebuild TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(self.processed_questions)
        print(f"✅ Added new FAQ: {question}")
    
    def display_all_faqs(self):
        """Display all available FAQs"""
        print("\n" + "="*70)
        print("📚 ALL AVAILABLE FAQS (I can answer these questions)")
        print("="*70)
        
        print("\n🔹 PRODUCT FAQs:")
        for i, (q, a) in enumerate(zip(self.questions[:10], self.answers[:10]), 1):
            print(f"   {i}. {q}")
        
        print("\n🔹 GENERAL KNOWLEDGE FAQs:")
        for i, (q, a) in enumerate(zip(self.questions[10:], self.answers[10:]), 1):
            print(f"   {i}. {q}")
        
        print("\n" + "="*70)
        print(f"Total FAQs: {len(self.questions)}")
        print("="*70)
    
    def chat_console(self):
        """Run chatbot in console mode"""
        print("\n" + "="*70)
        print("🤖 COMPLETE FAQ CHATBOT - Smart Assistant")
        print("="*70)
        print("\n📌 I can answer questions about:")
        print("   • Smart Home Assistant (battery, Wi-Fi, reset, camera, etc.)")
        print("   • Programming languages (Java, Python, etc.)")
        print("   • AI/ML concepts (AI, ML, Deep Learning, NLP)")
        print("   • Frameworks (TensorFlow, PyTorch)")
        print("-"*70)
        print("💡 Commands:")
        print("   • 'quit' - Exit the chatbot")
        print("   • 'faqs' - Show all questions I can answer")
        print("   • 'add' - Add a new FAQ")
        print("-"*70)
        
        while True:
            user_input = input("\n🧑 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n🤖 Chatbot: Thank you for chatting! Have a great day! 👋")
                break
            
            if user_input.lower() == 'faqs':
                self.display_all_faqs()
                continue
            
            if user_input.lower() == 'add':
                q = input("📝 Enter new question: ").strip()
                a = input("📝 Enter answer: ").strip()
                self.add_faq(q, a)
                print("✅ New FAQ added successfully!")
                continue
            
            if not user_input:
                continue
            
            # Get response
            response = self.get_response(user_input)
            
            print(f"\n🤖 Chatbot: {response['answer']}")
            if response['success']:
                print(f"   📊 Confidence: {response['confidence']*100:.1f}%")
                print(f"   ❓ Matched: {response['matched_question']}")
            else:
                # Show suggestions
                similar = self.find_similar_questions(user_input, top_k=2)
                if similar:
                    print(f"\n   💡 Did you mean one of these?")
                    for s in similar:
                        print(f"      • {s['question']} (similarity: {s['similarity']})")
    
    def run_web_ui(self, port=5000):
        """Run the Flask web UI"""
        try:
            from flask import Flask, render_template_string, request, jsonify
            
            app = Flask(__name__)
            
            # HTML template for web UI
            html_template = '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>FAQ Chatbot</title>
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    * { margin: 0; padding: 0; box-sizing: border-box; }
                    
                    body {
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        height: 100vh;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                    }
                    
                    .chat-container {
                        width: 90%;
                        max-width: 900px;
                        height: 85vh;
                        background: white;
                        border-radius: 20px;
                        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                        display: flex;
                        flex-direction: column;
                        overflow: hidden;
                    }
                    
                    .chat-header {
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 20px;
                        text-align: center;
                    }
                    
                    .chat-header h1 { font-size: 1.5em; margin-bottom: 5px; }
                    .chat-header p { font-size: 0.9em; opacity: 0.9; }
                    
                    .chat-messages {
                        flex: 1;
                        overflow-y: auto;
                        padding: 20px;
                        background: #f5f5f5;
                    }
                    
                    .message {
                        margin-bottom: 15px;
                        display: flex;
                        animation: fadeIn 0.3s ease;
                    }
                    
                    @keyframes fadeIn {
                        from { opacity: 0; transform: translateY(10px); }
                        to { opacity: 1; transform: translateY(0); }
                    }
                    
                    .message.user { justify-content: flex-end; }
                    .message.bot { justify-content: flex-start; }
                    
                    .message-content {
                        max-width: 70%;
                        padding: 12px 16px;
                        border-radius: 18px;
                        word-wrap: break-word;
                    }
                    
                    .user .message-content {
                        background: #667eea;
                        color: white;
                        border-bottom-right-radius: 4px;
                    }
                    
                    .bot .message-content {
                        background: white;
                        color: #333;
                        border-bottom-left-radius: 4px;
                        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
                    }
                    
                    .confidence {
                        font-size: 0.7em;
                        margin-top: 8px;
                        padding-top: 5px;
                        border-top: 1px solid rgba(0,0,0,0.1);
                        opacity: 0.7;
                    }
                    
                    .chat-input-area {
                        padding: 20px;
                        background: white;
                        border-top: 1px solid #e0e0e0;
                        display: flex;
                        gap: 10px;
                    }
                    
                    .chat-input {
                        flex: 1;
                        padding: 12px;
                        border: 1px solid #ddd;
                        border-radius: 25px;
                        font-size: 1em;
                        outline: none;
                    }
                    
                    .chat-input:focus { border-color: #667eea; }
                    
                    .send-button {
                        padding: 12px 24px;
                        background: #667eea;
                        color: white;
                        border: none;
                        border-radius: 25px;
                        cursor: pointer;
                        font-size: 1em;
                    }
                    
                    .send-button:hover { background: #5a67d8; }
                    
                    .suggestions {
                        display: flex;
                        flex-wrap: wrap;
                        gap: 8px;
                        padding: 10px 20px;
                        background: #f9f9f9;
                        border-top: 1px solid #e0e0e0;
                    }
                    
                    .suggestion-chip {
                        padding: 6px 12px;
                        background: #e0e0e0;
                        border-radius: 20px;
                        font-size: 0.85em;
                        cursor: pointer;
                        transition: all 0.3s;
                    }
                    
                    .suggestion-chip:hover {
                        background: #667eea;
                        color: white;
                    }
                    
                    @media (max-width: 600px) {
                        .message-content { max-width: 85%; }
                        .chat-header h1 { font-size: 1.2em; }
                    }
                </style>
            </head>
            <body>
                <div class="chat-container">
                    <div class="chat-header">
                        <h1>🤖 Complete FAQ Assistant</h1>
                        <p>Ask me about products, programming, or AI!</p>
                    </div>
                    
                    <div class="chat-messages" id="chatMessages">
                        <div class="message bot">
                            <div class="message-content">
                                Hello! 👋 I'm your FAQ assistant. I can help with:<br>
                                • 📱 Product questions (battery, Wi-Fi, reset)<br>
                                • 💻 Programming (Java, Python)<br>
                                • 🤖 AI/ML concepts (AI, ML, NLP)<br><br>
                                What would you like to know?
                            </div>
                        </div>
                    </div>
                    
                    <div class="suggestions" id="suggestions">
                        <div class="suggestion-chip">What is Java?</div>
                        <div class="suggestion-chip">What is Python?</div>
                        <div class="suggestion-chip">Battery life?</div>
                        <div class="suggestion-chip">How to connect to Wi-Fi?</div>
                        <div class="suggestion-chip">What is AI?</div>
                        <div class="suggestion-chip">Reset device</div>
                    </div>
                    
                    <div class="chat-input-area">
                        <input type="text" class="chat-input" id="userInput" placeholder="Type your question here..." onkeypress="handleKeyPress(event)">
                        <button class="send-button" onclick="sendMessage()">Send</button>
                    </div>
                </div>
                
                <script>
                    const chatMessages = document.getElementById('chatMessages');
                    const userInput = document.getElementById('userInput');
                    
                    function addMessage(text, isUser, confidence = null, matchedQuestion = null) {
                        const messageDiv = document.createElement('div');
                        messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
                        
                        const contentDiv = document.createElement('div');
                        contentDiv.className = 'message-content';
                        contentDiv.innerHTML = text.replace(/\\n/g, '<br>');
                        
                        if (!isUser && confidence) {
                            const confidenceSpan = document.createElement('div');
                            confidenceSpan.className = 'confidence';
                            confidenceSpan.innerHTML = `📊 ${Math.round(confidence * 100)}% confidence`;
                            if (matchedQuestion) {
                                confidenceSpan.innerHTML += `<br>❓ Matched: "${matchedQuestion}"`;
                            }
                            contentDiv.appendChild(confidenceSpan);
                        }
                        
                        messageDiv.appendChild(contentDiv);
                        chatMessages.appendChild(messageDiv);
                        chatMessages.scrollTop = chatMessages.scrollHeight;
                    }
                    
                    async function sendMessage() {
                        const message = userInput.value.trim();
                        if (!message) return;
                        
                        addMessage(message, true);
                        userInput.value = '';
                        
                        // Typing indicator
                        const typingDiv = document.createElement('div');
                        typingDiv.className = 'message bot';
                        typingDiv.id = 'typingIndicator';
                        typingDiv.innerHTML = '<div class="message-content">🤖 Typing...</div>';
                        chatMessages.appendChild(typingDiv);
                        chatMessages.scrollTop = chatMessages.scrollHeight;
                        
                        try {
                            const response = await fetch('/chat', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ message: message })
                            });
                            const data = await response.json();
                            
                            document.getElementById('typingIndicator')?.remove();
                            addMessage(data.answer, false, data.confidence, data.matched_question);
                        } catch (error) {
                            document.getElementById('typingIndicator')?.remove();
                            addMessage("Sorry, I'm having trouble connecting. Please try again.", false);
                        }
                    }
                    
                    function handleKeyPress(event) {
                        if (event.key === 'Enter') sendMessage();
                    }
                    
                    // Add click handlers for suggestion chips
                    document.querySelectorAll('.suggestion-chip').forEach(chip => {
                        chip.addEventListener('click', () => {
                            userInput.value = chip.textContent;
                            sendMessage();
                        });
                    });
                </script>
            </body>
            </html>
            '''
            
            @app.route('/')
            def index():
                return render_template_string(html_template)
            
            @app.route('/chat', methods=['POST'])
            def chat():
                data = request.json
                user_message = data.get('message', '')
                response = self.get_response(user_message)
                return jsonify(response)
            
            print("\n" + "="*50)
            print("🌐 STARTING WEB INTERFACE")
            print("="*50)
            print(f"📱 Open your browser and go to: http://localhost:{port}")
            print("🛑 Press Ctrl+C to stop the server")
            print("="*50 + "\n")
            
            app.run(debug=False, port=port, host='0.0.0.0')
            
        except ImportError:
            print("❌ Flask not installed. Installing now...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "flask"])
            print("✅ Flask installed! Please restart the chatbot.")
        except Exception as e:
            print(f"❌ Error starting web UI: {e}")
            print("💡 You can still use the console interface.")


def main():
    """Main function with menu"""
    print("\n" + "="*60)
    print("🤖 COMPLETE FAQ CHATBOT SYSTEM")
    print("="*60)
    print("\n🚀 Initializing chatbot...")
    
    chatbot = CompleteFAQChatbot()
    
    while True:
        print("\n" + "="*60)
        print("📋 MAIN MENU")
        print("="*60)
        print("1. 💬 Console Chat Interface")
        print("2. 🌐 Web UI Interface (Flask)")
        print("3. 📚 View All FAQs")
        print("4. ➕ Add New FAQ")
        print("5. 🔍 Test Single Question")
        print("6. 🚪 Exit")
        print("-"*60)
        
        choice = input("👉 Enter your choice (1-6): ").strip()
        
        if choice == '1':
            chatbot.chat_console()
        elif choice == '2':
            chatbot.run_web_ui()
        elif choice == '3':
            chatbot.display_all_faqs()
        elif choice == '4':
            print("\n📝 ADD NEW FAQ")
            question = input("Question: ").strip()
            answer = input("Answer: ").strip()
            if question and answer:
                chatbot.add_faq(question, answer)
            else:
                print("❌ Both question and answer are required!")
        elif choice == '5':
            print("\n🔍 TEST SINGLE QUESTION")
            question = input("Enter your question: ").strip()
            if question:
                response = chatbot.get_response(question)
                print(f"\n📝 Answer: {response['answer']}")
                if response['success']:
                    print(f"📊 Confidence: {response['confidence']*100:.1f}%")
                    print(f"❓ Matched with: {response['matched_question']}")
                
                # Show similar questions
                similar = chatbot.find_similar_questions(question, top_k=3)
                if similar:
                    print(f"\n🔍 Other similar questions found:")
                    for s in similar:
                        print(f"   • {s['question']} (score: {s['similarity']})")
        elif choice == '6':
            print("\n👋 Thank you for using the FAQ Chatbot! Goodbye!")
            break
        else:
            print("❌ Invalid choice! Please enter a number between 1 and 6.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("💡 Please try restarting the chatbot.")