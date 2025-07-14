import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tag import pos_tag
import re
import json
import pickle
import random
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

class TraditionalMLChatbot:
    """
    Traditional ML chatbot using classical NLP and machine learning techniques
    """
    
    def __init__(self):
        # Text preprocessing tools
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8
        )
        
        # Multiple classifiers for ensemble
        self.intent_classifier = LogisticRegression(
            random_state=42,
            max_iter=1000
        )
        self.sentiment_classifier = MultinomialNB()
        self.topic_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        
        # NLP tools
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Pipeline for intent classification
        self.intent_pipeline = Pipeline([
            ('tfidf', self.vectorizer),
            ('classifier', self.intent_classifier)
        ])
        
        # Knowledge base and similarity matching
        self.knowledge_base = []
        self.knowledge_vectors = None
        self.kb_vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=3000
        )
        
        # Conversation management
        self.conversation_history = []
        self.user_profile = {
            'name': None,
            'preferences': [],
            'interaction_count': 0,
            'common_topics': Counter()
        }
        
        # Rule-based patterns
        self.patterns = self._initialize_patterns()
        
        # Response templates
        self.response_templates = self._initialize_response_templates()
        
        # Entity extraction patterns
        self.entity_patterns = {
            'name': re.compile(r'(?:my name is|i am|i\'m|call me)\s+([a-zA-Z]+)', re.IGNORECASE),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            'date': re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'),
            'time': re.compile(r'\b\d{1,2}:\d{2}\s*(?:AM|PM)?\b', re.IGNORECASE)
        }
        
        # Trained models storage
        self.models_trained = False
    
    def _initialize_patterns(self) -> Dict:
        """Initialize rule-based patterns for quick responses"""
        return {
            'greeting': [
                r'\b(hi|hello|hey|good morning|good afternoon|good evening)\b',
                r'\bhow are you\b',
                r'\bwhat\'s up\b'
            ],
            'goodbye': [
                r'\b(bye|goodbye|see you|farewell|talk to you later)\b',
                r'\bthanks?\s*(you)?\s*(?:very\s*much)?\b',
                r'\bhave a good day\b'
            ],
            'question': [
                r'\b(what|how|when|where|why|who|which)\b',
                r'\bcan you\b',
                r'\bdo you know\b',
                r'\btell me about\b'
            ],
            'request': [
                r'\bplease\b',
                r'\bcan you help\b',
                r'\bi need\b',
                r'\bcould you\b'
            ],
            'affirmation': [
                r'\b(yes|yeah|yep|sure|okay|ok|alright)\b',
                r'\bthat\'s right\b',
                r'\bexactly\b'
            ],
            'negation': [
                r'\b(no|nope|not really|i don\'t think so)\b',
                r'\bthat\'s wrong\b',
                r'\bi disagree\b'
            ]
        }
    
    def _initialize_response_templates(self) -> Dict:
        """Initialize response templates for different intents"""
        return {
            'greeting': [
                "Hello! How can I help you today?",
                "Hi there! What can I do for you?",
                "Welcome! How may I assist you?",
                "Good to see you! How can I help?",
                "Hello! I'm here to help. What do you need?"
            ],
            'goodbye': [
                "Goodbye! Have a great day!",
                "See you later! Feel free to come back anytime.",
                "Thanks for chatting! Take care!",
                "Farewell! Hope I was helpful.",
                "Bye! Don't hesitate to return if you need help."
            ],
            'question': [
                "That's an interesting question. Let me think about that.",
                "I'll do my best to help you with that.",
                "Let me search my knowledge for that information.",
                "Good question! Here's what I know about that.",
                "I'll help you find the answer to that."
            ],
            'request': [
                "I'd be happy to help you with that.",
                "Let me assist you with that request.",
                "I'll do my best to help you.",
                "Sure, I can help you with that.",
                "I'm here to help! Let me see what I can do."
            ],
            'unknown': [
                "I'm not sure I understand. Could you rephrase that?",
                "That's outside my current knowledge. Can you be more specific?",
                "I need more information to help you properly.",
                "Could you explain that differently? I want to help you correctly.",
                "I'm having trouble understanding. Can you clarify?"
            ],
            'clarification': [
                "Could you provide more details about that?",
                "What specifically would you like to know?",
                "Can you be more specific about your question?",
                "I need a bit more information to help you better.",
                "Could you elaborate on what you're looking for?"
            ]
        }
    
    def preprocess_text(self, text: str) -> str:
        """Advanced text preprocessing"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs, emails (but save them for entity extraction)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Handle contractions
        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^a-zA-Z\s\.\!\?]', '', text)
        
        # Tokenize and POS tag
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        
        # Keep only meaningful words (nouns, verbs, adjectives, adverbs)
        meaningful_pos = ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']
        
        processed_tokens = []
        for token, pos in pos_tags:
            if (token not in self.stop_words and 
                len(token) > 2 and 
                any(pos.startswith(mp) for mp in ['NN', 'VB', 'JJ', 'RB'])):
                # Lemmatize based on POS
                if pos.startswith('V'):
                    lemmatized = self.lemmatizer.lemmatize(token, 'v')
                elif pos.startswith('J'):
                    lemmatized = self.lemmatizer.lemmatize(token, 'a')
                elif pos.startswith('R'):
                    lemmatized = self.lemmatizer.lemmatize(token, 'r')
                else:
                    lemmatized = self.lemmatizer.lemmatize(token, 'n')
                
                processed_tokens.append(lemmatized)
        
        return ' '.join(processed_tokens)
    
    def extract_entities(self, text: str) -> Dict:
        """Extract entities from user input"""
        entities = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = pattern.findall(text)
            if matches:
                entities[entity_type] = matches
        
        return entities
    
    def detect_intent_by_patterns(self, text: str) -> Tuple[str, float]:
        """Rule-based intent detection using regex patterns"""
        text_lower = text.lower()
        
        for intent, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    # Calculate confidence based on pattern specificity
                    confidence = len(pattern) / 100.0  # Simple heuristic
                    return intent, min(confidence, 0.9)
        
        return 'unknown', 0.0
    
    def train_models(self, training_data: List[Dict]):
        """Train all ML models with provided data"""
        if not training_data:
            print("No training data provided")
            return
        
        # Prepare data
        texts = []
        intents = []
        sentiments = []
        topics = []
        
        for item in training_data:
            preprocessed = self.preprocess_text(item['text'])
            texts.append(preprocessed)
            intents.append(item.get('intent', 'unknown'))
            sentiments.append(item.get('sentiment', 'neutral'))
            topics.append(item.get('topic', 'general'))
        
        # Train intent classifier
        self.intent_pipeline.fit(texts, intents)
        
        # Train sentiment classifier
        sentiment_vectorizer = CountVectorizer(stop_words='english', max_features=2000)
        sentiment_features = sentiment_vectorizer.fit_transform(texts)
        self.sentiment_classifier.fit(sentiment_features, sentiments)
        self.sentiment_vectorizer = sentiment_vectorizer
        
        # Train topic classifier
        topic_vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
        topic_features = topic_vectorizer.fit_transform(texts)
        self.topic_classifier.fit(topic_features, topics)
        self.topic_vectorizer = topic_vectorizer
        
        # Evaluate models
        self._evaluate_models(texts, intents, sentiments, topics)
        
        self.models_trained = True
        print("All models trained successfully!")
    
    def _evaluate_models(self, texts, intents, sentiments, topics):
        """Evaluate trained models"""
        # Intent classification evaluation
        intent_scores = cross_val_score(self.intent_pipeline, texts, intents, cv=5)
        print(f"Intent Classification Accuracy: {intent_scores.mean():.3f} (+/- {intent_scores.std() * 2:.3f})")
        
        # Sentiment classification evaluation
        sentiment_features = self.sentiment_vectorizer.transform(texts)
        sentiment_scores = cross_val_score(self.sentiment_classifier, sentiment_features, sentiments, cv=5)
        print(f"Sentiment Classification Accuracy: {sentiment_scores.mean():.3f} (+/- {sentiment_scores.std() * 2:.3f})")
        
        # Topic classification evaluation
        topic_features = self.topic_vectorizer.transform(texts)
        topic_scores = cross_val_score(self.topic_classifier, topic_features, topics, cv=5)
        print(f"Topic Classification Accuracy: {topic_scores.mean():.3f} (+/- {topic_scores.std() * 2:.3f})")
    
    def build_knowledge_base(self, knowledge_data: List[Dict]):
        """Build knowledge base for FAQ-style responses"""
        self.knowledge_base = knowledge_data
        
        if not knowledge_data:
            return
        
        # Preprocess questions
        questions = [self.preprocess_text(item['question']) for item in knowledge_data]
        
        # Create TF-IDF vectors for similarity matching
        self.knowledge_vectors = self.kb_vectorizer.fit_transform(questions)
        print(f"Knowledge base built with {len(knowledge_data)} entries")
    
    def find_best_answer(self, text: str, threshold: float = 0.3) -> Optional[Dict]:
        """Find best answer from knowledge base using similarity"""
        if not self.knowledge_base or self.knowledge_vectors is None:
            return None
        
        preprocessed = self.preprocess_text(text)
        query_vector = self.kb_vectorizer.transform([preprocessed])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.knowledge_vectors)[0]
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        
        if best_similarity >= threshold:
            return {
                'question': self.knowledge_base[best_match_idx]['question'],
                'answer': self.knowledge_base[best_match_idx]['answer'],
                'similarity': best_similarity,
                'confidence': best_similarity
            }
        
        return None
    
    def predict_intent(self, text: str) -> Tuple[str, float]:
        """Predict intent using trained classifier"""
        if not self.models_trained:
            return self.detect_intent_by_patterns(text)
        
        try:
            preprocessed = self.preprocess_text(text)
            intent = self.intent_pipeline.predict([preprocessed])[0]
            probabilities = self.intent_pipeline.predict_proba([preprocessed])[0]
            confidence = max(probabilities)
            return intent, confidence
        except:
            return self.detect_intent_by_patterns(text)
    
    def predict_sentiment(self, text: str) -> Tuple[str, float]:
        """Predict sentiment of user input"""
        if not self.models_trained:
            return 'neutral', 0.5
        
        try:
            preprocessed = self.preprocess_text(text)
            features = self.sentiment_vectorizer.transform([preprocessed])
            sentiment = self.sentiment_classifier.predict(features)[0]
            probabilities = self.sentiment_classifier.predict_proba(features)[0]
            confidence = max(probabilities)
            return sentiment, confidence
        except:
            return 'neutral', 0.5
    
    def update_user_profile(self, text: str, intent: str, entities: Dict):
        """Update user profile based on conversation"""
        self.user_profile['interaction_count'] += 1
        self.user_profile['common_topics'][intent] += 1
        
        # Extract and store name
        if 'name' in entities and entities['name']:
            self.user_profile['name'] = entities['name'][0]
        
        # Store preferences (simple keyword extraction)
        keywords = self.preprocess_text(text).split()
        for keyword in keywords:
            if len(keyword) > 4:  # Longer words might be preferences
                self.user_profile['preferences'].append(keyword)
        
        # Keep only recent preferences
        if len(self.user_profile['preferences']) > 20:
            self.user_profile['preferences'] = self.user_profile['preferences'][-20:]
    
    def generate_response(self, text: str) -> str:
        """Generate response using traditional ML techniques"""
        # Preprocess input
        entities = self.extract_entities(text)
        intent, intent_confidence = self.predict_intent(text)
        sentiment, sentiment_confidence = self.predict_sentiment(text)
        
        # Update user profile
        self.update_user_profile(text, intent, entities)
        
        # Try to find answer in knowledge base first
        kb_match = self.find_best_answer(text)
        if kb_match and kb_match['confidence'] > 0.5:
            response = kb_match['answer']
        else:
            # Use template-based responses
            if intent in self.response_templates:
                response = random.choice(self.response_templates[intent])
            else:
                response = random.choice(self.response_templates['unknown'])
        
        # Personalize response if we know the user's name
        if self.user_profile['name']:
            response = f"{self.user_profile['name']}, {response.lower()}"
        
        # Adjust response based on sentiment
        if sentiment == 'negative' and sentiment_confidence > 0.7:
            response = "I understand you might be frustrated. " + response
        elif sentiment == 'positive' and sentiment_confidence > 0.7:
            response = "I'm glad to hear that! " + response
        
        # Store conversation
        self.conversation_history.append({
            'user_input': text,
            'intent': intent,
            'sentiment': sentiment,
            'entities': entities,
            'response': response,
            'confidence': intent_confidence
        })
        
        return response
    
    def get_conversation_stats(self) -> Dict:
        """Get conversation statistics"""
        if not self.conversation_history:
            return {}
        
        intents = [turn['intent'] for turn in self.conversation_history]
        sentiments = [turn['sentiment'] for turn in self.conversation_history]
        confidences = [turn['confidence'] for turn in self.conversation_history]
        
        return {
            'total_interactions': len(self.conversation_history),
            'most_common_intent': Counter(intents).most_common(1)[0] if intents else None,
            'sentiment_distribution': dict(Counter(sentiments)),
            'average_confidence': np.mean(confidences),
            'user_profile': self.user_profile
        }
    
    def save_model(self, filepath: str):
        """Save trained models and data"""
        model_data = {
            'intent_pipeline': self.intent_pipeline,
            'sentiment_classifier': self.sentiment_classifier,
            'sentiment_vectorizer': self.sentiment_vectorizer,
            'topic_classifier': self.topic_classifier,
            'topic_vectorizer': self.topic_vectorizer,
            'kb_vectorizer': self.kb_vectorizer,
            'knowledge_base': self.knowledge_base,
            'knowledge_vectors': self.knowledge_vectors,
            'user_profile': self.user_profile,
            'models_trained': self.models_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained models and data"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.intent_pipeline = model_data['intent_pipeline']
            self.sentiment_classifier = model_data['sentiment_classifier']
            self.sentiment_vectorizer = model_data['sentiment_vectorizer']
            self.topic_classifier = model_data['topic_classifier']
            self.topic_vectorizer = model_data['topic_vectorizer']
            self.kb_vectorizer = model_data['kb_vectorizer']
            self.knowledge_base = model_data['knowledge_base']
            self.knowledge_vectors = model_data['knowledge_vectors']
            self.user_profile = model_data['user_profile']
            self.models_trained = model_data['models_trained']
            
            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")

# Example usage and testing
def demo_chatbot():
    """Demonstrate the chatbot functionality"""
    print("=== Traditional ML Chatbot Demo ===\n")
    
    # Initialize chatbot
    chatbot = TraditionalMLChatbot()
    
    # Sample training data
    training_data = [
        {'text': 'hello how are you', 'intent': 'greeting', 'sentiment': 'neutral', 'topic': 'general'},
        {'text': 'hi there good morning', 'intent': 'greeting', 'sentiment': 'positive', 'topic': 'general'},
        {'text': 'goodbye see you later', 'intent': 'goodbye', 'sentiment': 'neutral', 'topic': 'general'},
        {'text': 'what is machine learning', 'intent': 'question', 'sentiment': 'neutral', 'topic': 'technology'},
        {'text': 'can you help me please', 'intent': 'request', 'sentiment': 'neutral', 'topic': 'general'},
        {'text': 'this is not working properly', 'intent': 'complaint', 'sentiment': 'negative', 'topic': 'support'},
        {'text': 'thank you very much', 'intent': 'goodbye', 'sentiment': 'positive', 'topic': 'general'},
        {'text': 'how does this work', 'intent': 'question', 'sentiment': 'neutral', 'topic': 'general'},
    ]
    
    # Sample knowledge base
    knowledge_base = [
        {
            'question': 'What is machine learning?',
            'answer': 'Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.'
        },
        {
            'question': 'How does natural language processing work?',
            'answer': 'Natural Language Processing (NLP) uses computational techniques to analyze, understand, and generate human language in a valuable way.'
        },
        {
            'question': 'What is the difference between AI and ML?',
            'answer': 'AI is the broader concept of machines being able to carry out tasks in a smart way, while ML is a subset of AI that focuses on learning from data.'
        }
    ]
    
    # Train the chatbot
    print("Training chatbot...")
    chatbot.train_models(training_data)
    chatbot.build_knowledge_base(knowledge_base)
    
    # Interactive demo
    print("\n=== Chat with the bot (type 'quit' to exit) ===")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            break
        
        response = chatbot.generate_response(user_input)
        print(f"Bot: {response}")
    
    # Show conversation statistics
    print("\n=== Conversation Statistics ===")
    stats = chatbot.get_conversation_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    demo_chatbot()