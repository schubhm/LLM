# Traditional ML Chatbot

A comprehensive chatbot implementation using classical machine learning and natural language processing techniques. This chatbot combines multiple ML algorithms with rule-based patterns to provide intelligent conversational AI without relying on external APIs or large language models.

## Features

- **Multi-Model Architecture**: Intent classification, sentiment analysis, and topic detection
- **Knowledge Base Integration**: FAQ-style responses using similarity matching
- **Advanced Text Preprocessing**: POS tagging, lemmatization, and entity extraction
- **User Profile Management**: Personalized responses and conversation tracking
- **Rule-Based Fallbacks**: Pattern matching for reliable basic interactions
- **Model Persistence**: Save and load trained models
- **Conversation Analytics**: Detailed statistics and user insights

## Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Required Dependencies

```bash
pip install numpy pandas scikit-learn nltk
```

### NLTK Data Setup

The chatbot will automatically download required NLTK data on first run:
- punkt (tokenizer)
- stopwords
- wordnet (lemmatizer)
- averaged_perceptron_tagger (POS tagger)

## Quick Start

### Basic Usage

```python
from chatbot import TraditionalMLChatbot

# Initialize chatbot
chatbot = TraditionalMLChatbot()

# Sample training data
training_data = [
    {'text': 'hello how are you', 'intent': 'greeting', 'sentiment': 'neutral', 'topic': 'general'},
    {'text': 'goodbye see you later', 'intent': 'goodbye', 'sentiment': 'neutral', 'topic': 'general'},
    {'text': 'what is machine learning', 'intent': 'question', 'sentiment': 'neutral', 'topic': 'technology'},
    # Add more training examples...
]

# Train the models
chatbot.train_models(training_data)

# Chat with the bot
response = chatbot.generate_response("Hello!")
print(response)  # "Hello! How can I help you today?"
```

### With Knowledge Base

```python
# Add FAQ knowledge base
knowledge_base = [
    {
        'question': 'What is machine learning?',
        'answer': 'Machine learning is a subset of AI that enables computers to learn from data.'
    },
    {
        'question': 'How does NLP work?',
        'answer': 'NLP uses computational techniques to analyze and understand human language.'
    }
]

chatbot.build_knowledge_base(knowledge_base)

# Ask questions
response = chatbot.generate_response("What is machine learning?")
print(response)  # Returns the knowledge base answer
```

## Architecture

### Core Components

1. **Intent Classification Pipeline**
   - TF-IDF Vectorization
   - Logistic Regression Classifier
   - Cross-validation for evaluation

2. **Sentiment Analysis**
   - Count Vectorization
   - Multinomial Naive Bayes
   - Positive/Negative/Neutral classification

3. **Topic Classification**
   - TF-IDF Features
   - Random Forest Classifier
   - Multi-class topic detection

4. **Knowledge Base System**
   - Cosine similarity matching
   - TF-IDF vector space
   - Configurable similarity threshold

5. **Rule-Based Patterns**
   - Regex-based intent detection
   - Entity extraction (names, emails, dates)
   - Fallback response system

### Text Preprocessing Pipeline

```
Raw Text → Lowercase → Contraction Expansion → 
Tokenization → POS Tagging → Stop Word Removal → 
Lemmatization → Feature Vector
```

## Training Data Format

### Intent Classification Data

```python
training_data = [
    {
        'text': 'User input text',
        'intent': 'greeting|goodbye|question|request|complaint',
        'sentiment': 'positive|negative|neutral',
        'topic': 'general|technology|support|...'
    }
]
```

### Knowledge Base Format

```python
knowledge_base = [
    {
        'question': 'Frequently asked question',
        'answer': 'Corresponding answer or response'
    }
]
```

## Advanced Usage

### Model Persistence

```python
# Save trained models
chatbot.save_model('chatbot_model.pkl')

# Load pre-trained models
new_chatbot = TraditionalMLChatbot()
new_chatbot.load_model('chatbot_model.pkl')
```

### Conversation Analytics

```python
# Get detailed conversation statistics
stats = chatbot.get_conversation_stats()
print(stats)

# Output includes:
# - Total interactions
# - Most common intents
# - Sentiment distribution
# - Average confidence scores
# - User profile information
```

### Entity Extraction

```python
# Extract entities from user input
entities = chatbot.extract_entities("My name is John and my email is john@example.com")
print(entities)
# {'name': ['John'], 'email': ['john@example.com']}
```

### Custom Response Templates

```python
# Modify response templates
chatbot.response_templates['greeting'] = [
    "Welcome! How can I assist you today?",
    "Hello! What brings you here?",
    # Add custom greetings...
]
```

## Configuration Options

### Vectorizer Parameters

```python
# Customize TF-IDF settings
chatbot.vectorizer = TfidfVectorizer(
    max_features=5000,      # Maximum number of features
    ngram_range=(1, 3),     # Unigrams to trigrams
    min_df=2,               # Minimum document frequency
    max_df=0.8,             # Maximum document frequency
    stop_words='english'    # Stop words language
)
```

### Similarity Threshold

```python
# Adjust knowledge base matching sensitivity
answer = chatbot.find_best_answer(user_input, threshold=0.5)
```

### Classifier Options

```python
# Use different classifiers
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

chatbot.intent_classifier = SVC(kernel='rbf')
chatbot.topic_classifier = GradientBoostingClassifier()
```

## API Reference

### Main Classes

#### `TraditionalMLChatbot`

Main chatbot class with all functionality.

**Methods:**

- `train_models(training_data)` - Train all ML models
- `build_knowledge_base(knowledge_data)` - Build FAQ system
- `generate_response(text)` - Generate bot response
- `predict_intent(text)` - Classify user intent
- `predict_sentiment(text)` - Analyze sentiment
- `extract_entities(text)` - Extract named entities
- `save_model(filepath)` - Save trained models
- `load_model(filepath)` - Load trained models
- `get_conversation_stats()` - Get analytics

### Supported Intents

- `greeting` - Hello, hi, good morning
- `goodbye` - Bye, farewell, see you later
- `question` - What, how, when, where, why
- `request` - Please help, can you, I need
- `affirmation` - Yes, sure, okay
- `negation` - No, not really, disagree
- `unknown` - Fallback for unrecognized input

### Supported Entities

- `name` - Person names
- `email` - Email addresses
- `phone` - Phone numbers
- `date` - Date formats
- `time` - Time formats

## Performance Metrics

The chatbot provides evaluation metrics for all models:

- **Intent Classification**: Cross-validated accuracy
- **Sentiment Analysis**: Precision, recall, F1-score
- **Topic Classification**: Multi-class accuracy
- **Knowledge Base**: Similarity matching effectiveness

## Examples

### Customer Support Bot

```python
# Training data for customer support
support_data = [
    {'text': 'I have a problem with my account', 'intent': 'complaint', 'sentiment': 'negative', 'topic': 'account'},
    {'text': 'How do I reset my password', 'intent': 'question', 'sentiment': 'neutral', 'topic': 'account'},
    {'text': 'Thank you for your help', 'intent': 'goodbye', 'sentiment': 'positive', 'topic': 'general'},
]

support_kb = [
    {'question': 'How to reset password?', 'answer': 'Click on "Forgot Password" on the login page.'},
    {'question': 'Account locked help', 'answer': 'Contact support at support@company.com to unlock your account.'},
]

chatbot.train_models(support_data)
chatbot.build_knowledge_base(support_kb)
```

### Business Intelligence Bot

```python
# BI-specific training for Audacy
bi_data = [
    {'text': 'show me Q3 revenue numbers', 'intent': 'request', 'sentiment': 'neutral', 'topic': 'revenue'},
    {'text': 'what were our top performing shows', 'intent': 'question', 'sentiment': 'neutral', 'topic': 'content'},
    {'text': 'audience metrics look concerning', 'intent': 'complaint', 'sentiment': 'negative', 'topic': 'metrics'},
]

bi_kb = [
    {'question': 'Q3 revenue performance', 'answer': 'Q3 revenue was $X million, up Y% from Q2.'},
    {'question': 'Top performing content', 'answer': 'Morning show ratings increased 15% this quarter.'},
]
```

## Troubleshooting

### Common Issues

1. **NLTK Data Missing**
   ```python
   import nltk
   nltk.download('all')  # Download all NLTK data
   ```

2. **Low Accuracy**
   - Increase training data size
   - Improve data quality and labeling
   - Adjust vectorizer parameters
   - Try different classifiers

3. **Memory Issues**
   - Reduce `max_features` in vectorizers
   - Use smaller training datasets
   - Implement batch processing

4. **Slow Response Times**
   - Optimize preprocessing pipeline
   - Reduce knowledge base size
   - Use simpler classifiers

### Performance Optimization

```python
# Optimize for speed
chatbot.vectorizer = TfidfVectorizer(
    max_features=1000,      # Reduce feature space
    ngram_range=(1, 2),     # Simpler n-grams
    binary=True             # Binary features
)

# Optimize for accuracy
chatbot.vectorizer = TfidfVectorizer(
    max_features=10000,     # Larger feature space
    ngram_range=(1, 4),     # More complex n-grams
    sublinear_tf=True       # Sublinear TF scaling
)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

### Development Setup

```bash
git clone <repository-url>
cd traditional-ml-chatbot
pip install -r requirements.txt
python -m pytest tests/
```

## License

MIT License - see LICENSE file for details.

## Changelog

### v1.0.0
- Initial release
- Multi-model architecture
- Knowledge base integration
- User profile management
- Model persistence

## Support

For questions or issues:
- Create an issue on GitHub
- Check the troubleshooting section
- Review example implementations

## Acknowledgments

- Built with scikit-learn and NLTK
- Inspired by traditional NLP and ML techniques
- Designed for production-ready deployment