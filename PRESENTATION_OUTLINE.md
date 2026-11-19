# TopicLens AI — Presentation Outline
**BCA AI & Data Science Capstone — Project #9: Text Topic Modelling**  
Suggested Duration: 10–15 minutes

---

## Slide 1: Title Slide
- Project Name: TopicLens AI
- Tagline: Discover Hidden Meaning. Explore Every Topic.
- Student Name | Roll Number | Academic Year
- Project #9 — Text Topic Modelling
- BCA AI & Data Science Capstone

## Slide 2: Problem Statement
- Challenge: Large collections of unstructured text are difficult to organise manually
- Scale: Thousands of documents, emails, reviews, articles per day
- Solution need: Automated, unsupervised theme discovery

## Slide 3: Objectives
- Build an interactive NLP application using Python and Streamlit
- Implement LDA topic modelling with CountVectorizer
- Visualise discovered topics with Plotly, WordCloud, and Matplotlib
- Allow full pipeline: ingest → preprocess → model → evaluate → export

## Slide 4: Technology Stack
- Python 3.13 — Core programming language
- Streamlit — Interactive analytics dashboard
- Scikit-Learn — LDA + CountVectorizer
- NLTK — Text preprocessing (tokenisation, lemmatisation, stopword removal)
- Plotly + Matplotlib — Interactive and static visualisations
- Joblib — Model persistence

## Slide 5: System Architecture
- Architecture diagram: User → Streamlit UI → Session State → Pipeline Modules
- Modules: data_loader → preprocessing → feature_engineering → topic_model → evaluation → visualization → export_manager

## Slide 6: LDA Algorithm Explanation
- Brief: Documents are mixtures of topics; topics are distributions over words
- Input: CountVectorizer Document-Term Matrix (integer counts)
- Why NOT TF-IDF: LDA requires multinomial likelihood (discrete counts)
- Hyperparameters: K (topics), alpha (doc-topic prior), beta (topic-word prior)

## Slide 7: Text Preprocessing Pipeline
- Steps: Lowercase → HTML removal → URL removal → Punctuation → Numbers → Tokenise → Stopwords → Lemmatise → Drop empty → Dedup
- Before/After example side by side

## Slide 8: Live Application Demonstration
- Load 20 Newsgroups sample dataset
- Run preprocessing
- Train LDA (K=4)
- Show topic keywords
- Show document assignments

## Slide 9: Visualisations
- Topic keyword bar charts
- Word clouds per topic
- Document-topic probability heatmap
- Topic distribution pie chart

## Slide 10: Evaluation Results
- Perplexity, Log-Likelihood
- UMass Coherence
- Topic Diversity score
- Key limitation: metrics ≠ guaranteed human interpretability

## Slide 11: Export Features
- Document-topic CSV
- Topic keywords CSV
- Analysis report (Markdown + HTML)
- Trained model (Joblib)

## Slide 12: Testing
- 28 unit tests (17 core + 11 feature engineering)
- End-to-end pipeline test
- All tests pass

## Slide 13: Limitations & Future Work
- LDA assumes bag-of-words (no word order)
- Optimal K requires human judgment
- Future: BERTopic, dynamic topic modelling, real-time streaming

## Slide 14: Conclusion
- Successfully built a complete, functional NLP analytics platform
- Real machine learning pipeline — no hardcoded/fake results
- Professional Streamlit interface with 10 pages
- Modular, tested, documented codebase

## Slide 15: Q&A
- Thank you
- Questions?
