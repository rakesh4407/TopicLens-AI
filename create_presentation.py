"""
create_presentation.py
Generates the complete 20-slide TopicLens AI PowerPoint presentation (.pptx)
with widescreen 16:9 layout, modern styling, and embedded speaker notes.
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

def build_presentation(output_path="TopicLens_AI_Presentation.pptx"):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Color Palette
    C_DARK_BG   = RGBColor(13, 27, 42)     # #0D1B2A
    C_CARD_BG   = RGBColor(255, 255, 255)  # White
    C_LIGHT_BG  = RGBColor(245, 247, 250)  # #F5F7FA
    C_NAVY_TEXT = RGBColor(15, 23, 42)     # #0F172A
    C_BLUE      = RGBColor(37, 99, 235)    # #2563EB
    C_PURPLE    = RGBColor(124, 58, 237)   # #7C3AED
    C_MUTED     = RGBColor(100, 116, 139)  # #64748B
    C_WHITE     = RGBColor(255, 255, 255)
    C_EMERALD   = RGBColor(16, 185, 129)   # #10B981
    C_BORDER    = RGBColor(226, 232, 240)  # #E2E8F0

    # Slides Data
    slides_data = [
        # Slide 1
        {
            "is_title": True,
            "title": "TopicLens AI",
            "subtitle": "Intelligent Text Topic Modelling & Unsupervised Theme Discovery Platform",
            "bullets": [
                "BCA AI & Data Science Capstone Project · Project #9",
                "Individual Final Semester Implementation",
                "Stack: Python 3.13 | Streamlit | Scikit-Learn | NLTK | Plotly | Joblib",
                "Repository: github.com/rakesh4407/TopicLens-AI"
            ],
            "notes": "Good morning respected examiners and faculty. Today I present my final BCA Capstone Project: TopicLens AI, an unsupervised NLP system for automatic topic discovery from unstructured text using Latent Dirichlet Allocation."
        },
        # Slide 2
        {
            "title": "Problem Statement & Industry Motivation",
            "category": "PROJECT BACKGROUND",
            "bullets": [
                "The Unstructured Data Explosion: Over 80% of enterprise information is raw text (reviews, emails, research papers, news).",
                "The Manual Triage Bottleneck: Reading and categorizing large document archives manually is cost-prohibitive and unscalable.",
                "Lack of Annotations: Absence of ground-truth labels prevents the use of supervised classification models.",
                "The Solution Need: An unsupervised probabilistic system that groups text by latent semantic themes without human supervision."
            ],
            "notes": "Organizations handle massive unstructured textual data daily without labels. Supervised learning fails here due to lack of annotations. Unsupervised topic modeling solves this by extracting natural clusters directly from word co-occurrences."
        },
        # Slide 3
        {
            "title": "Project Aim & Core Objectives",
            "category": "PROJECT SCOPE",
            "bullets": [
                "Complete 7-Stage Pipeline: Ingestion → Preprocessing → Feature Engineering → LDA Modeling → Evaluation → Visual Analytics → Export.",
                "Standard-Compliant LDA Architecture: Strict usage of CountVectorizer integer counts to preserve multinomial likelihood assumptions.",
                "10-Page Interactive Streamlit Application: Responsive, modern UI with real-time pipeline status tracking and state persistence.",
                "Human-in-the-Loop Interpretation: Separation of automatic statistical labels from human-curated domain concepts.",
                "Live Out-of-Vocabulary (OOV) Inference: Real-time classification of unseen documents with vocabulary coverage diagnostics.",
                "Continuous Quality Assurance: 43 automated unit and integration tests verified via GitHub Actions CI."
            ],
            "notes": "The aim was to engineer a production-ready application rather than a trivial notebook. All modules are decoupled, tested, and wrapped in a 10-page Streamlit analytics dashboard."
        },
        # Slide 4
        {
            "title": "Theoretical Foundation: Latent Dirichlet Allocation (LDA)",
            "category": "MACHINE LEARNING THEORY",
            "bullets": [
                "Proposed by Blei, Ng, and Jordan (2003) as a hierarchical Bayesian generative probabilistic model.",
                "Document-Topic Mixture Assumption: Every document d is represented as a probability distribution θ_d over K topics.",
                "Topic-Word Distribution Assumption: Every topic k is represented as a probability distribution φ_k over vocabulary words.",
                "Observable vs. Latent Variables: Observable = document words w; Latent = topic distributions θ, word assignments z, and vocabulary weights φ.",
                "The Inversion Goal: Use statistical inference (Variational Bayes) to deduce the hidden topics from observed word co-occurrences."
            ],
            "notes": "LDA posits that documents are mixtures of topics, and topics are mixtures of words. Our algorithm reverses this generative process: given only the raw words, it deduces the underlying topic mixtures."
        },
        # Slide 5
        {
            "title": "Mathematical Formulation & Hyperparameters",
            "category": "MATHEMATICAL CORE",
            "bullets": [
                "Dirichlet Prior Alpha (α): Governs document-topic sparsity θ ~ Dir(α). Low α = documents focus on 1-2 topics; High α = documents blend many topics.",
                "Dirichlet Prior Beta (β / η): Governs topic-word sparsity φ ~ Dir(β). Low β = topics defined by specific keywords; High β = topics use broad vocabulary.",
                "Joint Probability Distribution: P(W, Z, θ, φ | α, β) = ∏ P(φ_k|β) ∏ P(θ_d|α) [ ∏ P(z_dn|θ_d) P(w_dn|φ_z_dn) ].",
                "Inference Engine: Online Variational Bayes (Hoffman et al., 2010) via scikit-learn for memory-efficient and scalable convergence."
            ],
            "notes": "The Dirichlet priors alpha and beta control distribution sparsity. In TopicLens AI, users can tune these priors or use empirical Bayes estimation to adapt to short reviews or lengthy research abstracts."
        },
        # Slide 6
        {
            "title": "Architectural Decision: CountVectorizer vs TF-IDF",
            "category": "CRITICAL ML DECISION",
            "bullets": [
                "CountVectorizer (Primary Input): Produces integer word counts required by LDA's Multinomial likelihood function.",
                "TF-IDF (Exploratory Only): Generates weighted floating-point scores that violate the discrete multinomial generative assumption.",
                "Theoretical Incompatibility: Feeding TF-IDF into standard LDA breaks Dirichlet-Multinomial conjugacy and distorts topic coherence.",
                "TopicLens AI Design: Strict standard compliance—CountVectorizer powers the LDA estimator, while TF-IDF is kept as an independent exploratory feature."
            ],
            "notes": "A widespread error in NLP capstone projects is feeding TF-IDF directly into standard LDA. We strictly follow statistical theory: CountVectorizer integer counts are used for LDA, preserving proper multinomial likelihood."
        },
        # Slide 7
        {
            "title": "End-to-End System Architecture",
            "category": "SOFTWARE ENGINEERING",
            "bullets": [
                "Layer 1: Streamlit Frontend (app.py) — 10 dynamic views, dark glass sidebar, custom responsive CSS.",
                "Layer 2: Session Manager (session_manager.py) — Cross-page state persistence and atomic pipeline reset mechanics.",
                "Layer 3: Core NLP Engine (src/) — data_loader, preprocessing, feature_engineering, topic_model.",
                "Layer 4: Evaluation & Inference (src/) — evaluation (UMass, Perplexity), inference (OOV-aware real-time predictor).",
                "Layer 5: Visualization & Export (src/) — visualization (Plotly, WordCloud), export_manager (Joblib, CSV, HTML reports)."
            ],
            "notes": "The architecture separates presentation from computation. No ML logic is hardcoded inside UI scripts; all business logic lives in modular, unit-tested Python classes."
        },
        # Slide 8
        {
            "title": "Multi-Format Data Ingestion & Benchmarks",
            "category": "DATA ENGINEERING",
            "bullets": [
                "File Format Flexibility: Delimited CSV files (with automatic text-column detection) and multi-file TXT folder archives.",
                "Four Built-in Academic Benchmark Datasets:",
                "  1. 20 Newsgroups Subset: Canonical multi-topic forum discussions (Tech, Science, Politics, Religion).",
                "  2. Biomedical Research Corpus: 25 scientific abstracts across Cardiology, Oncology, Neuroscience, Pharmacology.",
                "  3. Customer Feedback Corpus: 25 real-world e-commerce consumer product reviews.",
                "  4. Technology News Corpus: 20 tech journalism articles covering Artificial Intelligence, Cloud, and Security.",
                "Data Integrity: Empty file detection, missing row imputation, and Latin-1/UTF-8 character encoding handling."
            ],
            "notes": "TopicLens AI supports both custom CSV/TXT uploads and four diverse pre-packaged corpora across medicine, commerce, tech, and forums to facilitate instant demonstrations."
        },
        # Slide 9
        {
            "title": "10-Stage Text Preprocessing Pipeline",
            "category": "NATURAL LANGUAGE PROCESSING",
            "bullets": [
                "1. Lowercase Normalization | 2. HTML Tag Stripping (Regex) | 3. URL & Web Link Removal.",
                "4. Punctuation Filtering | 5. Standalone Number Filtering | 6. Alphanumeric Tokenization.",
                "7. Stopword Elimination (NLTK) | 8. WordNet Lemmatization (Grammatical base form reduction).",
                "9. Empty Document Pruning | 10. Deduplication with Synchronized Original Document IDs.",
                "Proof of Cleaning: 'Check out https://med.org! Patients received 100mg dosages, showing improvements.' → ['patient', 'receive', 'dosage', 'show', 'improvement']."
            ],
            "notes": "Our 10-step pipeline removes noise while retaining semantic roots through WordNet lemmatization. It also preserves original document IDs so post-cleaning distributions stay aligned."
        },
        # Slide 10
        {
            "title": "Feature Engineering & Sparsity Analysis",
            "category": "FEATURE ENGINEERING",
            "bullets": [
                "CountVectorizer Hyperparameter Controls:",
                "  - min_df: Minimum document frequency threshold to filter idiosyncratic noise terms.",
                "  - max_df: Maximum document frequency threshold to filter corpus-specific stopwords.",
                "  - max_features: Vocabulary capacity constraint (100 to 5,000 top terms).",
                "  - n-gram range: Flexible unigram (1,1) or unigram + bigram (1,2) support (e.g., 'machine learning').",
                "Real-Time Sparsity Computation: Sparsity = [1 - (Non-Zero Cells / (D × V))] × 100%. Displays exact matrix density to validate feature tuning."
            ],
            "notes": "Text matrices are notoriously sparse. Our Feature Engineer computes matrix sparsity in real-time, helping users calibrate min_df and max_df appropriately."
        },
        # Slide 11
        {
            "title": "Application Design: 10-Page Streamlit GUI",
            "category": "UI/UX DESIGN",
            "bullets": [
                "Structured 10-View Workflow: Home, Upload, Preprocessing, Modeling, Exploration, Analysis, Analytics, Evaluation, Export, About.",
                "Dark Glass Modern Sidebar: Multi-stop gradient (#0d1b2a to #1a2744), custom brand header, and BCA Capstone badge.",
                "Live Pipeline Status Monitor: Dynamic status indicators (green completed, blue pulsing active, grey pending).",
                "Custom CSS Design System: Blue gradient KPI cards, hover-lift transitions, and interactive data tables."
            ],
            "notes": "The UI provides a SaaS-like experience with 10 dedicated views, eliminating cluttered single-page scroll layouts and providing immediate visual feedback on pipeline progress."
        },
        # Slide 12
        {
            "title": "Topic Discovery & Human-in-the-Loop Labeling",
            "category": "THEMATIC INTERPRETATION",
            "bullets": [
                "Keyword Weight Extraction: Top-N terms per topic ranked by empirical word-probability weight φ_kw.",
                "The Unsupervised Semantic Gap: Algorithms produce mathematical word clusters, not conceptual human meanings.",
                "Dual Labeling Architecture:",
                "  - Automated Machine Label: Top 3 terms synthesized into a descriptive string (e.g., 'cardiac_patient_heart').",
                "  - Human-in-the-Loop Override: Allows domain analysts to input validated titles (e.g., 'Cardiology & Cardiovascular Care').",
                "Persistent Synchronization: Custom human labels propagate across all downstream heatmaps, charts, and downloadable reports."
            ],
            "notes": "Unsupervised models output word clusters, not semantic concepts. TopicLens AI bridges this semantic gap by allowing human domain experts to validate and assign real-world topic titles."
        },
        # Slide 13
        {
            "title": "Document Inference & Out-of-Vocabulary Engine",
            "category": "LIVE PREDICTION",
            "bullets": [
                "Real-Time Text Classifier: Interactive testing panel to input unseen sentences and compute instant topic probabilities.",
                "Zero Train-Serve Skew: The unseen text is automatically cleaned with the identical preprocessor instance fitted during training.",
                "Out-of-Vocabulary (OOV) Diagnostics: Detects words missing from the training dictionary and displays vocabulary coverage percentage.",
                "Confidence Scoring: Reports the dominant topic, posterior probability score, and relative margin of confidence."
            ],
            "notes": "To prevent train-serve skew, our inference engine applies the exact preprocessor fitted during training to incoming unseen text, and provides alerts if input tokens are out-of-vocabulary."
        },
        # Slide 14
        {
            "title": "Interactive Visual Analytics Suite",
            "category": "DATA VISUALIZATION",
            "bullets": [
                "Topic-Keyword Probability Bars: Interactive Plotly horizontal bar charts displaying relative word weights per topic.",
                "High-Resolution WordClouds: Matplotlib-rendered frequency clouds with dynamic color palettes per topic.",
                "Document-Topic Probability Heatmap: Two-dimensional matrix showing probabilistic distribution of all documents across topics.",
                "Corpus Topic Distribution: Interactive pie/donut charts showing dominant topic market share across the dataset.",
                "2D PCA Projection: Dimensionality reduction projecting high-dimensional topic vectors into 2D space to verify topic separation."
            ],
            "notes": "We provide five rich visualization views using Plotly and Matplotlib, including 2D PCA projections and heatmaps, enabling immediate visual confirmation of topic quality."
        },
        # Slide 15
        {
            "title": "Quantitative Evaluation & Validation Metrics",
            "category": "MODEL EVALUATION",
            "bullets": [
                "Perplexity: Exponential of negative normalized log-likelihood; measures how surprised the model is by the text (lower is better).",
                "Log-Likelihood: Total corpus log-likelihood under the fitted LDA model (higher is better).",
                "UMass Topic Coherence (Mimno et al., 2011): Computes logarithmic co-document frequencies of top keywords with canonical ε = 1.0 (closer to 0 is better).",
                "Topic Diversity: Ratio of unique words across all top topic keywords (range 0.0 to 1.0).",
                "The Academic Caveat: Chang et al. (2009 - 'Reading Tea Leaves') proved statistical metrics alone do not guarantee human interpretability."
            ],
            "notes": "We evaluate models using Perplexity, Log-Likelihood, Topic Diversity, and canonical UMass Coherence. We also emphasize that metrics must always be complemented by human qualitative validation."
        },
        # Slide 16
        {
            "title": "Model Persistence & Multi-Format Exports",
            "category": "DEPLOYMENT & SERIALIZATION",
            "bullets": [
                "Atomic Joblib Model Bundle: Serializes the fitted LDA estimator, CountVectorizer vocabulary, TextPreprocessor configuration, custom labels, and metadata.",
                "Multi-Format Analytical Downloads:",
                "  - CSV: Document-topic assignments with dominant topic and probability vectors.",
                "  - CSV: Topic keyword weight matrix across all vocabulary terms.",
                "  - CSV: Fully cleaned, preprocessed dataset.",
                "  - Joblib: Serialized model bundle ready for production Python deployment.",
                "  - Markdown & HTML: Executive analytical report summarizing corpus metrics, hyperparameters, topics, and evaluation scores."
            ],
            "notes": "TopicLens AI serializes an atomic Joblib bundle containing the preprocessor, vectorizer, and custom labels. Users can also export CSVs and comprehensive HTML/Markdown summary reports."
        },
        # Slide 17
        {
            "title": "Software Quality Assurance & Testing Suite",
            "category": "VERIFICATION & TESTING",
            "bullets": [
                "Automated Testing Suite: 8 dedicated test modules with 43 automated test methods built with Python's unittest.",
                "100% Passing Status: All 43 tests pass consistently in ~22 seconds.",
                "Test Coverage Matrix:",
                "  - Data Loading & Encoding: 6 tests | Preprocessing & ID Mapping: 5 tests.",
                "  - Feature Engineering & DTM Sparsity: 11 tests | Topic Model Fitting & Priors: 7 tests.",
                "  - Live Inference & OOV Diagnostics: 8 tests | Evaluation & Coherence: 5 tests.",
                "  - Multi-Format Exports: 2 tests | End-to-End Pipeline Integration: 1 complete test.",
                "Continuous Integration: GitHub Actions CI workflow running across Ubuntu and Windows on Python 3.12 and 3.13."
            ],
            "notes": "Every single mathematical calculation, preprocessing step, and inference call is verified by 43 automated unit tests, and continuously checked using GitHub Actions CI."
        },
        # Slide 18
        {
            "title": "Technical Challenges & Engineering Fixes",
            "category": "CHALLENGES & RESOLUTIONS",
            "bullets": [
                "Challenge 1: Train-Serve Preprocessing Divergence → Solved by binding preprocessor configuration directly into the serialized model bundle.",
                "Challenge 2: Document ID Desynchronization After Cleaning → Solved by implementing dual-key (original_doc_id & processed_doc_id) tracking.",
                "Challenge 3: Coherence Score Distortion from Infinitesimal Epsilon → Solved by enforcing canonical ε = 1.0 per Mimno (2011).",
                "Model Limitation: Inherent Bag-of-Words assumption ignores syntactic word ordering and semantic context.",
                "Hyperparameter Sensitivity: Specifying K a priori requires iterative tuning or domain intuition."
            ],
            "notes": "We addressed several real-world engineering challenges, including train-serve skew and document ID desynchronization, while acknowledging standard LDA's bag-of-words limitations."
        },
        # Slide 19
        {
            "title": "Future Enhancements & Technical Roadmap",
            "category": "FUTURE SCOPE",
            "bullets": [
                "Transformer-Based Embeddings (BERTopic): Integrating Sentence-BERT with UMAP dimensionality reduction and HDBSCAN clustering for contextual modeling.",
                "Dynamic & Temporal Topic Modelling: Tracking topic trajectory and theme drift across timestamped historical corpora.",
                "Automated Hyperparameter Optimization: Automated grid-search over K values plotting coherence curves to recommend optimal topic counts.",
                "Containerization & Microservices: Docker packaging and FastAPI REST service endpoints for enterprise enterprise integration."
            ],
            "notes": "Our roadmap includes contextual transformer models (BERTopic), temporal topic tracking across chronological corpora, and containerized deployment with FastAPI."
        },
        # Slide 20
        {
            "title": "Conclusion, Capstone Learnings & Q&A",
            "category": "SUMMARY & VIVA",
            "bullets": [
                "Delivered a complete, mathematically sound, production-quality NLP platform from scratch.",
                "Rigorous statistical compliance: CountVectorizer + Latent Dirichlet Allocation.",
                "Professional 10-page Streamlit application with modern UI and state persistence.",
                "Backed by 43 automated unit tests, CI/CD pipeline, and comprehensive documentation.",
                "Project Repository: https://github.com/rakesh4407/TopicLens-AI",
                "Thank you for your time and consideration! Questions & discussion welcomed."
            ],
            "notes": "TopicLens AI demonstrates the full end-to-end lifecycle of an AI and Data Science project. Thank you respected examiners, I am now ready for your questions."
        }
    ]

    for slide_data in slides_data:
        slide = prs.slides.add_slide(blank_layout)

        # Slide Notes
        notes_slide = slide.notes_slide
        notes_text_frame = notes_slide.notes_text_frame
        notes_text_frame.text = slide_data.get("notes", "")

        if slide_data.get("is_title", False):
            # Title Slide Layout (Dark Theme)
            bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
            bg.fill.solid()
            bg.fill.fore_color.rgb = C_DARK_BG
            bg.line.fill.background()

            # Accent banner bar
            bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.2), Inches(0.15), Inches(4.5))
            bar.fill.solid()
            bar.fill.fore_color.rgb = C_BLUE
            bar.line.fill.background()

            # Title Box
            tx_box = slide.shapes.add_textbox(Inches(1.2), Inches(1.2), Inches(11.0), Inches(5.0))
            tf = tx_box.text_frame
            tf.word_wrap = True

            p1 = tf.paragraphs[0]
            p1.text = slide_data["title"]
            p1.font.size = Pt(44)
            p1.font.bold = True
            p1.font.color.rgb = RGBColor(96, 165, 250)  # Light Blue
            p1.font.name = "Segoe UI"

            p2 = tf.add_paragraph()
            p2.text = slide_data["subtitle"]
            p2.font.size = Pt(20)
            p2.font.color.rgb = RGBColor(226, 232, 240)
            p2.font.name = "Segoe UI"
            p2.space_before = Pt(12)
            p2.space_after = Pt(24)

            for b in slide_data["bullets"]:
                p = tf.add_paragraph()
                p.text = "•  " + b
                p.font.size = Pt(15)
                p.font.color.rgb = RGBColor(148, 163, 184)
                p.font.name = "Segoe UI"
                p.space_after = Pt(6)

        else:
            # Content Slide Layout
            # Light Background
            bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
            bg.fill.solid()
            bg.fill.fore_color.rgb = C_LIGHT_BG
            bg.line.fill.background()

            # Top Header Bar
            header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(1.25))
            header.fill.solid()
            header.fill.fore_color.rgb = C_DARK_BG
            header.line.fill.background()

            # Top Accent Line
            line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(1.22), Inches(13.333), Inches(0.04))
            line.fill.solid()
            line.fill.fore_color.rgb = C_BLUE
            line.line.fill.background()

            # Header Category & Title Text
            h_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.12), Inches(11.5), Inches(1.0))
            h_tf = h_box.text_frame
            h_tf.word_wrap = True

            p_cat = h_tf.paragraphs[0]
            p_cat.text = slide_data.get("category", "TOPICLENS AI · BCA CAPSTONE").upper()
            p_cat.font.size = Pt(11)
            p_cat.font.bold = True
            p_cat.font.color.rgb = RGBColor(96, 165, 250)
            p_cat.font.name = "Segoe UI"

            p_title = h_tf.add_paragraph()
            p_title.text = slide_data["title"]
            p_title.font.size = Pt(24)
            p_title.font.bold = True
            p_title.font.color.rgb = C_WHITE
            p_title.font.name = "Segoe UI"
            p_title.space_before = Pt(2)

            # Main Content Card
            card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(11.733), Inches(5.4))
            card.fill.solid()
            card.fill.fore_color.rgb = C_CARD_BG
            card.line.color.rgb = C_BORDER
            card.line.width = Pt(1)

            # Bullet Points Inside Card
            b_box = slide.shapes.add_textbox(Inches(1.2), Inches(1.75), Inches(10.9), Inches(4.9))
            b_tf = b_box.text_frame
            b_tf.word_wrap = True

            bullets = slide_data["bullets"]
            for i, b_text in enumerate(bullets):
                p = b_tf.paragraphs[0] if i == 0 else b_tf.add_paragraph()
                p.text = "•  " + b_text
                p.font.size = Pt(15 if len(bullets) <= 5 else 14)
                p.font.color.rgb = C_NAVY_TEXT
                p.font.name = "Segoe UI"
                p.space_before = Pt(8 if len(bullets) <= 5 else 6)
                p.space_after = Pt(8 if len(bullets) <= 5 else 6)

            # Footer
            f_box = slide.shapes.add_textbox(Inches(0.8), Inches(7.0), Inches(11.733), Inches(0.35))
            f_tf = f_box.text_frame
            p_f = f_tf.paragraphs[0]
            p_f.text = "TopicLens AI · Project #9: Text Topic Modelling · BCA AI & Data Science"
            p_f.font.size = Pt(9)
            p_f.font.color.rgb = C_MUTED
            p_f.font.name = "Segoe UI"

    prs.save(output_path)
    print(f"Presentation saved successfully to: {output_path}")

if __name__ == "__main__":
    build_presentation()
