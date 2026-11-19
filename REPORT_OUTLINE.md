# TopicLens AI — Project Report Outline
**BCA AI & Data Science Capstone — Project #9: Text Topic Modelling**

---

## Chapter 1: Introduction
1.1 Background and Motivation  
1.2 Problem Statement  
1.3 Project Objectives  
1.4 Scope and Limitations  
1.5 Report Organisation  

## Chapter 2: Literature Review
2.1 Text Mining and Document Analysis  
2.2 Topic Modelling Overview  
2.3 Latent Dirichlet Allocation (LDA) — Theory and Development  
2.4 Comparative Methods: NMF, pLSA, BERTopic  
2.5 Evaluation Metrics for Topic Models  

## Chapter 3: System Design and Architecture
3.1 System Overview  
3.2 Technology Stack and Justification  
3.3 Application Architecture (Modular Design)  
3.4 Data Flow Diagram  
3.5 Key Design Decisions (CountVectorizer vs TF-IDF for LDA)  

## Chapter 4: Implementation
4.1 Development Environment Setup  
4.2 Dataset Ingestion and Management (Module: data_loader.py)  
4.3 Text Preprocessing Pipeline (Module: preprocessing.py)  
4.4 Feature Engineering — Document-Term Matrix (Module: feature_engineering.py)  
4.5 LDA Topic Modelling Engine (Module: topic_model.py)  
4.6 Model Evaluation (Module: evaluation.py)  
4.7 Visualisation Components (Module: visualization.py)  
4.8 Export and Reporting (Modules: export_manager.py, utils.py)  
4.9 Streamlit User Interface (app.py + views/)  

## Chapter 5: Experimental Results and Analysis
5.1 Datasets Used  
5.2 Preprocessing Results  
5.3 Topic Discovery Results — 20 Newsgroups Corpus  
5.4 Topic Discovery Results — Tech News Dataset  
5.5 Quantitative Evaluation (Perplexity, Coherence, Diversity)  
5.6 Qualitative Evaluation and Human Interpretability  
5.7 Comparison of Topic Counts (K=3, 4, 5, 6)  

## Chapter 6: User Interface Demonstration
6.1 Home Dashboard  
6.2 Dataset Upload Page  
6.3 Preprocessing Page  
6.4 Topic Modelling Configuration  
6.5 Topic Exploration  
6.6 Document Analysis  
6.7 Visual Analytics  
6.8 Model Evaluation  
6.9 Export Results  

## Chapter 7: Testing and Validation
7.1 Unit Test Strategy  
7.2 Test Coverage Report  
7.3 End-to-End Pipeline Test  
7.4 Manual Acceptance Testing Checklist  

## Chapter 8: Limitations and Future Work
8.1 Known Limitations  
8.2 Proposed Enhancements  
8.3 Scalability Considerations  

## Chapter 9: Conclusion
9.1 Summary of Achievements  
9.2 Learning Outcomes  
9.3 Contribution  

## References
(IEEE or APA citation format — include Blei et al. 2003 LDA paper, Mimno et al. 2011 Coherence paper, Scikit-learn documentation)

## Appendices
Appendix A: Installation Guide  
Appendix B: Source Code Listing (key modules)  
Appendix C: Sample Output Screenshots  
Appendix D: GitHub Repository Link  
