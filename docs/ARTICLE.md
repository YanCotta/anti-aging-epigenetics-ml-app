Here’s your text translated into English in **Markdown format**:

---

# POSSIBLE ABSTRACT

Aging is a multifactorial process influenced by the complex interaction between genetic predisposition and lifestyle factors. Personalized medicine seeks to decipher these interactions to promote healthy longevity but faces challenges such as the interpretability of predictive models, genomic data privacy, and accessibility for the general public. This work presents the development of a personalized anti-aging recommendation computational system, conceived as a minimum viable product (MVP) to translate complex data into actionable health insights.

The methodology included the generation of synthetic genomic data to ensure privacy, the integration of single nucleotide polymorphisms (SNPs) and behavioral factors, and the training of machine learning models (Random Forest and Multi-Layer Perceptron) for risk classification. Result interpretability was ensured through the implementation of Explainable Artificial Intelligence (XAI) techniques, specifically SHAP values (Shapley Additive Explanations), to provide transparent explanations of predictions.

The system’s architecture was developed with a FastAPI backend and React frontend, ensuring scalability, security, and an intuitive user interface. The result is a functional web platform that demonstrates the feasibility of integrating explainable AI and privacy-by-design principles to create personalized health tools, contributing to the fields of gerontechnology, bioinformatics, and health informatics by offering a robust and accessible framework for proactive aging management.

**Keywords:** Personalized Medicine. Machine Learning. Explainable Artificial Intelligence. Aging. Genomics.

---

# 1 INTRODUCTION

The intersection of genetics, lifestyle factors, and aging has emerged as one of the most promising frontiers in personalized medicine and public health. As global populations continue to age, there is an urgent need to understand how individual genetic variations interact with environmental and lifestyle factors to influence the aging process and susceptibility to age-related diseases.

The advent of machine learning and explainable artificial intelligence technologies has opened new paths for the development of personalized anti-aging intervention systems that can provide individuals with actionable insights based on their unique genetic and lifestyle profiles (HUANG et al., 2024).

---

## 1.1 Background and Rationale

Aging is a complex biological process characterized by the progressive decline of physiological functions, increased susceptibility to age-related diseases, and ultimately higher mortality risk (LÓPEZ-OTÍN et al., 2013). While chronological age provides a basic framework for understanding aging, biological age—determined by the interaction of genetic predisposition, epigenetic modifications, and lifestyle factors—offers a more accurate representation of an individual’s health status and disease risk (JYLHÄVÄ; PEDERSEN; HÄGG, 2017).

Research has shown that part of human longevity variation is determined by genetics, with genes such as SIRT1, APOE, FOXO3, and CETP showing associations with longevity, although these variants are not found in all individuals with exceptional longevity (SEBASTIANI; PERLS, 2012). Most of the variation, however, is influenced by environmental and lifestyle factors, making personalized interventions based on genetic and lifestyle data particularly promising (VAISERMAN; KOLIADA; LUSHCHAK, 2021).

Epigenetic mechanisms, including DNA methylation, histone modifications, and non-coding RNA regulation, serve as crucial mediators between genetic predisposition and environmental influences in the aging process (SEN et al., 2016). These reversible modifications can be influenced by lifestyle factors such as diet, physical activity, stress management, and sleep patterns, providing targets for personalized interventions (ALEGRÍA-TORRES; BACCARELLI; BOLLATI, 2011).

The development of epigenetic clocks, which use DNA methylation patterns to predict biological age with increasing accuracy, has demonstrated the feasibility of using molecular biomarkers for aging assessment and intervention monitoring (HORVATH, 2013; HANNUM et al., 2013; FIELD et al., 2018).

---

## 1.2 Current Challenges and Limitations

Despite significant advances in understanding the genetic and epigenetic basis of aging, several challenges limit the translation of this knowledge into practical applications for personalized anti-aging interventions:

* **Data Privacy and Ethical Issues**: The use of genetic data for personalized recommendations raises major privacy and ethical concerns, particularly regarding storage, sharing, and potential discrimination. Regulations such as the General Data Protection Regulation (GDPR) and Brazil’s General Data Protection Law (LGPD) have created a complex landscape for managing genetic data (WORLD HEALTH ORGANIZATION, 2024).
* **Limited Accessibility and Interpretability**: Current genetic testing and analysis platforms often provide complex results that are difficult for non-specialists to understand and apply. The “black box” nature of many machine learning models used in genetic analysis further exacerbates this issue, limiting user trust and adoption (NATURE, 2022).
* **Lack of Comprehensive Integration**: Most existing systems focus either on genetic or lifestyle factors in isolation, failing to capture the complex interactions between genetic predisposition and environmental influences that determine aging outcomes (PETERS et al., 2015).
* **Scalability and Cost**: Traditional genetic analysis approaches and personalized medicine recommendations often require expensive laboratory analyses and expert interpretation, limiting accessibility to wider populations.

---

## 1.3 Technological Solutions and Innovation

Recent advances in machine learning & deep learning, particularly in explainable artificial intelligence (XAI), offer promising solutions to these challenges. SHAP (Shapley Additive Explanations) and similar XAI techniques allow the creation of interpretable models that can provide transparent explanations for genetic risk assessments, enhancing user trust and supporting informed decision-making (ELSEVIER, 2024; MDPI, 2024; NATURE, 2025).

The development of synthetic data generation techniques addresses privacy concerns, enabling model training and validation without exposing sensitive personal genetic information (CHEN et al., 2025).

Modern web technologies, especially frameworks like FastAPI, have shown outstanding performance in healthcare applications, enabling real-time processing of complex genetic and lifestyle data while maintaining security and scalability. Containerization technologies such as Docker ensure consistent deployment and reduce technical barriers for implementing personalized medicine systems. Additionally, MLflow was adopted as an experiment tracking platform, enabling model versioning, reproducibility (parameters, metrics, and artifacts), and comparisons between Random Forest and MLP.

---

## 1.4 Research Objectives and Contributions

This thesis presents the development and implementation of a comprehensive personalized anti-aging recommendation system that addresses current limitations in the field through several key innovations:

* **Data Integration**: Integration of SNPs, lifestyle factors, and demographics to estimate biological age (regression) and enable sensitivity analyses.
* **Applicable XAI**: Incorporation of SHAP values (emphasis on tree models) and approximate approaches for neural networks, promoting transparency and trust.
* **Privacy by Design**: Use of synthetic data for development/training; architecture and authentication (JWT) prepared for real data with consent.
* **Modern Web Architecture**: FastAPI + SQLAlchemy + JWT backend; MVP frontend in Streamlit (for defense), with planned migration to Next.js/React.
* **ML Operationalization**: Experiment tracking with MLflow (runs, parameters, metrics, artifacts); RF exported to ONNX/onnxruntime; MLP in PyTorch/TorchScript.

---

## 1.5 Thesis Structure and Scope

This work focuses on the development of a minimum viable product (MVP) that demonstrates the feasibility and effectiveness of personalized anti-aging recommendation systems. The system is designed to process individual genetic profiles, evaluate lifestyle factors, and provide personalized recommendations based on established scientific literature on aging interventions.

Although the current implementation uses synthetic data for privacy and ethical reasons, the architecture is designed to accommodate real genomic data as regulatory frameworks and user consent processes evolve.

The thesis contributes to the fields of personalized medicine, gerontechnology, and health informatics, demonstrating how modern machine learning and web technologies can be integrated to create accessible, interpretable, and privacy-preserving systems for managing aging-related health (HUANG et al., 2024). It also provides a framework for future research in applying explainable AI to genetic risk assessment and personalized health interventions.

---

# 2 Methodology

The methodology employed in this thesis follows a comprehensive systems development approach that integrates modern software engineering practices with rigorous data science methodologies. The development process was structured into five distinct phases, each designed to address specific technical and scientific challenges, ensuring the creation of a robust, scalable, and user-friendly personalized anti-aging recommendation system.

## 2.1 General Approach and Design Philosophy

The system architecture adopts a microservices approach with a clear separation of responsibilities, enabling independent development, testing, and deployment of different system components. This design philosophy ensures maintainability, scalability, and allows iterative development cycles that support rapid prototyping and testing of new features.

The development methodology emphasizes privacy-by-design principles, incorporating synthetic data generation, secure authentication mechanisms, and transparent data handling practices throughout the system’s lifecycle. All genetic and lifestyle data processing follows established ethical guidelines for human genetics research, with particular attention to data minimization and purpose limitation principles (World Health Organization, 2024).

## 2.2 Phase 1: Data Generation and Validation

The first phase focused on creating realistic synthetic genomic datasets that capture the statistical properties of real genetic variation while avoiding privacy concerns associated with actual human genetic data (Chen et al., 2025).

Key components included:

* **Selection and Modeling of Genetic Variants:** A comprehensive literature review identified 10 key single nucleotide polymorphisms (SNPs) associated with aging-related processes, including genes involved in cellular senescence (SIRT1\_rs7896005), DNA repair (FOXO3\_rs2802292), and lipid metabolism (APOE\_rs429358). Allele frequencies were based on population genetic data from the 1000 Genomes Project to ensure realistic representation of genetic diversity.

* **Integration of Lifestyle Factors:** The synthetic dataset incorporates six major lifestyle factors shown to influence aging outcomes: exercise frequency, dietary patterns, alcohol consumption, smoking history, sleep quality, and stress levels. These were selected based on epidemiological evidence linking lifestyle behaviors such as physical activity (Nitert et al., 2012) and diet (Gensous et al., 2019; Colman et al., 2009) to aging outcomes. Precision nutrition based on nutrigenetics and nutrigenomics is also a relevant area (Ramos-Lopez et al., 2017).

* **Risk Model Development:** A multiclass aging risk classification system was implemented using domain knowledge-based rules combined with statistical noise to simulate complex interactions between genetic and lifestyle factors. Risk categories (low, medium, high) were designed to reflect clinically meaningful distinctions in aging-related health outcomes such as healthspan and lifespan (Levine et al., 2018; Lu et al., 2019).

* **Data Validation and Quality Assurance:** Synthetic datasets were validated using chi-square tests to ensure Hardy–Weinberg equilibrium for genetic variants and statistical distribution analysis to confirm realistic ranges for lifestyle factors. Three datasets were generated: a training set (N=5,000) for machine learning model development and two smaller test sets (N=10 and N=1) for system demonstration and user testing.

## 2.3 Phase 2: Machine Learning and Deep Learning Model Development

The second phase involved developing and optimizing models to estimate biological age (regression task), comparing ensemble and deep learning methods.

* **Preprocessing Pipeline:** Standardized pipeline for heterogeneous data. Categorical variables (genotypes, demographics) encoded with one-hot encoding; continuous variables (habits) scaled using robust scaling.

* **Models and Training:** Two complementary approaches were implemented and compared:

  * **Random Forest Regressor:** Tree-based ensemble, interpretable and robust for tabular data. Tuned via cross-validation (e.g., n\_estimators 100–300; max\_depth 5–15) with grid/random search.
  * **Multi-Layer Perceptron (MLP) Regressor:** Feedforward neural network to capture nonlinear interactions. Typical architecture with 2 hidden layers (e.g., 64–128 neurons), ReLU activation, dropout, and early stopping.

* **Evaluation and Validation:** Stratified k-fold validation where applicable; key metrics included MAE, RMSE, and R². Training/validation error curves were recorded.

* **Experiment Tracking:** MLflow was used to log parameters, metrics, dataset versions, and artifacts (models and pipeline). RF exported to ONNX; MLP saved in PyTorch/TorchScript. Run comparisons supported model selection for production.

* **Explainable AI (XAI):** SHAP values calculated for RF (TreeExplainer) and approximate approaches for MLP (Kernel SHAP or feature importance proxies), with discussion of limitations and computational cost (MDPI, 2024; Nature, 2025).

## 2.4 Phase 3: System Architecture and Backend Development

The backend was developed in **FastAPI**, with data validation via **Pydantic** and ORM using **SQLAlchemy**, prioritizing high performance and static typing.

* **Authentication and Security:** JWT-based authentication (python-jose) and password hashing with bcrypt (passlib). Authorization applied to sensitive endpoints (user data upload/query).
* **Database:** PostgreSQL as main DBMS; schema included users, genetic profiles, lifestyle habits, and predictions. JSON fields allowed flexibility for varied genetic profiles.
* **API Endpoints:** Follow OpenAPI/Swagger auto-generated documentation. Main routes: registration/authentication, genetic CSV upload (schema validation), lifestyle data submission, prediction request (model selection rf|nn) with explanations.
* **Inference Service:** RF exported in ONNX and served with onnxruntime; MLP served with PyTorch/TorchScript. Preprocessing pipeline shared between training and inference for consistency. Artifacts versioned via MLflow.

## 2.5 Phase 4: Frontend Development and User Interface Design

For the MVP and defense, the frontend was implemented in **Streamlit**, consuming FastAPI endpoints (authentication, CSV upload, lifestyle submission, prediction with explanations). This choice accelerated iterations, simplified visualizations, and reduced build complexity.

* **Visualization and UX:** Explainability visualizations (e.g., SHAP) and metrics presented in Streamlit components, focusing on clarity and responsiveness. Loading states and informative error messages ensured smooth UX.
* **Post-Defense Plan:** Migration/expansion to Next.js/React is planned, maintaining API contracts. Chart.js and Jest testing will be introduced at that stage.
* **Client-Side Security:** Token storage, session timeouts, and input sanitization remain best practices.

## 2.6 Phase 5: Integration, Testing, and Deployment

The final phase focused on system integration, testing, and deployment readiness.

* **Containerization and Orchestration:** Services containerized for backend (FastAPI), database (Postgres), MLflow, frontend (Streamlit), and NGINX. Docker Compose ensured local reproducibility; MLflow exposed at :5000 with artifact volume.
* **Testing Strategy:** Unit and integration tests with Pytest for ML and API (goal ≥70% coverage). Basic load testing for `/predict`. More extensive frontend tests planned for Next.js/React stage.
* **Observability and Performance:** Metrics logged via MLflow; query and preprocessing optimizations; prediction response time targeted <2s.
* **Documentation and Deployment:** API documentation via Swagger (FastAPI), execution guide with Docker Compose, and MVP user manuals (Streamlit). NGINX routing updated for FastAPI as needed.

## 2.7 Ethical Considerations and Compliance

Throughout development, significant attention was given to ethical considerations in handling genetic data and applications of personalized medicine.

* **Data Protection and Privacy:** The system implements privacy-by-design, using synthetic data for training and development while maintaining an architecture capable of handling real genetic data under proper consent and security frameworks. Data processing follows GDPR and LGPD requirements for genetic data (World Health Organization, 2024).
* **Informed Consent and Transparency:** The user interface includes clear explanations of how genetic and lifestyle data are processed, what predictions mean in a clinical context, and the system’s current limitations. Users are informed that the system provides educational information rather than medical advice, a critical issue in direct-to-consumer genetic testing (Nature, 2022).
* **Bias Prevention and Fairness:** Synthetic data generation processes were designed to represent diverse populations and avoid systematic biases that could lead to unfair treatment of demographic groups. Model validation included performance evaluation across demographic categories.
* **Scientific Integrity:** All genetic associations and lifestyle recommendations are based on peer-reviewed scientific literature, with proper citations and acknowledgment of evidence quality and limitations (Sen et al., 2016; Alegría-Torres, Baccarelli & Bollati, 2011).

## 2.8 Validation and Evaluation Metrics

System validation employed multiple approaches to ensure scientific rigor and practical utility:

* **Technical Validation:** Machine learning models were validated using standard metrics appropriate for multiclass classification problems, including weighted macro F1 score, accuracy, recall, and AUC-ROC. Cross-validation procedures ensured robust performance estimates and generalization assessment.
* **Usability Testing:** User interface components were tested with representative users to identify usability issues and optimize presentation of complex genetic and lifestyle information. Feedback was iteratively incorporated throughout development.
* **Synthetic Data Validation:** Generated synthetic datasets were validated against known population genetic parameters and realistic lifestyle factor distributions to ensure scientific plausibility while preserving privacy.

---

