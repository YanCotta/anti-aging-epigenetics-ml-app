# GitHub Issues for Anti-Aging ML App Development Plan

This document contains all GitHub issues derived from DEV_PLAN.md.

## Milestones

### Phase 1: Setup + Data
**Due Date:** 2024-09-07

Scale synthetic dataset and prepare data infrastructure

### Phase 2: Backend + ML
**Due Date:** 2024-09-18

Implement FastAPI backend with ML models and MLFlow integration

### Phase 3: Frontend + Integration
**Due Date:** 2024-09-29

Build Streamlit MVP and integrate with backend

### Phase 4: Docker, Testing, Validation
**Due Date:** 2024-10-06

Complete containerization and testing infrastructure

### Phase 5: Thesis + Demo
**Due Date:** 2024-10-15

Finalize thesis materials and demo preparation

## Labels

- **phase-1** (`0075ca`): Phase 1: Setup + Data
- **phase-2** (`0075ca`): Phase 2: Backend + ML
- **phase-3** (`0075ca`): Phase 3: Frontend + Integration
- **phase-4** (`0075ca`): Phase 4: Docker, Testing, Validation
- **phase-5** (`0075ca`): Phase 5: Thesis + Demo
- **backend** (`d73a4a`): Backend development
- **frontend** (`a2eeef`): Frontend development
- **ml** (`0e8a16`): Machine Learning related
- **infrastructure** (`f9d0c4`): Infrastructure and DevOps
- **documentation** (`7057ff`): Documentation and thesis
- **testing** (`fef2c0`): Testing and validation
- **high-priority** (`d93f0b`): High priority task
- **medium-priority** (`fbca04`): Medium priority task
- **low-priority** (`0e8a16`): Low priority task
- **backlog** (`c5def5`): Backlog item

## Issues

### Pivot Update: Linear Regression Baseline First

To ensure rigorous model comparison, we are introducing a Linear Regression baseline as an explicit task in Phase 2. This baseline will be trained and logged to MLflow immediately after data upload endpoints (Issue #4). Subsequent models (Random Forest, MLP) must be logged to the same experiment to enable side-by-side comparison using standardized regression metrics (RMSE, R², MAE) and shared artifacts (model + preprocessor).

---

### Issue #1: Scale synthetic dataset to 5000 records with improved validation

**Labels:** phase-1, ml, high-priority

**Milestone:** Phase 1: Setup + Data

## Description
Scale the current synthetic dataset from its current size to approximately 5000 records to ensure model viability during training. The dataset should maintain proper distributions and realistic demographic + habits data.

## Acceptance Criteria
- [ ] Generate synthetic dataset with N≈5000 records
- [ ] Validate data distributions are realistic and consistent
- [ ] Ensure demographic and habits data follows expected patterns
- [ ] Create both training dataset (train.csv) and small test datasets
- [ ] Place datasets under `backend/api/data/datasets/` directory
- [ ] Verify no data quality issues (missing values, outliers, inconsistencies)
- [ ] Document data generation process and validation rules

## Implementation Notes
- Use existing generator in `backend/api/data/generator.py` as starting point
- Keep current regression target (biological_age) for continuity
- Consider adding data validation checks and summary statistics
- Ensure reproducibility with random seeds

## Files to Modify
- `backend/api/data/generator.py`
- `backend/api/data/datasets/` (create directory if needed)

## Definition of Done
- Datasets are generated and placed in correct location
- Data quality validation passes
- Dataset size meets requirements (≈5000 records)
- Documentation is updated with data generation process


---

### Issue #41: Train Linear Regression baseline with MLflow tracking and comparison protocol

**Labels:** phase-2, ml, high-priority

**Milestone:** Phase 2: Backend + ML

## Description
Train a Linear Regression baseline on `backend/api/data/datasets/train.csv`, log metrics and artifacts to MLflow, and establish a comparison protocol that will also be applied to Random Forest and MLP runs.

## Acceptance Criteria
- [ ] Train Linear Regression using scikit-learn on the prepared features
- [ ] Use a consistent preprocessing pipeline (serialize preprocessor)
- [ ] Log to MLflow: params, RMSE, R², MAE; save model and preprocessor artifacts
- [ ] Create or reuse an experiment name shared by RF/MLP for comparison
- [ ] Document how to view comparisons in MLflow UI (runs table, metrics plot)
- [ ] Add a brief summary of baseline results to CHANGELOG

## Implementation Notes
- Script location: `backend/api/ml/train_linear.py` (ensure runnable via CLI and module)
- Reuse `DataPreprocessor` for fit/transform; persist artifacts with joblib
- Use fixed random seed for deterministic splits
- Consider adding a small README note under `backend/api/ml/` on how to run

## Files to Create/Modify
- `backend/api/ml/train_linear.py` (ensure robust import when executed directly)
- `backend/api/ml/preprocessor.py` (ensure save/load methods)
- `docs/CHANGELOG.md` (append baseline result summary)

## Definition of Done
- MLflow run exists with metrics and artifacts
- Artifacts saved under a deterministic output directory
- Baseline metrics recorded and documented for future comparison
- Instructions to reproduce are present

---

### Issue #42: Implement Multivariate Statistical Analysis for Feature Groupings and Colinear Relationships

**Labels:** phase-2, ml, high-priority

**Milestone:** Phase 2: Backend + ML

## Description
Implement clustering/grouping analysis and canonical correlation analysis to discover colinear relationships between dataset variables. The hypothesis is that variables can be grouped (e.g., SNPs/methylation as genetic factors, lifestyle variables as behavioral factors) with different weights and impacts on model performance based on their colinear relationships.

## Acceptance Criteria
- [ ] Implement clustering analysis to group similar variables (K-means, hierarchical clustering)
- [ ] Implement canonical correlation analysis to identify relationships between variable groups
- [ ] Create feature grouping module that categorizes variables by type (genetic, lifestyle, demographic, health markers)
- [ ] Analyze colinear relationships within and between groups
- [ ] Generate comprehensive multivariate analysis report with visualizations
- [ ] Integrate findings into preprocessing pipeline for group-aware feature engineering
- [ ] Document feature weights and group impacts for model interpretation
- [ ] Create validation metrics for group-based feature importance

## Implementation Notes
- Use scikit-learn for clustering algorithms and feature analysis
- Implement canonical correlation using scipy.stats or specialized libraries
- Create visualization functions for correlation matrices and dendrograms
- Consider Principal Component Analysis (PCA) within groups for dimensionality reduction
- Ensure analysis is integrated with existing validation pipeline

## Expected Feature Groups
- **Genetic Group**: SNPs (APOE, FOXO3, SIRT1, etc.) and CpG methylation sites
- **Lifestyle Group**: exercise_frequency, sleep_hours, stress_level, diet_quality, smoking, alcohol
- **Demographic Group**: age, gender, height, weight, BMI
- **Health Markers Group**: systolic_bp, diastolic_bp, cholesterol, glucose, telomere_length
- **Environmental Group**: pollution_exposure, sun_exposure, occupation_stress

## Files to Create/Modify
- `backend/api/ml/multivariate_analysis.py` (new - clustering and canonical correlation)
- `backend/api/data/feature_groups.py` (new - feature grouping definitions)
- `backend/api/data/validation.py` (update - integrate multivariate analysis)
- `backend/api/ml/preprocessor.py` (update - group-aware preprocessing)

## Definition of Done
- Multivariate analysis reveals clear feature groupings and colinear relationships
- Analysis results are documented with visualizations and statistical significance
- Feature groups are integrated into preprocessing pipeline
- Model training can leverage group-based feature engineering
- Validation report includes multivariate analysis findings

---

## CRITICAL NEW ISSUES - Address Before Original Roadmap

### Issue #43: Critical Synthetic Data Realism Overhaul - Age Correlation Fix

**Labels:** phase-1, ml, critical-priority

**Milestone:** Immediate Fix Required

## Description
**CRITICAL FINDING**: Current age-biological age correlation (0.945) is scientifically implausible and invalidates all model results. Real aging research shows correlations of 0.6-0.8 maximum. This high correlation makes models appear artificially successful and prevents realistic evaluation.

## Acceptance Criteria
- [ ] **URGENT**: Reduce age-biological age correlation to realistic range (0.7-0.8)
- [ ] Add substantial individual variation in aging rates (genetic and environmental)
- [ ] Introduce realistic noise and measurement error to biological markers
- [ ] Add outliers representing exceptional agers (both directions)
- [ ] Implement non-linear aging patterns especially at age extremes
- [ ] Add sex-specific aging differences in feature relationships
- [ ] Include population structure variation (ancestry effects on aging)
- [ ] Validate against published aging biomarker correlations from literature
- [ ] Add missing values to simulate real-world data collection challenges

## Scientific Justification
- Real epigenetic clocks (Horvath, Hannum) show R² ~0.7-0.85 with 3-8 year errors
- Individual aging rates vary dramatically due to genetics, lifestyle, environment
- Biological aging involves stochastic processes that create natural variance
- Current synthetic data lacks complexity found in real aging biology

## Implementation Notes
- Revise biological age calculation to include more noise and individual variation
- Add genetic modifiers that create realistic inter-individual differences
- Implement environmental stress factors that modulate aging rates
- Add measurement error appropriate for biological assays
- Include rare variants and exceptional aging phenotypes

## Files to Modify
- `backend/api/data/generator.py` (major revision required)
- `backend/api/data/validation.py` (update correlation thresholds)
- All existing model training scripts (results will change significantly)

## Definition of Done
- Age-biological age correlation reduced to 0.7-0.8 range
- Individual variation in aging rates reflects biological reality
- Model performance becomes realistic (R² ~0.6-0.8, MAE 4-8 years)
- Synthetic data validated against real aging research benchmarks

---

### Issue #44: Genomics-Specific Data Quality and Preprocessing Pipeline

**Labels:** phase-1, ml, high-priority

**Milestone:** Data Quality Foundation

## Description
**CRITICAL FINDING**: Current preprocessing treats genetic data like generic features, ignoring fundamental genomics principles. SNPs require specialized handling for population genetics, linkage disequilibrium, and proper genetic models.

## Acceptance Criteria
- [ ] Implement Hardy-Weinberg equilibrium testing for SNP quality control
- [ ] Add allele frequency validation against reference populations
- [ ] Implement proper genetic encoding (additive, dominant, recessive models)
- [ ] Add linkage disequilibrium analysis and pruning capabilities
- [ ] Create methylation-specific preprocessing (beta vs M-value transformations)
- [ ] Implement population stratification controls and ancestry informative markers
- [ ] Add batch effect detection and correction for methylation data
- [ ] Create feature-type aware preprocessing (genetic vs epigenetic vs lifestyle)
- [ ] Implement missing genotype imputation strategies
- [ ] Add genetic pathway-based feature grouping

## Scientific Requirements
- SNP call rates >95%, MAF >0.05, HWE p-value >1e-6
- Linkage disequilibrium pruning (r² >0.8 threshold)
- Proper handling of X-chromosome and mitochondrial variants
- Population structure correction using principal components
- Methylation probe filtering (cross-reactive, polymorphic sites)

## Implementation Notes
- Use specialized genomics libraries (PyVCF, allel, methylation analysis tools)
- Implement standard GWAS quality control pipelines
- Add ancestry estimation and population stratification
- Create genetic model-aware feature encoding
- Ensure preprocessing maintains biological interpretation

## Files to Create/Modify
- `backend/api/ml/genomics_preprocessing.py` (new - specialized genetic preprocessing)
- `backend/api/data/genetic_qc.py` (new - quality control pipeline)
- `backend/api/data/population_structure.py` (new - ancestry controls)
- `backend/api/ml/preprocessor.py` (update - integrate genomics preprocessing)

## Definition of Done
- Genetic data preprocessing follows genomics best practices
- Population structure and quality control implemented
- Feature encoding respects genetic inheritance models
- Preprocessing maintains biological interpretability
- Integration with existing ML pipeline maintained

---

### Issue #45: Realistic Model Performance Baselines and Benchmarking

**Labels:** phase-2, ml, high-priority

**Milestone:** Realistic Model Evaluation

## Description
**CRITICAL FINDING**: Current model performance (R² 0.97, MAE 2 years) is implausibly high for biological aging prediction. Real aging research achieves R² 0.6-0.8 with 4-8 year errors. Need realistic baselines and comparison with published aging clocks.

## Acceptance Criteria
- [ ] Establish realistic performance targets based on aging research literature
- [ ] Implement comparison with published aging clocks (Horvath, Hannum, PhenoAge)
- [ ] Add statistical significance testing for model comparisons
- [ ] Create age-stratified performance analysis (young vs old predictions)
- [ ] Implement sex-specific model evaluation
- [ ] Add confidence intervals and uncertainty quantification
- [ ] Create learning curves to assess data requirements
- [ ] Implement cross-validation with proper biological stratification
- [ ] Add feature importance stability analysis across different data splits
- [ ] Document expected vs actual performance gaps

## Realistic Performance Targets
- **Excellent aging predictor**: R² 0.7-0.8, MAE 4-6 years
- **Good aging predictor**: R² 0.6-0.7, MAE 6-8 years
- **Research-grade baseline**: R² 0.5-0.6, MAE 8-10 years

## Implementation Notes
- Use nested cross-validation for proper model selection
- Implement bootstrap confidence intervals
- Add permutation tests for statistical significance
- Create age-group stratified evaluation
- Compare against simple baselines (age-only models)

## Files to Create/Modify
- `backend/api/ml/aging_benchmarks.py` (new - literature benchmarks)
- `backend/api/ml/evaluation.py` (enhance - realistic metrics)
- `backend/api/ml/statistical_tests.py` (new - significance testing)
- `notebooks/02_realistic_model_benchmarking.ipynb` (new)

## Definition of Done
- Model performance targets aligned with aging research reality
- Statistical significance of model improvements validated
- Age and sex-specific performance documented
- Comparison with published aging clocks implemented
- Performance gaps and limitations clearly documented

---

### Issue #46: Advanced Feature Engineering for Aging Biology

**Labels:** phase-2, ml, high-priority

**Milestone:** Biologically-Informed Features

## Description
**CRITICAL FINDING**: Current feature engineering ignores known aging biology. Missing gene-environment interactions, pathway-based features, and aging-specific transformations that are critical for realistic aging prediction.

## Acceptance Criteria
- [ ] Implement aging pathway-based feature grouping (telomere, DNA repair, senescence)
- [ ] Add gene-environment interaction terms (genetic variants × lifestyle factors)
- [ ] Create epigenetic clock-inspired feature combinations
- [ ] Implement non-linear age transformations for biological markers
- [ ] Add polygenic risk scores for aging-related diseases
- [ ] Create composite lifestyle scores (Mediterranean diet, exercise patterns)
- [ ] Implement sex-specific feature engineering
- [ ] Add temporal feature patterns (age-related changes)
- [ ] Create aging biomarker ratios and composite indices
- [ ] Implement feature selection based on aging biology literature

## Biological Features to Add
- **Telomere Biology**: telomere length interactions with genetic variants
- **DNA Damage Response**: DNA repair gene variants × environmental exposures  
- **Cellular Senescence**: senescence pathway genes × oxidative stress markers
- **Metabolic Aging**: metabolic syndrome components × genetic susceptibility
- **Inflammatory Aging**: inflammatory markers × lifestyle factors

## Implementation Notes
- Use biological pathway databases (KEGG, Reactome, GO)
- Implement literature-based feature combinations
- Add aging-specific non-linear transformations
- Create interaction terms based on known biology
- Ensure feature interpretability for aging research

## Files to Create/Modify
- `backend/api/ml/aging_features.py` (new - aging-specific feature engineering)
- `backend/api/ml/pathway_analysis.py` (new - biological pathway features)
- `backend/api/ml/gene_environment_interactions.py` (new)
- `backend/api/data/aging_pathways.py` (new - pathway definitions)

## Definition of Done
- Feature engineering incorporates known aging biology
- Gene-environment interactions implemented
- Pathway-based features created and validated
- Feature interpretability maintained for aging research
- Literature-based feature combinations implemented

---

### Issue #47: Statistical Rigor and Multiple Testing Correction

**Labels:** phase-2, ml, testing, high-priority

**Milestone:** Scientific Statistical Standards

## Description
**CRITICAL FINDING**: Current analysis lacks statistical rigor expected in genomics research. Missing multiple testing correction, confidence intervals, and proper hypothesis testing frameworks essential for scientific validity.

## Acceptance Criteria
- [ ] Implement False Discovery Rate (FDR) correction for multiple testing
- [ ] Add Bonferroni correction for family-wise error rate control
- [ ] Implement bootstrap confidence intervals for all metrics
- [ ] Add permutation tests for feature importance validation
- [ ] Create power analysis for sample size requirements
- [ ] Implement proper cross-validation with biological stratification
- [ ] Add statistical tests for model comparison (McNemar, Wilcoxon)
- [ ] Create effect size calculations (Cohen's d, Cliff's delta)
- [ ] Implement stability analysis for feature selection
- [ ] Add reproducibility metrics and random seed management

## Statistical Tests Required
- **Multiple Testing**: FDR (Benjamini-Hochberg), Bonferroni, Holm-Sidak
- **Model Comparison**: Paired t-tests, McNemar test, DeLong test for AUC
- **Effect Sizes**: Cohen's d for mean differences, Cliff's delta for distributions
- **Stability**: Jaccard index for feature selection, bootstrap aggregation

## Implementation Notes
- Use statsmodels for advanced statistical testing
- Implement proper p-value correction workflows
- Add confidence interval calculation for all metrics
- Create reproducible statistical analysis pipelines
- Document statistical assumptions and validations

## Files to Create/Modify
- `backend/api/ml/statistical_tests.py` (new - comprehensive statistical testing)
- `backend/api/ml/multiple_testing.py` (new - p-value corrections)
- `backend/api/ml/effect_sizes.py` (new - effect size calculations)
- `backend/api/ml/reproducibility.py` (new - reproducibility metrics)

## Definition of Done
- Multiple testing correction implemented for all analyses
- Statistical significance properly calculated and reported
- Confidence intervals provided for all performance metrics
- Effect sizes calculated and interpreted
- Reproducibility and stability metrics implemented

---

### Issue #48: Repository Structure Cleanup and Code Organization

**Labels:** infrastructure, cleanup, medium-priority

**Milestone:** Clean Codebase Foundation

## Description
**IDENTIFIED NEED**: Repository contains outdated, duplicate, and misplaced files that create confusion and potential for using incorrect code. Need systematic cleanup and proper organization.

## Acceptance Criteria
- [ ] Audit all Python scripts for data leakage issues and fix or remove
- [ ] Consolidate duplicate preprocessing functionality
- [ ] Move misplaced files to appropriate directories
- [ ] Remove outdated placeholder code and comments
- [ ] Standardize import patterns and dependency management
- [ ] Create clear separation between training and inference code
- [ ] Document file purposes and relationships
- [ ] Implement consistent naming conventions
- [ ] Remove unused dependencies and imports
- [ ] Create proper module structure with `__init__.py` files

## Files to Audit and Clean
- `backend/api/ml/train.py` (contains data leakage, needs fixing or removal)
- `backend/api/ml/train_linear.py` (newly fixed, verify consistency)
- `backend/fastapi_app/` (ensure no duplicate preprocessing logic)
- All import statements and dependency management
- Placeholder code and TODO comments

## Implementation Notes
- Create deprecation plan for outdated scripts
- Ensure no breaking changes to working functionality
- Document migration paths for any moved files
- Test all remaining scripts for data leakage issues
- Standardize error handling and logging

## Files to Create/Modify
- Document cleanup in `docs/code_organization.md`
- Update all import statements affected by moves
- Remove or fix identified problematic files
- Update documentation to reflect new structure

## Definition of Done
- No duplicate or contradictory functionality exists
- All remaining code follows consistent patterns
- File organization is logical and well-documented
- No data leakage issues remain in any scripts
- Dependencies are clean and minimal

---

### Issue #2: Validate synthetic data distributions and implement quality checks

**Labels:** phase-1, ml, testing, medium-priority

**Milestone:** Phase 1: Setup + Data

## Description
Implement comprehensive validation for the synthetic dataset to ensure realistic distributions, proper correlations between features, and data quality standards that will support effective ML model training.

## Acceptance Criteria
- [ ] Create data validation pipeline/script
- [ ] Validate age distributions follow realistic patterns
- [ ] Verify genetic markers have appropriate frequencies
- [ ] Check lifestyle/habits data for logical consistency
- [ ] Implement automated data quality checks
- [ ] Generate data summary reports and visualizations
- [ ] Document validation rules and thresholds
- [ ] Create test suite for data validation

## Implementation Notes
- Build validation functions that can be reused
- Include statistical tests for distribution validation
- Check for correlation patterns between related variables
- Ensure no impossible combinations (e.g., age vs certain health metrics)

## Files to Create/Modify
- `backend/api/data/validation.py` (new)
- `backend/api/data/datasets/validation_report.md` (generated)
- Add validation tests to test suite

## Definition of Done
- Data validation pipeline is implemented and passing
- Validation report shows acceptable data quality
- Automated checks prevent bad data from being used
- Documentation covers validation process and standards


---

### Issue #3: Implement FastAPI authentication system with JWT and core user endpoints

**Labels:** phase-2, backend, high-priority

**Milestone:** Phase 2: Backend + ML

## Description
Implement secure authentication system for the FastAPI backend using JWT tokens, password hashing, and OAuth2 flow. Create core user management endpoints as foundation for the ML application.

## Acceptance Criteria
- [ ] Implement JWT token generation and validation using python-jose
- [ ] Add password hashing with passlib[bcrypt]
- [ ] Create POST /signup endpoint with user registration
- [ ] Create POST /token endpoint with OAuth2PasswordRequestForm
- [ ] Implement JWT authentication dependency for protected routes
- [ ] Add proper error handling and validation
- [ ] Configure CORS for frontend integration
- [ ] Add health check endpoint GET /health
- [ ] Create Pydantic schemas for request/response validation
- [ ] Add comprehensive API documentation in FastAPI Swagger

## Implementation Notes
- Use SQLAlchemy models for user persistence
- Follow FastAPI security best practices
- Ensure password validation and security requirements
- Add rate limiting considerations for production

## Files to Create/Modify
- `backend/fastapi_app/auth.py` (enhance existing)
- `backend/fastapi_app/schemas.py` (enhance existing)
- `backend/fastapi_app/main.py` (add endpoints)
- `backend/fastapi_app/db.py` (ensure user models)

## Definition of Done
- Authentication endpoints are working and tested
- JWT tokens are properly generated and validated
- Swagger documentation is complete and accurate
- Security best practices are implemented
- Integration tests pass for auth flow


---

### Issue #4: Create genetic data upload and habits submission endpoints with validation

**Labels:** phase-2, backend, high-priority

**Milestone:** Phase 2: Backend + ML

## Description
Implement secure endpoints for users to upload genetic data (CSV format) and submit lifestyle/habits information. Include proper validation, persistence, and user association.

## Acceptance Criteria
- [ ] Create POST /upload-genetic endpoint with CSV file upload
- [ ] Implement strict CSV schema validation for genetic data
- [ ] Create POST /submit-habits endpoint with JSON payload validation
- [ ] Add user authentication requirements for both endpoints
- [ ] Persist latest genetic profile per user (replace previous uploads)
- [ ] Store habits data with versioning/timestamps
- [ ] Add comprehensive input validation and error handling
- [ ] Implement file size and format restrictions for uploads
- [ ] Add progress tracking for large file uploads
- [ ] Create endpoints to retrieve user's current genetic profile and habits

## Implementation Notes
- Use Pydantic for JSON validation of habits data
- Implement CSV parsing with pandas/csv library
- Add database models for genetic profiles and habits
- Consider file storage strategy (database vs file system)
- Ensure proper cleanup of old files if applicable

## Files to Create/Modify
- `backend/fastapi_app/main.py` (add endpoints)
- `backend/fastapi_app/schemas.py` (add validation schemas)
- `backend/fastapi_app/db.py` (add database models)
- `backend/fastapi_app/validation.py` (new file for CSV validation)

## Definition of Done
- Upload endpoints accept and validate genetic data correctly
- Habits submission handles all required lifestyle factors
- Data is properly associated with authenticated users
- File validation prevents invalid or malicious uploads
- API documentation includes request/response examples


---

### Issue #5: Create unified ML preprocessing pipeline for training and inference

**Labels:** phase-2, ml, high-priority

**Milestone:** Phase 2: Backend + ML

## Description
Develop a consistent preprocessing pipeline that can be used for both model training and real-time prediction. Ensure feature engineering, scaling, and transformation steps are reproducible and aligned.

## Acceptance Criteria
- [ ] Create preprocessing pipeline class with fit/transform methods
- [ ] Implement feature engineering for genetic and habits data
- [ ] Add data scaling and normalization steps
- [ ] Handle missing values and outliers appropriately
- [ ] Ensure pipeline serialization for inference consistency
- [ ] Add preprocessing validation and sanity checks
- [ ] Create unit tests for all preprocessing steps
- [ ] Document feature engineering decisions and rationale
- [ ] Implement pipeline versioning for reproducibility

## Implementation Notes
- Use scikit-learn Pipeline for consistency
- Ensure preprocessing steps are identical between training and prediction
- Consider feature selection and dimensionality reduction
- Add logging for preprocessing steps and transformations

## Files to Create/Modify
- `backend/api/ml/preprocessor.py` (enhance existing)
- `backend/fastapi_app/ml/preprocessor.py` (create aligned version)
- `backend/api/ml/features.py` (new - feature engineering)
- Add preprocessing tests

## Definition of Done
- Preprocessing pipeline works consistently for training and inference
- All feature engineering steps are documented and tested
- Pipeline can be serialized and loaded for production use
- Validation ensures data quality before model training/prediction


---

### Issue #6: Train Random Forest baseline model with ONNX export and SHAP explanations

**Labels:** phase-2, ml, high-priority

**Milestone:** Phase 2: Backend + ML

## Description
Implement Random Forest model training pipeline with MLFlow tracking, ONNX export for efficient inference, and SHAP explanations for model interpretability.

## Acceptance Criteria
- [ ] Implement Random Forest training with hyperparameter optimization
- [ ] Log training metrics, parameters, and artifacts to MLFlow
- [ ] Export trained model to ONNX format for inference
- [ ] Implement SHAP explanations for model predictions
- [ ] Add model evaluation metrics (R²/RMSE for regression or F1/accuracy for classification)
- [ ] Create model validation pipeline with cross-validation
- [ ] Implement feature importance analysis
- [ ] Add model performance visualization and reports
- [ ] Ensure reproducibility with random seeds and versioning

## Implementation Notes
- Use scikit-learn RandomForestRegressor/Classifier
- Implement grid search or random search for hyperparameter tuning
- Use skl2onnx for ONNX conversion
- SHAP TreeExplainer for fast explanations on tree models
- Consider ensemble methods and model validation strategies

## Files to Create/Modify
- `backend/api/ml/train.py` (enhance existing)
- `backend/api/ml/models/random_forest.py` (new)
- `backend/api/ml/explain.py` (new - SHAP explanations)
- `backend/api/ml/evaluate.py` (new - model evaluation)

## Definition of Done
- Random Forest model trains successfully with good performance
- MLFlow tracking captures all relevant metrics and artifacts
- ONNX export works and produces consistent predictions
- SHAP explanations provide meaningful feature attributions
- Model evaluation meets performance thresholds


---

### Issue #7: Create MLP neural network model with PyTorch and MLFlow tracking

**Labels:** phase-2, ml, medium-priority

**Milestone:** Phase 2: Backend + ML

## Description
Develop a Multi-Layer Perceptron (MLP) using PyTorch as an alternative model to the Random Forest. Include proper training pipeline, MLFlow integration, and model comparison capabilities.

## Acceptance Criteria
- [ ] Design MLP architecture appropriate for the problem size
- [ ] Implement PyTorch training loop with proper optimization
- [ ] Add MLFlow tracking for neural network experiments
- [ ] Implement early stopping and learning rate scheduling
- [ ] Add model evaluation and comparison with Random Forest
- [ ] Export model for inference (TorchScript or ONNX)
- [ ] Implement neural network explanations (SHAP Kernel or other methods)
- [ ] Add visualization of training progress and model performance
- [ ] Create hyperparameter tuning pipeline for MLP

## Implementation Notes
- Keep architecture simple but effective (2-3 hidden layers)
- Use appropriate activation functions and regularization
- Implement proper train/validation/test splits
- Consider batch normalization and dropout for regularization
- Use appropriate loss function for the target variable type

## Files to Create/Modify
- `backend/api/ml/models/mlp.py` (new)
- `backend/api/ml/train_mlp.py` (new)
- `backend/api/ml/torch_utils.py` (new - PyTorch utilities)
- Update `backend/api/ml/explain.py` for neural network explanations

## Definition of Done
- MLP model trains successfully and converges
- MLFlow captures neural network metrics and artifacts
- Model performance is comparable to Random Forest baseline
- Model can be exported and loaded for inference
- Neural network explanations are implemented and functional


---

### Issue #8: Implement prediction endpoint with model selection and explanations

**Labels:** phase-2, backend, ml, high-priority

**Milestone:** Phase 2: Backend + ML

## Description
Create the core prediction endpoint that loads trained models, processes user data, and returns predictions with explanations. Support both Random Forest and MLP models with dynamic selection.

## Acceptance Criteria
- [ ] Create GET /predict endpoint with model_type parameter (rf|nn)
- [ ] Load latest trained models from MLFlow or local storage
- [ ] Fetch user's latest genetic profile and habits data
- [ ] Apply consistent preprocessing pipeline
- [ ] Return prediction with confidence intervals/probabilities
- [ ] Include SHAP explanations for prediction interpretability
- [ ] Add prediction caching for performance optimization
- [ ] Implement proper error handling for missing data or failed predictions
- [ ] Add prediction logging for monitoring and improvement

## Implementation Notes
- Use the same preprocessing pipeline as training
- Implement model loading and caching strategy
- Ensure predictions are consistent between ONNX and original models
- Add validation for input data completeness
- Consider prediction versioning for reproducibility

## Files to Create/Modify
- `backend/fastapi_app/main.py` (add prediction endpoint)
- `backend/fastapi_app/ml/predict.py` (enhance existing)
- `backend/fastapi_app/ml/model_loader.py` (new)
- `backend/fastapi_app/schemas.py` (add prediction response schemas)

## Definition of Done
- Prediction endpoint works for both model types
- Predictions include explanations and confidence measures
- Error handling covers all edge cases
- Performance is acceptable for real-time use
- API documentation includes prediction examples


---

### Issue #9: Develop Streamlit MVP with complete FastAPI integration

**Labels:** phase-3, frontend, high-priority

**Milestone:** Phase 3: Frontend + Integration

## Description
Create a functional Streamlit application that provides a complete user interface for the anti-aging ML app, integrating with all FastAPI endpoints for authentication, data upload, habits submission, and predictions.

## Acceptance Criteria
- [ ] Implement user registration and login interface
- [ ] Create genetic data upload interface with drag-and-drop CSV support
- [ ] Build habits/lifestyle questionnaire form
- [ ] Add prediction interface with model selection (RF vs MLP)
- [ ] Display prediction results with visualizations and explanations
- [ ] Implement session management for user authentication
- [ ] Add data validation and error handling in the UI
- [ ] Create dashboard for user's historical predictions
- [ ] Add loading states and progress indicators
- [ ] Ensure responsive design and good UX

## Implementation Notes
- Use Streamlit's session state for user management
- Implement proper error handling and user feedback
- Add data visualization for SHAP explanations
- Consider using Streamlit components for enhanced UI elements
- Ensure secure communication with FastAPI backend

## Files to Create/Modify
- `streamlit_app/app.py` (enhance existing)
- `streamlit_app/pages/` (create page modules)
- `streamlit_app/utils/api_client.py` (new - FastAPI client)
- `streamlit_app/components/` (new - reusable UI components)

## Definition of Done
- Complete user workflow from registration to prediction works
- All FastAPI endpoints are integrated and functional
- UI provides good user experience with proper feedback
- Error handling prevents application crashes
- Application is ready for thesis defense demo


---

### Issue #10: Document React/Next.js migration strategy and stabilize API contracts

**Labels:** phase-3, frontend, documentation, medium-priority

**Milestone:** Phase 3: Frontend + Integration

## Description
Prepare for post-defense migration from Streamlit to a production-ready React/Next.js frontend by documenting the migration strategy, stabilizing API contracts, and creating reusable components plan.

## Acceptance Criteria
- [ ] Document complete API contract specifications with OpenAPI/Swagger
- [ ] Create migration roadmap from Streamlit to React/Next.js
- [ ] Design component architecture for React frontend
- [ ] Identify reusable UI patterns and state management needs
- [ ] Plan authentication flow for React application
- [ ] Document data flow and API integration patterns
- [ ] Create wireframes/mockups for key user interfaces
- [ ] Establish frontend development standards and conventions
- [ ] Plan testing strategy for React components

## Implementation Notes
- Ensure API contracts are stable and well-documented
- Consider using TypeScript for type safety in React app
- Plan for modern React patterns (hooks, context, etc.)
- Consider state management solutions (Redux, Zustand, etc.)
- Plan for responsive design and accessibility

## Files to Create/Modify
- `docs/api_specification.md` (new)
- `docs/react_migration_plan.md` (new)
- `frontend/README.md` (update with migration plan)
- `docs/frontend_architecture.md` (new)

## Definition of Done
- API contracts are fully documented and stable
- Migration plan is detailed and actionable
- React component architecture is planned
- Documentation provides clear guidance for frontend development


---

### Issue #11: Implement end-to-end integration testing for complete user workflows

**Labels:** phase-3, testing, medium-priority

**Milestone:** Phase 3: Frontend + Integration

## Description
Create comprehensive integration tests that validate the complete user journey from registration through prediction, ensuring all components work together correctly in the Docker environment.

## Acceptance Criteria
- [ ] Set up integration testing framework and environment
- [ ] Create test scenarios for complete user workflows
- [ ] Test user registration, login, and session management
- [ ] Validate genetic data upload and processing flow
- [ ] Test habits submission and data persistence
- [ ] Verify prediction generation with both model types
- [ ] Test explanation generation and display
- [ ] Add performance testing for prediction endpoints
- [ ] Implement automated test data generation
- [ ] Create test reporting and monitoring

## Implementation Notes
- Use pytest for Python testing framework
- Consider using Playwright or Selenium for UI testing
- Test against actual Docker containers
- Include negative test cases and error scenarios
- Add load testing for prediction endpoints

## Files to Create/Modify
- `tests/integration/` (new directory)
- `tests/integration/test_user_workflows.py` (new)
- `tests/integration/test_api_integration.py` (new)
- `tests/integration/conftest.py` (new - test configuration)

## Definition of Done
- All major user workflows are covered by integration tests
- Tests run reliably in CI/CD environment
- Performance requirements are validated
- Test coverage includes error scenarios and edge cases


---

### Issue #12: Finalize Docker infrastructure with health checks and service optimization

**Labels:** phase-4, infrastructure, high-priority

**Milestone:** Phase 4: Docker, Testing, Validation

## Description
Complete the Docker Compose infrastructure by adding proper health checks, optimizing service configurations, and ensuring reliable container orchestration for all services including MLFlow.

## Acceptance Criteria
- [ ] Add comprehensive health checks for all services
- [ ] Optimize Docker Compose service dependencies and startup order
- [ ] Configure NGINX routing to properly proxy FastAPI endpoints
- [ ] Ensure MLFlow service integration and persistence
- [ ] Add environment variable configuration for all services
- [ ] Implement proper logging and monitoring for containers
- [ ] Add development vs production environment configurations
- [ ] Create service restart policies and failure handling
- [ ] Document Docker setup and troubleshooting guide

## Implementation Notes
- Use Docker Compose depends_on with health conditions
- Configure proper network communication between services
- Ensure persistent volumes for database and MLFlow data
- Add resource limits and constraints for containers
- Consider multi-stage builds for optimization

## Files to Create/Modify
- `antiaging-mvp/docker-compose.yml` (update existing)
- `antiaging-mvp/docker-compose.prod.yml` (new - production config)
- `antiaging-mvp/.env.example` (new - environment template)
- `docs/docker_setup.md` (new - Docker documentation)

## Definition of Done
- All services start reliably with proper dependencies
- Health checks accurately reflect service status
- NGINX properly routes requests to FastAPI
- MLFlow integration works seamlessly
- Documentation covers setup and troubleshooting


---

### Issue #13: Create comprehensive testing suite with ≥70% coverage

**Labels:** phase-4, testing, high-priority

**Milestone:** Phase 4: Docker, Testing, Validation

## Description
Develop a complete testing strategy covering unit tests, integration tests, and API tests with comprehensive coverage of ML pipelines, API endpoints, and critical business logic.

## Acceptance Criteria
- [ ] Achieve ≥70% test coverage across the codebase
- [ ] Create unit tests for ML preprocessing and model training
- [ ] Add comprehensive API endpoint testing
- [ ] Implement authentication and authorization testing
- [ ] Test data validation and error handling
- [ ] Add performance tests for prediction endpoints
- [ ] Create mock data and fixtures for testing
- [ ] Implement test database setup and teardown
- [ ] Add continuous integration testing workflow
- [ ] Generate test coverage reports and documentation

## Implementation Notes
- Use pytest as the primary testing framework
- Create separate test databases and test data
- Mock external dependencies (MLFlow, file system)
- Include both positive and negative test cases
- Add parameterized tests for different data scenarios

## Files to Create/Modify
- `tests/unit/` (new directory structure)
- `tests/api/` (new directory for API tests)
- `tests/ml/` (new directory for ML tests)
- `pytest.ini` (new - pytest configuration)
- `.github/workflows/tests.yml` (new - CI workflow)

## Definition of Done
- Test coverage meets or exceeds 70% threshold
- All critical paths are covered by tests
- Tests run reliably in CI/CD environment
- Test documentation provides clear guidance
- Performance tests validate response times


---

### Issue #14: Implement performance optimization and load testing for prediction endpoints

**Labels:** phase-4, testing, infrastructure, medium-priority

**Milestone:** Phase 4: Docker, Testing, Validation

## Description
Optimize application performance, especially for ML prediction endpoints, and implement load testing to ensure the system can handle expected user loads with acceptable response times.

## Acceptance Criteria
- [ ] Profile prediction endpoint performance and identify bottlenecks
- [ ] Implement model loading optimization and caching strategies
- [ ] Add response time monitoring and logging
- [ ] Create load testing scenarios for key endpoints
- [ ] Optimize database queries and connection pooling
- [ ] Implement prediction result caching where appropriate
- [ ] Add performance metrics and monitoring dashboards
- [ ] Document performance requirements and SLAs
- [ ] Test system behavior under various load conditions

## Implementation Notes
- Use tools like locust or artillery for load testing
- Profile Python code to identify performance bottlenecks
- Consider async/await patterns for I/O operations
- Implement proper connection pooling for database
- Use caching (Redis) for frequently accessed data

## Files to Create/Modify
- `tests/load/` (new directory for load tests)
- `tests/load/locustfile.py` (new - load testing scenarios)
- `backend/fastapi_app/performance.py` (new - performance monitoring)
- `docs/performance_requirements.md` (new)

## Definition of Done
- Prediction endpoints respond within acceptable time limits
- System handles expected concurrent user loads
- Performance monitoring is in place
- Load testing scenarios cover realistic usage patterns
- Performance requirements are documented and met


---

### Issue #15: Create MLFlow model comparison analysis for thesis documentation

**Labels:** phase-5, documentation, ml, high-priority

**Milestone:** Phase 5: Thesis + Demo

## Description
Generate comprehensive model comparison analysis using MLFlow data, create visualizations and documentation comparing Random Forest vs MLP performance for inclusion in thesis materials.

## Acceptance Criteria
- [ ] Extract model performance metrics from MLFlow experiments
- [ ] Create comparative analysis of Random Forest vs MLP models
- [ ] Generate performance visualization charts and graphs
- [ ] Document model strengths, weaknesses, and use cases
- [ ] Include feature importance analysis and explanations
- [ ] Create MLFlow experiment screenshots for thesis
- [ ] Write model selection rationale and recommendations
- [ ] Document hyperparameter tuning results and insights
- [ ] Add statistical significance testing for performance differences

## Implementation Notes
- Use MLFlow API to extract experiment data
- Create visualizations with matplotlib/plotly
- Include metrics like accuracy, precision, recall, F1-score
- Document computational complexity and inference times
- Consider business impact of different model choices

## Files to Create/Modify
- `docs/thesis/model_comparison.md` (new)
- `docs/thesis/figures/` (new directory for visualizations)
- `scripts/generate_thesis_analysis.py` (new)
- `docs/thesis/mlflow_screenshots/` (new)

## Definition of Done
- Comprehensive model comparison analysis is complete
- Visualizations clearly show performance differences
- MLFlow screenshots document experiment tracking
- Analysis supports thesis conclusions and recommendations


---

### Issue #16: Document ethics considerations and system limitations for thesis

**Labels:** phase-5, documentation, high-priority

**Milestone:** Phase 5: Thesis + Demo

## Description
Create comprehensive documentation covering ethical considerations, system limitations, data privacy, and responsible AI practices for the anti-aging ML application thesis.

## Acceptance Criteria
- [ ] Document ethical considerations for health-related AI predictions
- [ ] Detail data privacy and security measures implemented
- [ ] Explain limitations of synthetic data and model predictions
- [ ] Address bias considerations in ML models and data
- [ ] Document responsible AI practices and transparency measures
- [ ] Include disclaimers about medical advice and limitations
- [ ] Explain model interpretability and explanation methods
- [ ] Detail data handling and user consent practices
- [ ] Address potential misuse and mitigation strategies

## Implementation Notes
- Follow established AI ethics frameworks
- Include specific examples and use cases
- Reference relevant academic literature
- Consider regulatory and legal implications
- Address both technical and social aspects

## Files to Create/Modify
- `docs/thesis/ethics_and_limitations.md` (new)
- `docs/thesis/responsible_ai.md` (new)
- `docs/privacy_policy.md` (new)
- `docs/disclaimer.md` (new)

## Definition of Done
- Ethics documentation covers all relevant considerations
- Limitations are clearly articulated and justified
- Privacy and security measures are documented
- Responsible AI practices are explained
- Documentation supports thesis defense


---

### Issue #17: Prepare demo video and thesis presentation materials

**Labels:** phase-5, documentation, medium-priority

**Milestone:** Phase 5: Thesis + Demo

## Description
Create comprehensive demo materials including video demonstration, presentation slides, and supporting documentation for thesis defense and project showcase.

## Acceptance Criteria
- [ ] Record comprehensive demo video showing end-to-end functionality
- [ ] Create presentation slides covering project overview, methodology, and results
- [ ] Prepare live demo script and backup plans
- [ ] Document user scenarios and use cases for demonstration
- [ ] Create technical architecture diagrams and system overview
- [ ] Prepare Q&A materials for potential thesis defense questions
- [ ] Test demo environment and ensure reliability
- [ ] Create project showcase materials for portfolio
- [ ] Document future work and potential improvements

## Implementation Notes
- Use screen recording software for demo video
- Create engaging and professional presentation materials
- Include actual system screenshots and results
- Prepare for both technical and non-technical audiences
- Have backup plans for live demo technical issues

## Files to Create/Modify
- `docs/demo/` (new directory)
- `docs/demo/demo_script.md` (new)
- `docs/demo/presentation_slides.pptx` (new)
- `docs/demo/demo_video.mp4` (new - or link to video)
- `docs/thesis/architecture_diagrams/` (new)

## Definition of Done
- Demo video clearly shows all system capabilities
- Presentation materials are professional and comprehensive
- Live demo is tested and reliable
- All materials support successful thesis defense
- Project is ready for showcase and portfolio inclusion


---

### Issue #18: Migrate persistence layer from Django ORM to SQLAlchemy models

**Labels:** backlog, backend, infrastructure, low-priority

## Description
Complete the transition from Django ORM to SQLAlchemy models in the FastAPI application, removing dependency on Django for data persistence and creating a unified data layer.

## Acceptance Criteria
- [ ] Create equivalent SQLAlchemy models for all Django models
- [ ] Implement database migration scripts for data transfer
- [ ] Update all FastAPI endpoints to use SQLAlchemy models
- [ ] Remove Django ORM dependencies from FastAPI code
- [ ] Update database initialization and seeding scripts
- [ ] Ensure data integrity during migration process
- [ ] Update tests to work with SQLAlchemy models
- [ ] Document new data layer architecture

## Implementation Notes
- Use Alembic for database migrations
- Ensure backward compatibility during transition
- Test migration process thoroughly
- Consider maintaining Django models temporarily for gradual migration

## Files to Create/Modify
- `backend/fastapi_app/models.py` (new - SQLAlchemy models)
- `backend/fastapi_app/database.py` (database configuration)
- `migration_scripts/` (new directory)
- Update all FastAPI endpoints and dependencies

## Definition of Done
- FastAPI application uses only SQLAlchemy for data persistence
- Migration scripts successfully transfer data
- All tests pass with new data layer
- Django dependency can be safely removed


---

### Issue #19: Implement advanced CSV schema validation for genetic data uploads

**Labels:** backlog, backend, medium-priority

## Description
Enhance the genetic data upload functionality with sophisticated CSV schema validation, including column validation, data type checking, and genetic marker format verification.

## Acceptance Criteria
- [ ] Define comprehensive CSV schema for genetic data
- [ ] Implement column name validation and standardization
- [ ] Add data type checking for each column
- [ ] Validate genetic marker formats and ranges
- [ ] Implement missing value detection and handling
- [ ] Add data quality scoring and reporting
- [ ] Create user-friendly validation error messages
- [ ] Support multiple CSV formats and dialects
- [ ] Add validation result visualization for users

## Implementation Notes
- Use libraries like pandas, cerberus, or pydantic for validation
- Create flexible schema that can handle format variations
- Provide clear feedback to users about validation issues
- Consider auto-correction for common formatting problems

## Files to Create/Modify
- `backend/fastapi_app/validation/csv_validator.py` (new)
- `backend/fastapi_app/schemas/genetic_data.py` (new)
- `backend/fastapi_app/validation/schemas/` (new directory)

## Definition of Done
- CSV validation catches all format and data issues
- Users receive clear feedback about validation problems
- System handles various CSV formats gracefully
- Validation performance is acceptable for large files


---

### Issue #20: Implement advanced model explanations and interpretability features

**Labels:** backlog, ml, frontend, low-priority

## Description
Enhance model interpretability by implementing advanced explanation methods beyond basic SHAP, including counterfactual explanations, feature interaction analysis, and personalized explanation narratives.

## Acceptance Criteria
- [ ] Implement SHAP kernel explanations for neural networks
- [ ] Add counterfactual explanation generation
- [ ] Create feature interaction analysis and visualization
- [ ] Implement personalized explanation narratives
- [ ] Add explanation confidence measures
- [ ] Create interactive explanation visualizations
- [ ] Implement explanation comparison between models
- [ ] Add explanation export and sharing functionality

## Implementation Notes
- Use libraries like SHAP, LIME, and DiCE for explanations
- Create user-friendly explanation interfaces
- Consider performance implications of explanation generation
- Ensure explanations are scientifically accurate

## Files to Create/Modify
- `backend/api/ml/explanations/` (new directory)
- `backend/api/ml/explanations/advanced_shap.py` (new)
- `backend/api/ml/explanations/counterfactual.py` (new)
- `streamlit_app/components/explanations.py` (new)

## Definition of Done
- Advanced explanations provide deeper insights than basic SHAP
- Explanation interfaces are user-friendly and informative
- Performance is acceptable for real-time use
- Explanations help users understand and trust predictions


---

### Issue #21: Scale synthetic dataset to 5000 records with improved validation

**Labels:** phase-1, ml, high-priority

**Milestone:** Phase 1: Setup + Data

## Description
Scale the current synthetic dataset from its current size to approximately 5000 records to ensure model viability during training. The dataset should maintain proper distributions and realistic demographic + habits data.

## Acceptance Criteria
- [ ] Generate synthetic dataset with N≈5000 records
- [ ] Validate data distributions are realistic and consistent
- [ ] Ensure demographic and habits data follows expected patterns
- [ ] Create both training dataset (train.csv) and small test datasets
- [ ] Place datasets under `backend/api/data/datasets/` directory
- [ ] Verify no data quality issues (missing values, outliers, inconsistencies)
- [ ] Document data generation process and validation rules

## Implementation Notes
- Use existing generator in `backend/api/data/generator.py` as starting point
- Keep current regression target (biological_age) for continuity
- Consider adding data validation checks and summary statistics
- Ensure reproducibility with random seeds

## Files to Modify
- `backend/api/data/generator.py`
- `backend/api/data/datasets/` (create directory if needed)

## Definition of Done
- Datasets are generated and placed in correct location
- Data quality validation passes
- Dataset size meets requirements (≈5000 records)
- Documentation is updated with data generation process


---

### Issue #22: Validate synthetic data distributions and implement quality checks

**Labels:** phase-1, ml, testing, medium-priority

**Milestone:** Phase 1: Setup + Data

## Description
Implement comprehensive validation for the synthetic dataset to ensure realistic distributions, proper correlations between features, and data quality standards that will support effective ML model training.

## Acceptance Criteria
- [ ] Create data validation pipeline/script
- [ ] Validate age distributions follow realistic patterns
- [ ] Verify genetic markers have appropriate frequencies
- [ ] Check lifestyle/habits data for logical consistency
- [ ] Implement automated data quality checks
- [ ] Generate data summary reports and visualizations
- [ ] Document validation rules and thresholds
- [ ] Create test suite for data validation

## Implementation Notes
- Build validation functions that can be reused
- Include statistical tests for distribution validation
- Check for correlation patterns between related variables
- Ensure no impossible combinations (e.g., age vs certain health metrics)

## Files to Create/Modify
- `backend/api/data/validation.py` (new)
- `backend/api/data/datasets/validation_report.md` (generated)
- Add validation tests to test suite

## Definition of Done
- Data validation pipeline is implemented and passing
- Validation report shows acceptable data quality
- Automated checks prevent bad data from being used
- Documentation covers validation process and standards


---

### Issue #23: Implement FastAPI authentication system with JWT and core user endpoints

**Labels:** phase-2, backend, high-priority

**Milestone:** Phase 2: Backend + ML

## Description
Implement secure authentication system for the FastAPI backend using JWT tokens, password hashing, and OAuth2 flow. Create core user management endpoints as foundation for the ML application.

## Acceptance Criteria
- [ ] Implement JWT token generation and validation using python-jose
- [ ] Add password hashing with passlib[bcrypt]
- [ ] Create POST /signup endpoint with user registration
- [ ] Create POST /token endpoint with OAuth2PasswordRequestForm
- [ ] Implement JWT authentication dependency for protected routes
- [ ] Add proper error handling and validation
- [ ] Configure CORS for frontend integration
- [ ] Add health check endpoint GET /health
- [ ] Create Pydantic schemas for request/response validation
- [ ] Add comprehensive API documentation in FastAPI Swagger

## Implementation Notes
- Use SQLAlchemy models for user persistence
- Follow FastAPI security best practices
- Ensure password validation and security requirements
- Add rate limiting considerations for production

## Files to Create/Modify
- `backend/fastapi_app/auth.py` (enhance existing)
- `backend/fastapi_app/schemas.py` (enhance existing)
- `backend/fastapi_app/main.py` (add endpoints)
- `backend/fastapi_app/db.py` (ensure user models)

## Definition of Done
- Authentication endpoints are working and tested
- JWT tokens are properly generated and validated
- Swagger documentation is complete and accurate
- Security best practices are implemented
- Integration tests pass for auth flow


---

### Issue #24: Create genetic data upload and habits submission endpoints with validation

**Labels:** phase-2, backend, high-priority

**Milestone:** Phase 2: Backend + ML

## Description
Implement secure endpoints for users to upload genetic data (CSV format) and submit lifestyle/habits information. Include proper validation, persistence, and user association.

## Acceptance Criteria
- [ ] Create POST /upload-genetic endpoint with CSV file upload
- [ ] Implement strict CSV schema validation for genetic data
- [ ] Create POST /submit-habits endpoint with JSON payload validation
- [ ] Add user authentication requirements for both endpoints
- [ ] Persist latest genetic profile per user (replace previous uploads)
- [ ] Store habits data with versioning/timestamps
- [ ] Add comprehensive input validation and error handling
- [ ] Implement file size and format restrictions for uploads
- [ ] Add progress tracking for large file uploads
- [ ] Create endpoints to retrieve user's current genetic profile and habits

## Implementation Notes
- Use Pydantic for JSON validation of habits data
- Implement CSV parsing with pandas/csv library
- Add database models for genetic profiles and habits
- Consider file storage strategy (database vs file system)
- Ensure proper cleanup of old files if applicable

## Files to Create/Modify
- `backend/fastapi_app/main.py` (add endpoints)
- `backend/fastapi_app/schemas.py` (add validation schemas)
- `backend/fastapi_app/db.py` (add database models)
- `backend/fastapi_app/validation.py` (new file for CSV validation)

## Definition of Done
- Upload endpoints accept and validate genetic data correctly
- Habits submission handles all required lifestyle factors
- Data is properly associated with authenticated users
- File validation prevents invalid or malicious uploads
- API documentation includes request/response examples


---

### Issue #25: Create unified ML preprocessing pipeline for training and inference

**Labels:** phase-2, ml, high-priority

**Milestone:** Phase 2: Backend + ML

## Description
Develop a consistent preprocessing pipeline that can be used for both model training and real-time prediction. Ensure feature engineering, scaling, and transformation steps are reproducible and aligned.

## Acceptance Criteria
- [ ] Create preprocessing pipeline class with fit/transform methods
- [ ] Implement feature engineering for genetic and habits data
- [ ] Add data scaling and normalization steps
- [ ] Handle missing values and outliers appropriately
- [ ] Ensure pipeline serialization for inference consistency
- [ ] Add preprocessing validation and sanity checks
- [ ] Create unit tests for all preprocessing steps
- [ ] Document feature engineering decisions and rationale
- [ ] Implement pipeline versioning for reproducibility

## Implementation Notes
- Use scikit-learn Pipeline for consistency
- Ensure preprocessing steps are identical between training and prediction
- Consider feature selection and dimensionality reduction
- Add logging for preprocessing steps and transformations

## Files to Create/Modify
- `backend/api/ml/preprocessor.py` (enhance existing)
- `backend/fastapi_app/ml/preprocessor.py` (create aligned version)
- `backend/api/ml/features.py` (new - feature engineering)
- Add preprocessing tests

## Definition of Done
- Preprocessing pipeline works consistently for training and inference
- All feature engineering steps are documented and tested
- Pipeline can be serialized and loaded for production use
- Validation ensures data quality before model training/prediction


---

### Issue #26: Train Random Forest baseline model with ONNX export and SHAP explanations

**Labels:** phase-2, ml, high-priority

**Milestone:** Phase 2: Backend + ML

## Description
Implement Random Forest model training pipeline with MLFlow tracking, ONNX export for efficient inference, and SHAP explanations for model interpretability.

## Acceptance Criteria
- [ ] Implement Random Forest training with hyperparameter optimization
- [ ] Log training metrics, parameters, and artifacts to MLFlow
- [ ] Export trained model to ONNX format for inference
- [ ] Implement SHAP explanations for model predictions
- [ ] Add model evaluation metrics (R²/RMSE for regression or F1/accuracy for classification)
- [ ] Create model validation pipeline with cross-validation
- [ ] Implement feature importance analysis
- [ ] Add model performance visualization and reports
- [ ] Ensure reproducibility with random seeds and versioning

## Implementation Notes
- Use scikit-learn RandomForestRegressor/Classifier
- Implement grid search or random search for hyperparameter tuning
- Use skl2onnx for ONNX conversion
- SHAP TreeExplainer for fast explanations on tree models
- Consider ensemble methods and model validation strategies

## Files to Create/Modify
- `backend/api/ml/train.py` (enhance existing)
- `backend/api/ml/models/random_forest.py` (new)
- `backend/api/ml/explain.py` (new - SHAP explanations)
- `backend/api/ml/evaluate.py` (new - model evaluation)

## Definition of Done
- Random Forest model trains successfully with good performance
- MLFlow tracking captures all relevant metrics and artifacts
- ONNX export works and produces consistent predictions
- SHAP explanations provide meaningful feature attributions
- Model evaluation meets performance thresholds


---

### Issue #27: Create MLP neural network model with PyTorch and MLFlow tracking

**Labels:** phase-2, ml, medium-priority

**Milestone:** Phase 2: Backend + ML

## Description
Develop a Multi-Layer Perceptron (MLP) using PyTorch as an alternative model to the Random Forest. Include proper training pipeline, MLFlow integration, and model comparison capabilities.

## Acceptance Criteria
- [ ] Design MLP architecture appropriate for the problem size
- [ ] Implement PyTorch training loop with proper optimization
- [ ] Add MLFlow tracking for neural network experiments
- [ ] Implement early stopping and learning rate scheduling
- [ ] Add model evaluation and comparison with Random Forest
- [ ] Export model for inference (TorchScript or ONNX)
- [ ] Implement neural network explanations (SHAP Kernel or other methods)
- [ ] Add visualization of training progress and model performance
- [ ] Create hyperparameter tuning pipeline for MLP

## Implementation Notes
- Keep architecture simple but effective (2-3 hidden layers)
- Use appropriate activation functions and regularization
- Implement proper train/validation/test splits
- Consider batch normalization and dropout for regularization
- Use appropriate loss function for the target variable type

## Files to Create/Modify
- `backend/api/ml/models/mlp.py` (new)
- `backend/api/ml/train_mlp.py` (new)
- `backend/api/ml/torch_utils.py` (new - PyTorch utilities)
- Update `backend/api/ml/explain.py` for neural network explanations

## Definition of Done
- MLP model trains successfully and converges
- MLFlow captures neural network metrics and artifacts
- Model performance is comparable to Random Forest baseline
- Model can be exported and loaded for inference
- Neural network explanations are implemented and functional


---

### Issue #28: Implement prediction endpoint with model selection and explanations

**Labels:** phase-2, backend, ml, high-priority

**Milestone:** Phase 2: Backend + ML

## Description
Create the core prediction endpoint that loads trained models, processes user data, and returns predictions with explanations. Support both Random Forest and MLP models with dynamic selection.

## Acceptance Criteria
- [ ] Create GET /predict endpoint with model_type parameter (rf|nn)
- [ ] Load latest trained models from MLFlow or local storage
- [ ] Fetch user's latest genetic profile and habits data
- [ ] Apply consistent preprocessing pipeline
- [ ] Return prediction with confidence intervals/probabilities
- [ ] Include SHAP explanations for prediction interpretability
- [ ] Add prediction caching for performance optimization
- [ ] Implement proper error handling for missing data or failed predictions
- [ ] Add prediction logging for monitoring and improvement

## Implementation Notes
- Use the same preprocessing pipeline as training
- Implement model loading and caching strategy
- Ensure predictions are consistent between ONNX and original models
- Add validation for input data completeness
- Consider prediction versioning for reproducibility

## Files to Create/Modify
- `backend/fastapi_app/main.py` (add prediction endpoint)
- `backend/fastapi_app/ml/predict.py` (enhance existing)
- `backend/fastapi_app/ml/model_loader.py` (new)
- `backend/fastapi_app/schemas.py` (add prediction response schemas)

## Definition of Done
- Prediction endpoint works for both model types
- Predictions include explanations and confidence measures
- Error handling covers all edge cases
- Performance is acceptable for real-time use
- API documentation includes prediction examples


---

### Issue #29: Develop Streamlit MVP with complete FastAPI integration

**Labels:** phase-3, frontend, high-priority

**Milestone:** Phase 3: Frontend + Integration

## Description
Create a functional Streamlit application that provides a complete user interface for the anti-aging ML app, integrating with all FastAPI endpoints for authentication, data upload, habits submission, and predictions.

## Acceptance Criteria
- [ ] Implement user registration and login interface
- [ ] Create genetic data upload interface with drag-and-drop CSV support
- [ ] Build habits/lifestyle questionnaire form
- [ ] Add prediction interface with model selection (RF vs MLP)
- [ ] Display prediction results with visualizations and explanations
- [ ] Implement session management for user authentication
- [ ] Add data validation and error handling in the UI
- [ ] Create dashboard for user's historical predictions
- [ ] Add loading states and progress indicators
- [ ] Ensure responsive design and good UX

## Implementation Notes
- Use Streamlit's session state for user management
- Implement proper error handling and user feedback
- Add data visualization for SHAP explanations
- Consider using Streamlit components for enhanced UI elements
- Ensure secure communication with FastAPI backend

## Files to Create/Modify
- `streamlit_app/app.py` (enhance existing)
- `streamlit_app/pages/` (create page modules)
- `streamlit_app/utils/api_client.py` (new - FastAPI client)
- `streamlit_app/components/` (new - reusable UI components)

## Definition of Done
- Complete user workflow from registration to prediction works
- All FastAPI endpoints are integrated and functional
- UI provides good user experience with proper feedback
- Error handling prevents application crashes
- Application is ready for thesis defense demo


---

### Issue #30: Document React/Next.js migration strategy and stabilize API contracts

**Labels:** phase-3, frontend, documentation, medium-priority

**Milestone:** Phase 3: Frontend + Integration

## Description
Prepare for post-defense migration from Streamlit to a production-ready React/Next.js frontend by documenting the migration strategy, stabilizing API contracts, and creating reusable components plan.

## Acceptance Criteria
- [ ] Document complete API contract specifications with OpenAPI/Swagger
- [ ] Create migration roadmap from Streamlit to React/Next.js
- [ ] Design component architecture for React frontend
- [ ] Identify reusable UI patterns and state management needs
- [ ] Plan authentication flow for React application
- [ ] Document data flow and API integration patterns
- [ ] Create wireframes/mockups for key user interfaces
- [ ] Establish frontend development standards and conventions
- [ ] Plan testing strategy for React components

## Implementation Notes
- Ensure API contracts are stable and well-documented
- Consider using TypeScript for type safety in React app
- Plan for modern React patterns (hooks, context, etc.)
- Consider state management solutions (Redux, Zustand, etc.)
- Plan for responsive design and accessibility

## Files to Create/Modify
- `docs/api_specification.md` (new)
- `docs/react_migration_plan.md` (new)
- `frontend/README.md` (update with migration plan)
- `docs/frontend_architecture.md` (new)

## Definition of Done
- API contracts are fully documented and stable
- Migration plan is detailed and actionable
- React component architecture is planned
- Documentation provides clear guidance for frontend development


---

### Issue #31: Implement end-to-end integration testing for complete user workflows

**Labels:** phase-3, testing, medium-priority

**Milestone:** Phase 3: Frontend + Integration

## Description
Create comprehensive integration tests that validate the complete user journey from registration through prediction, ensuring all components work together correctly in the Docker environment.

## Acceptance Criteria
- [ ] Set up integration testing framework and environment
- [ ] Create test scenarios for complete user workflows
- [ ] Test user registration, login, and session management
- [ ] Validate genetic data upload and processing flow
- [ ] Test habits submission and data persistence
- [ ] Verify prediction generation with both model types
- [ ] Test explanation generation and display
- [ ] Add performance testing for prediction endpoints
- [ ] Implement automated test data generation
- [ ] Create test reporting and monitoring

## Implementation Notes
- Use pytest for Python testing framework
- Consider using Playwright or Selenium for UI testing
- Test against actual Docker containers
- Include negative test cases and error scenarios
- Add load testing for prediction endpoints

## Files to Create/Modify
- `tests/integration/` (new directory)
- `tests/integration/test_user_workflows.py` (new)
- `tests/integration/test_api_integration.py` (new)
- `tests/integration/conftest.py` (new - test configuration)

## Definition of Done
- All major user workflows are covered by integration tests
- Tests run reliably in CI/CD environment
- Performance requirements are validated
- Test coverage includes error scenarios and edge cases


---

### Issue #32: Finalize Docker infrastructure with health checks and service optimization

**Labels:** phase-4, infrastructure, high-priority

**Milestone:** Phase 4: Docker, Testing, Validation

## Description
Complete the Docker Compose infrastructure by adding proper health checks, optimizing service configurations, and ensuring reliable container orchestration for all services including MLFlow.

## Acceptance Criteria
- [ ] Add comprehensive health checks for all services
- [ ] Optimize Docker Compose service dependencies and startup order
- [ ] Configure NGINX routing to properly proxy FastAPI endpoints
- [ ] Ensure MLFlow service integration and persistence
- [ ] Add environment variable configuration for all services
- [ ] Implement proper logging and monitoring for containers
- [ ] Add development vs production environment configurations
- [ ] Create service restart policies and failure handling
- [ ] Document Docker setup and troubleshooting guide

## Implementation Notes
- Use Docker Compose depends_on with health conditions
- Configure proper network communication between services
- Ensure persistent volumes for database and MLFlow data
- Add resource limits and constraints for containers
- Consider multi-stage builds for optimization

## Files to Create/Modify
- `antiaging-mvp/docker-compose.yml` (update existing)
- `antiaging-mvp/docker-compose.prod.yml` (new - production config)
- `antiaging-mvp/.env.example` (new - environment template)
- `docs/docker_setup.md` (new - Docker documentation)

## Definition of Done
- All services start reliably with proper dependencies
- Health checks accurately reflect service status
- NGINX properly routes requests to FastAPI
- MLFlow integration works seamlessly
- Documentation covers setup and troubleshooting


---

### Issue #33: Create comprehensive testing suite with ≥70% coverage

**Labels:** phase-4, testing, high-priority

**Milestone:** Phase 4: Docker, Testing, Validation

## Description
Develop a complete testing strategy covering unit tests, integration tests, and API tests with comprehensive coverage of ML pipelines, API endpoints, and critical business logic.

## Acceptance Criteria
- [ ] Achieve ≥70% test coverage across the codebase
- [ ] Create unit tests for ML preprocessing and model training
- [ ] Add comprehensive API endpoint testing
- [ ] Implement authentication and authorization testing
- [ ] Test data validation and error handling
- [ ] Add performance tests for prediction endpoints
- [ ] Create mock data and fixtures for testing
- [ ] Implement test database setup and teardown
- [ ] Add continuous integration testing workflow
- [ ] Generate test coverage reports and documentation

## Implementation Notes
- Use pytest as the primary testing framework
- Create separate test databases and test data
- Mock external dependencies (MLFlow, file system)
- Include both positive and negative test cases
- Add parameterized tests for different data scenarios

## Files to Create/Modify
- `tests/unit/` (new directory structure)
- `tests/api/` (new directory for API tests)
- `tests/ml/` (new directory for ML tests)
- `pytest.ini` (new - pytest configuration)
- `.github/workflows/tests.yml` (new - CI workflow)

## Definition of Done
- Test coverage meets or exceeds 70% threshold
- All critical paths are covered by tests
- Tests run reliably in CI/CD environment
- Test documentation provides clear guidance
- Performance tests validate response times


---

### Issue #34: Implement performance optimization and load testing for prediction endpoints

**Labels:** phase-4, testing, infrastructure, medium-priority

**Milestone:** Phase 4: Docker, Testing, Validation

## Description
Optimize application performance, especially for ML prediction endpoints, and implement load testing to ensure the system can handle expected user loads with acceptable response times.

## Acceptance Criteria
- [ ] Profile prediction endpoint performance and identify bottlenecks
- [ ] Implement model loading optimization and caching strategies
- [ ] Add response time monitoring and logging
- [ ] Create load testing scenarios for key endpoints
- [ ] Optimize database queries and connection pooling
- [ ] Implement prediction result caching where appropriate
- [ ] Add performance metrics and monitoring dashboards
- [ ] Document performance requirements and SLAs
- [ ] Test system behavior under various load conditions

## Implementation Notes
- Use tools like locust or artillery for load testing
- Profile Python code to identify performance bottlenecks
- Consider async/await patterns for I/O operations
- Implement proper connection pooling for database
- Use caching (Redis) for frequently accessed data

## Files to Create/Modify
- `tests/load/` (new directory for load tests)
- `tests/load/locustfile.py` (new - load testing scenarios)
- `backend/fastapi_app/performance.py` (new - performance monitoring)
- `docs/performance_requirements.md` (new)

## Definition of Done
- Prediction endpoints respond within acceptable time limits
- System handles expected concurrent user loads
- Performance monitoring is in place
- Load testing scenarios cover realistic usage patterns
- Performance requirements are documented and met


---

### Issue #35: Create MLFlow model comparison analysis for thesis documentation

**Labels:** phase-5, documentation, ml, high-priority

**Milestone:** Phase 5: Thesis + Demo

## Description
Generate comprehensive model comparison analysis using MLFlow data, create visualizations and documentation comparing Random Forest vs MLP performance for inclusion in thesis materials.

## Acceptance Criteria
- [ ] Extract model performance metrics from MLFlow experiments
- [ ] Create comparative analysis of Random Forest vs MLP models
- [ ] Generate performance visualization charts and graphs
- [ ] Document model strengths, weaknesses, and use cases
- [ ] Include feature importance analysis and explanations
- [ ] Create MLFlow experiment screenshots for thesis
- [ ] Write model selection rationale and recommendations
- [ ] Document hyperparameter tuning results and insights
- [ ] Add statistical significance testing for performance differences

## Implementation Notes
- Use MLFlow API to extract experiment data
- Create visualizations with matplotlib/plotly
- Include metrics like accuracy, precision, recall, F1-score
- Document computational complexity and inference times
- Consider business impact of different model choices

## Files to Create/Modify
- `docs/thesis/model_comparison.md` (new)
- `docs/thesis/figures/` (new directory for visualizations)
- `scripts/generate_thesis_analysis.py` (new)
- `docs/thesis/mlflow_screenshots/` (new)

## Definition of Done
- Comprehensive model comparison analysis is complete
- Visualizations clearly show performance differences
- MLFlow screenshots document experiment tracking
- Analysis supports thesis conclusions and recommendations


---

### Issue #36: Document ethics considerations and system limitations for thesis

**Labels:** phase-5, documentation, high-priority

**Milestone:** Phase 5: Thesis + Demo

## Description
Create comprehensive documentation covering ethical considerations, system limitations, data privacy, and responsible AI practices for the anti-aging ML application thesis.

## Acceptance Criteria
- [ ] Document ethical considerations for health-related AI predictions
- [ ] Detail data privacy and security measures implemented
- [ ] Explain limitations of synthetic data and model predictions
- [ ] Address bias considerations in ML models and data
- [ ] Document responsible AI practices and transparency measures
- [ ] Include disclaimers about medical advice and limitations
- [ ] Explain model interpretability and explanation methods
- [ ] Detail data handling and user consent practices
- [ ] Address potential misuse and mitigation strategies

## Implementation Notes
- Follow established AI ethics frameworks
- Include specific examples and use cases
- Reference relevant academic literature
- Consider regulatory and legal implications
- Address both technical and social aspects

## Files to Create/Modify
- `docs/thesis/ethics_and_limitations.md` (new)
- `docs/thesis/responsible_ai.md` (new)
- `docs/privacy_policy.md` (new)
- `docs/disclaimer.md` (new)

## Definition of Done
- Ethics documentation covers all relevant considerations
- Limitations are clearly articulated and justified
- Privacy and security measures are documented
- Responsible AI practices are explained
- Documentation supports thesis defense


---

### Issue #37: Prepare demo video and thesis presentation materials

**Labels:** phase-5, documentation, medium-priority

**Milestone:** Phase 5: Thesis + Demo

## Description
Create comprehensive demo materials including video demonstration, presentation slides, and supporting documentation for thesis defense and project showcase.

## Acceptance Criteria
- [ ] Record comprehensive demo video showing end-to-end functionality
- [ ] Create presentation slides covering project overview, methodology, and results
- [ ] Prepare live demo script and backup plans
- [ ] Document user scenarios and use cases for demonstration
- [ ] Create technical architecture diagrams and system overview
- [ ] Prepare Q&A materials for potential thesis defense questions
- [ ] Test demo environment and ensure reliability
- [ ] Create project showcase materials for portfolio
- [ ] Document future work and potential improvements

## Implementation Notes
- Use screen recording software for demo video
- Create engaging and professional presentation materials
- Include actual system screenshots and results
- Prepare for both technical and non-technical audiences
- Have backup plans for live demo technical issues

## Files to Create/Modify
- `docs/demo/` (new directory)
- `docs/demo/demo_script.md` (new)
- `docs/demo/presentation_slides.pptx` (new)
- `docs/demo/demo_video.mp4` (new - or link to video)
- `docs/thesis/architecture_diagrams/` (new)

## Definition of Done
- Demo video clearly shows all system capabilities
- Presentation materials are professional and comprehensive
- Live demo is tested and reliable
- All materials support successful thesis defense
- Project is ready for showcase and portfolio inclusion


---

### Issue #38: Migrate persistence layer from Django ORM to SQLAlchemy models

**Labels:** backlog, backend, infrastructure, low-priority

## Description
Complete the transition from Django ORM to SQLAlchemy models in the FastAPI application, removing dependency on Django for data persistence and creating a unified data layer.

## Acceptance Criteria
- [ ] Create equivalent SQLAlchemy models for all Django models
- [ ] Implement database migration scripts for data transfer
- [ ] Update all FastAPI endpoints to use SQLAlchemy models
- [ ] Remove Django ORM dependencies from FastAPI code
- [ ] Update database initialization and seeding scripts
- [ ] Ensure data integrity during migration process
- [ ] Update tests to work with SQLAlchemy models
- [ ] Document new data layer architecture

## Implementation Notes
- Use Alembic for database migrations
- Ensure backward compatibility during transition
- Test migration process thoroughly
- Consider maintaining Django models temporarily for gradual migration

## Files to Create/Modify
- `backend/fastapi_app/models.py` (new - SQLAlchemy models)
- `backend/fastapi_app/database.py` (database configuration)
- `migration_scripts/` (new directory)
- Update all FastAPI endpoints and dependencies

## Definition of Done
- FastAPI application uses only SQLAlchemy for data persistence
- Migration scripts successfully transfer data
- All tests pass with new data layer
- Django dependency can be safely removed


---

### Issue #39: Implement advanced CSV schema validation for genetic data uploads

**Labels:** backlog, backend, medium-priority

## Description
Enhance the genetic data upload functionality with sophisticated CSV schema validation, including column validation, data type checking, and genetic marker format verification.

## Acceptance Criteria
- [ ] Define comprehensive CSV schema for genetic data
- [ ] Implement column name validation and standardization
- [ ] Add data type checking for each column
- [ ] Validate genetic marker formats and ranges
- [ ] Implement missing value detection and handling
- [ ] Add data quality scoring and reporting
- [ ] Create user-friendly validation error messages
- [ ] Support multiple CSV formats and dialects
- [ ] Add validation result visualization for users

## Implementation Notes
- Use libraries like pandas, cerberus, or pydantic for validation
- Create flexible schema that can handle format variations
- Provide clear feedback to users about validation issues
- Consider auto-correction for common formatting problems

## Files to Create/Modify
- `backend/fastapi_app/validation/csv_validator.py` (new)
- `backend/fastapi_app/schemas/genetic_data.py` (new)
- `backend/fastapi_app/validation/schemas/` (new directory)

## Definition of Done
- CSV validation catches all format and data issues
- Users receive clear feedback about validation problems
- System handles various CSV formats gracefully
- Validation performance is acceptable for large files


---

### Issue #40: Implement advanced model explanations and interpretability features

**Labels:** backlog, ml, frontend, low-priority

## Description
Enhance model interpretability by implementing advanced explanation methods beyond basic SHAP, including counterfactual explanations, feature interaction analysis, and personalized explanation narratives.

## Acceptance Criteria
- [ ] Implement SHAP kernel explanations for neural networks
- [ ] Add counterfactual explanation generation
- [ ] Create feature interaction analysis and visualization
- [ ] Implement personalized explanation narratives
- [ ] Add explanation confidence measures
- [ ] Create interactive explanation visualizations
- [ ] Implement explanation comparison between models
- [ ] Add explanation export and sharing functionality

## Implementation Notes
- Use libraries like SHAP, LIME, and DiCE for explanations
- Create user-friendly explanation interfaces
- Consider performance implications of explanation generation
- Ensure explanations are scientifically accurate

## Files to Create/Modify
- `backend/api/ml/explanations/` (new directory)
- `backend/api/ml/explanations/advanced_shap.py` (new)
- `backend/api/ml/explanations/counterfactual.py` (new)
- `streamlit_app/components/explanations.py` (new)

## Definition of Done
- Advanced explanations provide deeper insights than basic SHAP
- Explanation interfaces are user-friendly and informative
- Performance is acceptable for real-time use
- Explanations help users understand and trust predictions


---

