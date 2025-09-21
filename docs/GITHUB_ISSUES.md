# GitHub Issues for Anti-Aging ML App Development Plan

This document contains actionable GitHub issues derived from the comprehensive development plan in `DEV_PLAN.md`. Each issue is designed to be specific, measurable, and achievable, with clear acceptance criteria and implementation guidance.

## Quick Reference - Linear Development Order

For systematic development, follow these issues in order:

### **🚨 CRITICAL PRIORITY - Address Before Original Roadmap**
1. **Issue #43:** 🔥 **URGENT** Synthetic Data Realism Overhaul - Fix Age Correlation (Critical) ⏳ **MUST FIX FIRST**
2. **Issue #44:** Genomics-Specific Data Quality and Preprocessing (High Priority) 
3. **Issue #45:** Realistic Model Performance Baselines and Benchmarking (High Priority)
4. **Issue #46:** Advanced Feature Engineering for Aging Biology (High Priority)
5. **Issue #47:** Statistical Rigor and Multiple Testing Correction (High Priority)
6. **Issue #48:** Repository Structure Cleanup and Code Organization (Medium Priority)

### Phase 1: Setup + Data (Revised)
7. **Issue #1:** Scale synthetic dataset (Phase 1) ✅ COMPLETE
8. **Issue #2:** Data validation pipeline (Phase 1) ✅ COMPLETE

### Phase 2: Backend + ML (After Critical Fixes)
9. **Issue #3:** FastAPI authentication (Phase 2) ✅ COMPLETE
10. **Issue #4:** Data upload endpoints (Phase 2) ✅ COMPLETE
11. **Issue #21:** Linear Regression baseline + MLFlow (Phase 2) ✅ COMPLETE
12. **Issue #42:** Multivariate statistical analysis for feature groupings (Phase 2) 🔄 CURRENT
13. **Issue #5:** ML preprocessing pipeline (Phase 2)
14. **Issue #6:** Random Forest training (Phase 2)
15. **Issue #7:** MLP neural network (Phase 2)
16. **Issue #8:** Prediction endpoint (Phase 2)

### Phase 3: Frontend + Integration
17. **Issue #9:** Streamlit MVP (Phase 3)
18. **Issue #10:** React migration planning (Phase 3)
19. **Issue #11:** End-to-end integration testing (Phase 3)

### Phase 4: Docker, Testing, Validation
20. **Issue #12:** Docker infrastructure (Phase 4)
21. **Issue #13:** Testing suite (Phase 4)
22. **Issue #14:** Performance optimization (Phase 4)

### Phase 5: Thesis + Demo
23. **Issue #15:** MLFlow analysis (Phase 5)
24. **Issue #16:** Ethics documentation (Phase 5)
25. **Issue #17:** Demo preparation (Phase 5)

### Backlog
26. **Issue #18:** Django to SQLAlchemy migration
27. **Issue #19:** Advanced CSV validation
28. **Issue #20:** Enhanced model explanations

### 🚨 **CRITICAL FINDINGS from Notebook Analysis**

Our comprehensive skeptical analysis revealed **fundamental issues** that invalidate current results:

1. **🔴 Unrealistic Age Correlation (0.945)**: Current synthetic data has implausibly high age-biological age correlation. Real aging research shows 0.6-0.8 maximum. This makes all model results artificially optimistic.

2. **🔴 Implausible Model Performance**: Current R² ~0.97 and MAE ~2 years is unprecedented for biological aging prediction. Real aging clocks achieve R² 0.6-0.8 with 4-8 year errors.

3. **🔴 Missing Genomics Best Practices**: Genetic data preprocessing ignores population genetics, Hardy-Weinberg equilibrium, linkage disequilibrium, and proper genetic models.

4. **🔴 Lack of Statistical Rigor**: Missing multiple testing correction, confidence intervals, and proper hypothesis testing essential for genomics research.

5. **🔴 Unrealistic Feature Engineering**: Ignores gene-environment interactions, aging pathways, and established aging biology patterns.

### **⚠️ Action Required Before Continuing**

**ALL DEVELOPMENT MUST PAUSE** until Issues #43-47 are addressed. Current results are scientifically implausible and would not be acceptable in peer review or thesis defense. The synthetic data generator needs major revision to create realistic biological complexity.

### **Expected Changes After Fixes**
- Model performance will drop to realistic levels (R² 0.6-0.8, MAE 4-8 years)
- Feature importance patterns will become more biologically plausible  
- Model comparisons will show meaningful differences
- Results will be defendable in academic/scientific contexts

### Pivot: Baseline Linear Regression First

To ensure a rigorous comparison across models, we've introduced Issue #21 to train a Linear Regression baseline immediately after Issue #4. All subsequent models (RF, MLP) must be logged to MLFlow under a shared experiment name to enable side-by-side comparison (metrics: RMSE, R², MAE; artifacts: model + preprocessor). This pivots the sequence but keeps Phase 2 scope intact.

### Addition: Multivariate Statistical Analysis

Issue #42 adds clustering/grouping analysis and canonical correlation analysis to discover colinear relationships between variables. The hypothesis is that feature groups (genetic, lifestyle, demographic, health markers, environmental) have different weights and impacts on model performance based on their relationships. This analysis will inform group-aware preprocessing and feature engineering for improved model performance.lementation guidance.

## Overview

**Total Issues:** 20 issues across 5 phases + backlog items  
**Timeline:** 6 weeks (September 1 - October 15, 2024)  
**Approach:** Each phase builds upon the previous, ensuring systematic progress toward MVP completion.

## Phase Structure

### Phase 1: Setup + Data (Sep 1-7)
**Goal:** Scale synthetic dataset and prepare data infrastructure  
**Issues:** 2 issues focused on data generation and validation

### Phase 2: Backend + ML (Sep 8-18)  
**Goal:** Implement FastAPI backend with ML models and MLFlow integration  
**Issues:** 6 issues covering authentication, data handling, ML training, and prediction endpoints

### Phase 3: Frontend + Integration (Sep 19-29)
**Goal:** Build Streamlit MVP and integrate with backend  
**Issues:** 3 issues for UI development and end-to-end integration

### Phase 4: Docker, Testing, Validation (Sep 30 - Oct 6)
**Goal:** Complete containerization and testing infrastructure  
**Issues:** 3 issues for infrastructure and comprehensive testing

### Phase 5: Thesis + Demo (Oct 7-15)
**Goal:** Finalize thesis materials and demo preparation  
**Issues:** 3 issues for documentation, analysis, and demo creation

### Backlog Items
**Goal:** Future enhancements and technical debt  
**Issues:** 3 issues for post-MVP improvements

## Labels System

| Label | Color | Description |
|-------|-------|-------------|
| `phase-1` | `0075ca` | Phase 1: Setup + Data |
| `phase-2` | `0075ca` | Phase 2: Backend + ML |
| `phase-3` | `0075ca` | Phase 3: Frontend + Integration |
| `phase-4` | `0075ca` | Phase 4: Docker, Testing, Validation |
| `phase-5` | `0075ca` | Phase 5: Thesis + Demo |
| `backend` | `d73a4a` | Backend development |
| `frontend` | `a2eeef` | Frontend development |
| `ml` | `0e8a16` | Machine Learning related |
| `infrastructure` | `f9d0c4` | Infrastructure and DevOps |
| `documentation` | `7057ff` | Documentation and thesis |
| `testing` | `fef2c0` | Testing and validation |
| `high-priority` | `d93f0b` | High priority task |
| `medium-priority` | `fbca04` | Medium priority task |
| `low-priority` | `0e8a16` | Low priority task |
| `backlog` | `c5def5` | Backlog item |

## Milestones

1. **Phase 1: Setup + Data** - Due: September 7, 2024
2. **Phase 2: Backend + ML** - Due: September 18, 2024  
3. **Phase 3: Frontend + Integration** - Due: September 29, 2024
4. **Phase 4: Docker, Testing, Validation** - Due: October 6, 2024
5. **Phase 5: Thesis + Demo** - Due: October 15, 2024

## Implementation Strategy

### Priority Order
1. **High Priority Issues:** Must be completed for MVP functionality
2. **Medium Priority Issues:** Important for robustness and user experience
3. **Low Priority Issues:** Nice-to-have features and optimizations
4. **Backlog Issues:** Future work and technical debt

### Dependencies
- Phase issues should generally be completed in order
- Some issues within phases can be worked on in parallel
- Backend authentication must be completed before frontend integration
- ML models must be trained before prediction endpoints
- Testing infrastructure should be developed alongside features

### Success Metrics
- All high and medium priority issues completed
- Core user workflow functional (register → upload → predict → explain)
- Both Random Forest and MLP models operational
- MLFlow tracking functional
- Streamlit MVP demonstrates full functionality
- Docker environment runs reliably
- Test coverage ≥70%
- Thesis materials complete and demo ready

## Issue Creation Instructions

To create these issues in GitHub:

1. **Create Labels:** Copy the labels table above to create all necessary labels
2. **Create Milestones:** Set up the 5 phase milestones with due dates
3. **Create Issues:** Use the detailed issue descriptions in the sections below
4. **Assign Priorities:** Use labels to indicate priority levels
5. **Set Dependencies:** Note issue dependencies in descriptions or comments

## Detailed Issues

### Phase 1 Issues

#### Issue #1: Scale synthetic dataset to 5000 records with improved validation
**Labels:** `phase-1`, `ml`, `high-priority`  
**Milestone:** Phase 1: Setup + Data

**Description:**
Scale the current synthetic dataset from its current size to approximately 5000 records to ensure model viability during training. The dataset should maintain proper distributions and realistic demographic + habits data.

**Acceptance Criteria:**
- [ ] Generate synthetic dataset with N≈5000 records
- [ ] Validate data distributions are realistic and consistent
- [ ] Ensure demographic and habits data follows expected patterns
- [ ] Create both training dataset (train.csv) and small test datasets
- [ ] Place datasets under `backend/api/data/datasets/` directory
- [ ] Verify no data quality issues (missing values, outliers, inconsistencies)
- [ ] Document data generation process and validation rules

**Implementation Notes:**
- Use existing generator in `backend/api/data/generator.py` as starting point
- Keep current regression target (biological_age) for continuity
- Consider adding data validation checks and summary statistics
- Ensure reproducibility with random seeds

**Files to Modify:**
- `backend/api/data/generator.py`
- `backend/api/data/datasets/` (create directory if needed)

**Definition of Done:**
- Datasets are generated and placed in correct location
- Data quality validation passes
- Dataset size meets requirements (≈5000 records)
- Documentation is updated with data generation process

---

#### Issue #2: Validate synthetic data distributions and implement quality checks
**Labels:** `phase-1`, `ml`, `testing`, `medium-priority`  
**Milestone:** Phase 1: Setup + Data

**Description:**
Implement comprehensive validation for the synthetic dataset to ensure realistic distributions, proper correlations between features, and data quality standards that will support effective ML model training.

**Acceptance Criteria:**
- [ ] Create data validation pipeline/script
- [ ] Validate age distributions follow realistic patterns
- [ ] Verify genetic markers have appropriate frequencies
- [ ] Check lifestyle/habits data for logical consistency
- [ ] Implement automated data quality checks
- [ ] Generate data summary reports and visualizations
- [ ] Document validation rules and thresholds
- [ ] Create test suite for data validation

**Implementation Notes:**
- Build validation functions that can be reused
- Include statistical tests for distribution validation
- Check for correlation patterns between related variables
- Ensure no impossible combinations (e.g., age vs certain health metrics)

**Files to Create/Modify:**
- `backend/api/data/validation.py` (new)
- `backend/api/data/datasets/validation_report.md` (generated)
- Add validation tests to test suite

**Definition of Done:**
- Data validation pipeline is implemented and passing
- Validation report shows acceptable data quality
- Automated checks prevent bad data from being used
- Documentation covers validation process and standards

---

### Phase 2 Issues

#### Issue #3: Implement FastAPI authentication system with JWT and core user endpoints
**Labels:** `phase-2`, `backend`, `high-priority`  
**Milestone:** Phase 2: Backend + ML

**Description:**
Implement secure authentication system for the FastAPI backend using JWT tokens, password hashing, and OAuth2 flow. Create core user management endpoints as foundation for the ML application.

**Acceptance Criteria:**
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

**Implementation Notes:**
- Use SQLAlchemy models for user persistence
- Follow FastAPI security best practices
- Ensure password validation and security requirements
- Add rate limiting considerations for production

**Files to Create/Modify:**
- `backend/fastapi_app/auth.py` (enhance existing)
- `backend/fastapi_app/schemas.py` (enhance existing)
- `backend/fastapi_app/main.py` (add endpoints)
- `backend/fastapi_app/db.py` (ensure user models)

**Definition of Done:**
- Authentication endpoints are working and tested
- JWT tokens are properly generated and validated
- Swagger documentation is complete and accurate
- Security best practices are implemented
- Integration tests pass for auth flow

---

#### Issue #4: Create genetic data upload and habits submission endpoints with validation
**Labels:** `phase-2`, `backend`, `high-priority`  
**Milestone:** Phase 2: Backend + ML

**Description:**
Implement secure endpoints for users to upload genetic data (CSV format) and submit lifestyle/habits information. Include proper validation, persistence, and user association.

**Acceptance Criteria:**
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

**Implementation Notes:**
- Use Pydantic for JSON validation of habits data
- Implement CSV parsing with pandas/csv library
- Add database models for genetic profiles and habits
- Consider file storage strategy (database vs file system)
- Ensure proper cleanup of old files if applicable

**Files to Create/Modify:**
- `backend/fastapi_app/main.py` (add endpoints)
- `backend/fastapi_app/schemas.py` (add validation schemas)
- `backend/fastapi_app/db.py` (add database models)
- `backend/fastapi_app/validation.py` (new file for CSV validation)

**Definition of Done:**
- Upload endpoints accept and validate genetic data correctly
- Habits submission handles all required lifestyle factors
- Data is properly associated with authenticated users
- File validation prevents invalid or malicious uploads
- API documentation includes request/response examples

---

*[Continue with remaining 15 issues following the same detailed format...]*

## Quick Reference

For immediate implementation, focus on these high-priority issues in order:

1. **Issue #1:** Scale synthetic dataset (Phase 1)
2. **Issue #3:** FastAPI authentication (Phase 2)
3. **Issue #4:** Data upload endpoints (Phase 2)
4. **Issue #21:** Linear Regression baseline + MLFlow (Phase 2, pivot)
5. **Issue #5:** ML preprocessing pipeline (Phase 2)
6. **Issue #6:** Random Forest training (Phase 2)
7. **Issue #8:** Prediction endpoint (Phase 2)
8. **Issue #9:** Streamlit MVP (Phase 3)
9. **Issue #12:** Docker infrastructure (Phase 4)
10. **Issue #13:** Testing suite (Phase 4)
11. **Issue #15:** MLFlow analysis (Phase 5)

### Pivot: Baseline Linear Regression First

To ensure a rigorous comparison across models, we’ve introduced Issue #21 to train a Linear Regression baseline immediately after Issue #4. All subsequent models (RF, MLP) must be logged to MLFlow under a shared experiment name to enable side-by-side comparison (metrics: RMSE, R², MAE; artifacts: model + preprocessor). This pivots the sequence but keeps Phase 2 scope intact.

## Next Steps

1. Create GitHub repository labels and milestones
2. Create issues #1-2 for Phase 1 to begin data preparation
3. Start Phase 1 implementation while preparing Phase 2 issues
4. Monitor progress against DEV_PLAN.md timeline
5. Adjust priorities based on implementation progress and findings

---

*This document serves as the actionable translation of DEV_PLAN.md into specific GitHub issues. Each issue is designed to be independently implementable while contributing to the overall MVP goals.*