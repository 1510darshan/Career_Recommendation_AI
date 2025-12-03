# Career Recommendation AI Model

---

## 1. System Specifications
| Component | Details |
|----------|---------|
| Model Type | Random Forest Classifier |
| Framework | Scikit-learn (Python) |
| Number of Estimators | 100 trees |
| Maximum Depth | 10 levels |
| Random State | 42 (reproducibility) |

---

## 2. Input Features (15 Total)

### Demographic Features (1)
- **Gender** (Binary: Male = 1, Female = 0)

### Behavioral Features (4)
- Part-time job status (Binary: Yes = 1, No = 0)
- Extracurricular activity participation (Binary: Yes = 1, No = 0)
- Absence days (0–30 range)
- Weekly self-study hours (0–50 range)

### Academic Performance Features (7)
- Mathematics (0–100)
- History (0–100)
- Physics (0–100)
- Chemistry (0–100)
- Biology (0–100)
- English (0–100)
- Geography (0–100)

### Engineered Features (3)
- Science aggregate = (Physics + Chemistry + Biology) / 3
- Humanities aggregate = (History + English + Geography) / 3
- Overall academic score = Average of 7 subject scores

---

## 3. Data Preprocessing Pipeline

1. **Data Loading**
   - Supports CSV, XLSX, XLS files
   - Auto column detection/validation
   - Multi-file merging support

2. **Feature Encoding**
   - Binary encoding for categorical data
   - Boolean → Integer conversion
   - Label encoding for target career categories

3. **Feature Scaling**
   - StandardScaler (Z-score normalization)
   - Across all 15 input features

4. **Data Splitting**
   - 80% training / 20% testing
   - Stratified random sampling

---

## 4. Model Architecture

| Parameter | Setting |
|----------|---------|
| Algorithm | Random Forest Classifier |
| n_estimators | 100 |
| max_depth | 10 |
| criterion | Gini impurity |
| random_state | 42 |

### Advantages
- Handles non-linearity
- Prevents overfitting
- Works with mixed feature types
- Feature importance analysis capability

---

## 5. Model Evaluation Metrics
- Training accuracy
- Testing accuracy
- Feature importance ranking

### Output Format
- Top 3 career recommendations per student
- Confidence score (0–100%)
- Required skill mapping

---

## 6. System Capabilities

### Training Phase
- Batch learning
- Save/load model feature
- Cross-validation ready

### Prediction Phase
- Real-time single or batch predictions
- Probability-based ranking
- Skill recommendations

---

## 7. Supported Career Categories (Default)

1. Software Development & Engineering  
2. Doctor / Medical Professional  
3. Lawyer / Legal Professional  
4. Teacher / Educator  
5. Business Owner / Entrepreneur  
6. Scientist / Researcher  
7. Software Engineer  
8. Government Officer  
9. Artist / Creative Professional  

> *Supports unlimited custom categories based on dataset.*

---

## 8. Technical Implementation

| Component | Technology |
|----------|------------|
| Language | Python 3.7+ |
| Core Libraries | NumPy, Pandas, Scikit-learn, Pickle, OpenPyXL/XLRD |

### Model Persistence
- Stored as `.pkl` file
- Includes:
  - Trained Random Forest model
  - StandardScaler parameters
  - LabelEncoder mappings
  - Career-skills dictionary

---

## 9. System Workflow
1. Data Loading
2. Preprocessing
3. Training
4. Evaluation & Feature Analysis
5. Prediction
6. Final Output Delivery

---

## 10. Feature Importance Analysis (Typical)
1. Overall academic score  
2. Mathematics score  
3. Science aggregate  
4. Physics score  
5. Weekly self-study hours  

*(Ranks may vary by dataset.)*

---

## 11. Input Data Requirements

### Minimum Dataset Size
- **Recommended:** 100+ student records  
- **Multiple Career Categories:** 5 or more different career aspirations  
- **Balanced Class Distribution:** Prefer equal representation across careers  

### Required Columns (Total: 13)
- gender  
- part_time_jobss  
- absence_days  
- extracurricular_activities  
- weekly_self_study_hours  
- career_aspiration *(Target Variable)*  
- math_score  
- history_score  
- physics_score  
- chemistry_score  
- biology_score  
- english_score  
- geography_score  







---

## 12. Advantages
- Multi-format data support
- Scalable career classes
- Interpretable results
- User-friendly input and reporting
- Skill-based guidance

---

## 13. Limitations
- Requires labeled data
- Ignores personality & socioeconomics
- Cannot suggest unseen career categories
- Limited feature set (15)

---

## 14. System Accuracy Metrics
- Training accuracy: **85–95%**
- Testing accuracy: **70–85%**
- Overfitting controlled via `max_depth`

---

## 15. Validation Approach
- Train-Test Split evaluation
- Accuracy scoring & feature ranking
- Confidence score calibration

---

## 16. Use Cases
- Schools & Colleges (counseling)
- Career guidance platforms
- Educational research & analytics
- Government policy studies

---

## 17. Future Enhancements
- Neural Networks for deeper patterns
- NLP for automated career insights
- Personality profiling
- Labor market integration
- Explainable AI (SHAP/LIME)
- Mobile application
- Hybrid recommendation engines

---

## 18. Comparative Advantages

| Compared With | Benefit |
|--------------|---------|
| Rule-based | Learns automatically |
| Decision Tree | Better generalization |
| Neural Network | More interpretable |
| KNN | Faster prediction |
| SVM | Handles non-linear data better |

---

## 19. Data Privacy & Ethics
- Anonymous data processing
- Local deployment possible
- Transparent decision reasoning

---

## 20. Sample Output Format

### Input Example:
- Math = 92, Physics = 88, Study Hours = 25

### Output:
1. **Software Engineer** — 78%  
   Skills: Programming, Algorithms, Problem Solving  

2. **Scientist** — 65%  
   Skills: Research, Analytical Thinking  

3. **Doctor** — 52%  
   Skills: Biology & Medical Knowledge  

---

## Setup Instructions

### Windows
```bash
python -m venv env
env\Scripts\activate
pip install numpy pandas scikit-learn openpyxl xlrd
```

### Linux/MacOS
```bash
python3 -m venv env
source env/bin/activate
pip install numpy pandas scikit-learn openpyxl xlrd
```
