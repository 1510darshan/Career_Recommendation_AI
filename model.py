"""
Enhanced Career Recommendation AI Model
Supports loading data from multiple CSV/Excel files:
- Student data file
- Career-Skills mapping file
- Optional: Skills matrix file
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import warnings
import os
warnings.filterwarnings('ignore')


class MultiFileDataLoader:
    """Handles loading and merging data from multiple CSV/Excel files"""
    
    def __init__(self):
        self.student_data = None
        self.career_skills_mapping = None
        self.skills_matrix = None
    
    @staticmethod
    def load_file(filepath):
        """Load a CSV or Excel file"""
        try:
            file_ext = os.path.splitext(filepath)[1].lower()
            
            if file_ext == '.csv':
                df = pd.read_csv(filepath)
                print(f"✓ Loaded CSV: {os.path.basename(filepath)} ({len(df)} rows, {len(df.columns)} columns)")
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(filepath)
                print(f"✓ Loaded Excel: {os.path.basename(filepath)} ({len(df)} rows, {len(df.columns)} columns)")
            else:
                print(f"✗ Unsupported format: {file_ext}")
                return None
            
            return df
        except FileNotFoundError:
            print(f"✗ File not found: {filepath}")
            return None
        except Exception as e:
            print(f"✗ Error loading file: {e}")
            return None
    
    def load_student_data(self, filepath):
        """Load student data from CSV/Excel"""
        print("\n--- Loading Student Data ---")
        self.student_data = self.load_file(filepath)
        
        if self.student_data is not None:
            # Convert boolean columns if they're strings
            bool_columns = ['part_time_job', 'extracurricular_activities']
            for col in bool_columns:
                if col in self.student_data.columns:
                    if self.student_data[col].dtype == 'object':
                        self.student_data[col] = self.student_data[col].map({
                            'TRUE': True, 'FALSE': False, 
                            'True': True, 'False': False,
                            'true': True, 'false': False, 
                            1: True, 0: False
                        })
            
            print("\nColumns found:")
            for col in self.student_data.columns:
                print(f"  - {col}")
            
            # Check required columns
            required_cols = ['gender', 'part_time_job', 'absence_days', 
                           'extracurricular_activities', 'weekly_self_study_hours', 
                           'career_aspiration', 'math_score', 'history_score',
                           'physics_score', 'chemistry_score', 'biology_score', 
                           'english_score', 'geography_score']
            
            missing = [col for col in required_cols if col not in self.student_data.columns]
            if missing:
                print(f"\n⚠ Warning: Missing columns: {', '.join(missing)}")
            else:
                print("\n✓ All required columns present!")
        
        return self.student_data
    
    def load_career_skills_mapping(self, filepath):
        """Load career to skills mapping from CSV/Excel"""
        print("\n--- Loading Career-Skills Mapping ---")
        df = self.load_file(filepath)
        
        if df is not None:
            # Expected columns: 'Career' and 'Skill' (or similar)
            print("\nColumns found:")
            for col in df.columns:
                print(f"  - {col}")
            
            # Try to identify career and skill columns
            career_col = None
            skill_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if 'career' in col_lower:
                    career_col = col
                elif 'skill' in col_lower:
                    skill_col = col
            
            if career_col and skill_col:
                print(f"\n✓ Identified: Career='{career_col}', Skills='{skill_col}'")
                
                # Convert to dictionary format
                self.career_skills_mapping = {}
                for _, row in df.iterrows():
                    career = row[career_col]
                    skills = row[skill_col]
                    
                    # Parse skills (assuming comma-separated)
                    if pd.notna(skills):
                        skill_list = [s.strip() for s in str(skills).split(',')]
                        self.career_skills_mapping[career] = skill_list
                
                print(f"\n✓ Loaded {len(self.career_skills_mapping)} career mappings")
                print("\nCareers found:")
                for career in list(self.career_skills_mapping.keys())[:5]:
                    print(f"  - {career}")
                if len(self.career_skills_mapping) > 5:
                    print(f"  ... and {len(self.career_skills_mapping) - 5} more")
            else:
                print("\n⚠ Warning: Could not identify Career and Skill columns")
                print("   Expected column names containing 'career' and 'skill'")
        
        return self.career_skills_mapping
    
    def load_skills_matrix(self, filepath):
        """Load skills matrix from CSV/Excel (optional)"""
        print("\n--- Loading Skills Matrix (Optional) ---")
        self.skills_matrix = self.load_file(filepath)
        
        if self.skills_matrix is not None:
            print(f"\n✓ Skills matrix loaded: {self.skills_matrix.shape}")
            print(f"   Columns: {len(self.skills_matrix.columns)}")
            print(f"   Rows: {len(self.skills_matrix)}")
        
        return self.skills_matrix
    
    def get_merged_data(self):
        """Get the loaded student data"""
        if self.student_data is None:
            print("✗ No student data loaded!")
            return None
        
        return self.student_data


class CareerRecommendationModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.career_skills = {}
    
    def set_career_skills(self, career_skills_dict):
        """Set the career to skills mapping"""
        self.career_skills = career_skills_dict
    
    def prepare_features(self, df):
        """Prepare features for training"""
        data = df.copy()
        
        # Encode categorical variables
        data['gender_encoded'] = (data['gender'] == 'male').astype(int)
        data['part_time_job_encoded'] = data['part_time_job'].astype(int)
        data['extracurricular_encoded'] = data['extracurricular_activities'].astype(int)
        
        # Calculate aggregate scores
        data['science_score'] = (data['physics_score'] + data['chemistry_score'] + 
                                 data['biology_score']) / 3
        data['humanities_score'] = (data['history_score'] + data['english_score'] + 
                                    data['geography_score']) / 3
        data['overall_score'] = (data['math_score'] + data['history_score'] + 
                                 data['physics_score'] + data['chemistry_score'] + 
                                 data['biology_score'] + data['english_score'] + 
                                 data['geography_score']) / 7
        
        # Feature columns
        self.feature_columns = [
            'gender_encoded', 'part_time_job_encoded', 'extracurricular_encoded',
            'absence_days', 'weekly_self_study_hours', 'math_score', 'history_score',
            'physics_score', 'chemistry_score', 'biology_score', 'english_score',
            'geography_score', 'science_score', 'humanities_score', 'overall_score'
        ]
        
        return data[self.feature_columns]
    
    def train(self, df):
        """Train the model"""
        print("\n" + "=" * 60)
        print("TRAINING MODEL")
        print("=" * 60)
        
        X = self.prepare_features(df)
        y = self.label_encoder.fit_transform(df['career_aspiration'])
        
        print(f"\nDataset: {len(df)} students, {len(self.label_encoder.classes_)} careers")
        print("\nCareer distribution:")
        for i, career in enumerate(self.label_encoder.classes_, 1):
            count = (df['career_aspiration'] == career).sum()
            print(f"  {i}. {career}: {count} students")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("\nTraining Random Forest...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_train_pred = self.model.predict(X_train_scaled)
        
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n{'='*60}")
        print(f"Training Accuracy: {train_accuracy:.2%}")
        print(f"Testing Accuracy: {test_accuracy:.2%}")
        print(f"{'='*60}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print("-" * 60)
        for idx, row in feature_importance.head(10).iterrows():
            bar = '█' * int(row['importance'] * 50)
            print(f"{row['feature']:30} {bar} {row['importance']:.4f}")
        
        return test_accuracy
    
    def predict(self, student_data):
        """Predict career for a student"""
        features = self.prepare_features(pd.DataFrame([student_data]))
        features_scaled = self.scaler.transform(features)
        
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Get top 3 recommendations
        top_indices = np.argsort(probabilities)[-3:][::-1]
        recommendations = []
        
        for idx in top_indices:
            career = self.label_encoder.inverse_transform([idx])[0]
            probability = probabilities[idx]
            skills = self.career_skills.get(career, ['Skills information not available'])
            
            recommendations.append({
                'career': career,
                'probability': probability,
                'skills': skills
            })
        
        return recommendations
    
    def save_model(self, filepath='career_model.pkl'):
        """Save trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'career_skills': self.career_skills
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"\n✓ Model saved to {filepath}")
    
    def load_model(self, filepath='career_model.pkl'):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_columns = model_data['feature_columns']
        self.career_skills = model_data['career_skills']
        print(f"✓ Model loaded from {filepath}")


def get_manual_input():
    """Get student data through manual input"""
    print("\n" + "=" * 60)
    print("ENTER STUDENT INFORMATION")
    print("=" * 60)
    
    student_data = {}
    
    # Gender
    while True:
        gender = input("\nGender (male/female): ").strip().lower()
        if gender in ['male', 'female']:
            student_data['gender'] = gender
            break
        print("Invalid input. Please enter 'male' or 'female'")
    
    # Part-time job
    while True:
        job = input("Has part-time job? (yes/no): ").strip().lower()
        if job in ['yes', 'no', 'y', 'n']:
            student_data['part_time_job'] = job in ['yes', 'y']
            break
        print("Invalid input. Please enter 'yes' or 'no'")
    
    # Absence days
    while True:
        try:
            absence = int(input("Number of absence days (0-30): ").strip())
            if 0 <= absence <= 30:
                student_data['absence_days'] = absence
                break
            print("Please enter a number between 0 and 30")
        except ValueError:
            print("Invalid input. Please enter a number")
    
    # Extracurricular activities
    while True:
        extra = input("Participates in extracurricular activities? (yes/no): ").strip().lower()
        if extra in ['yes', 'no', 'y', 'n']:
            student_data['extracurricular_activities'] = extra in ['yes', 'y']
            break
        print("Invalid input. Please enter 'yes' or 'no'")
    
    # Weekly self-study hours
    while True:
        try:
            hours = int(input("Weekly self-study hours (0-50): ").strip())
            if 0 <= hours <= 50:
                student_data['weekly_self_study_hours'] = hours
                break
            print("Please enter a number between 0 and 50")
        except ValueError:
            print("Invalid input. Please enter a number")
    
    # Academic scores
    print("\n--- ACADEMIC SCORES (0-100) ---")
    subjects = ['math', 'history', 'physics', 'chemistry', 'biology', 'english', 'geography']
    
    for subject in subjects:
        while True:
            try:
                score = int(input(f"{subject.capitalize()} score: ").strip())
                if 0 <= score <= 100:
                    student_data[f'{subject}_score'] = score
                    break
                print("Please enter a score between 0 and 100")
            except ValueError:
                print("Invalid input. Please enter a number")
    
    return student_data


def interactive_menu():
    """Interactive menu for the application"""
    model = None
    data_loader = MultiFileDataLoader()
    
    while True:
        print("\n" + "=" * 60)
        print("CAREER RECOMMENDATION SYSTEM - MAIN MENU")
        print("=" * 60)
        print("1. Load your own dataset (multiple CSV/Excel files)")
        print("2. Train model with loaded dataset")
        print("3. Load existing trained model")
        print("4. Make prediction (manual input)")
        print("5. Make prediction (sample student)")
        print("6. Save current model")
        print("7. View loaded data summary")
        print("8. Exit")
        print("=" * 60)
        
        choice = input("\nEnter your choice (1-8): ").strip()
        
        if choice == '1':
            print("\n" + "=" * 60)
            print("LOAD YOUR OWN DATASET")
            print("=" * 60)
            print("\nYou'll need to provide:")
            print("  1. Student data file (CSV/Excel)")
            print("  2. Career-Skills mapping file (CSV/Excel) - optional")
            print("  3. Skills matrix file (CSV/Excel) - optional")
            
            # Load student data
            student_file = input("\nEnter path to student data file: ").strip()
            student_df = data_loader.load_student_data(student_file)
            
            if student_df is None:
                print("Failed to load student data. Please try again.")
                continue
            
            # Load career-skills mapping (optional)
            load_mapping = input("\nDo you have a career-skills mapping file? (yes/no): ").strip().lower()
            if load_mapping in ['yes', 'y']:
                mapping_file = input("Enter path to career-skills mapping file: ").strip()
                mapping = data_loader.load_career_skills_mapping(mapping_file)
            
            # Load skills matrix (optional)
            load_matrix = input("\nDo you have a skills matrix file? (yes/no): ").strip().lower()
            if load_matrix in ['yes', 'y']:
                matrix_file = input("Enter path to skills matrix file: ").strip()
                data_loader.load_skills_matrix(matrix_file)
            
            print("\n✓ Dataset loaded successfully!")
        
        elif choice == '2':
            if data_loader.student_data is None:
                print("\n✗ No dataset loaded. Please load a dataset first (Option 1).")
                continue
            
            print("\n" + "=" * 60)
            print("TRAIN MODEL")
            print("=" * 60)
            
            model = CareerRecommendationModel()
            
            # Set career skills if available
            if data_loader.career_skills_mapping:
                model.set_career_skills(data_loader.career_skills_mapping)
                print(f"✓ Using {len(data_loader.career_skills_mapping)} career-skill mappings")
            else:
                print("⚠ No career-skills mapping loaded. Using default mapping.")
                # Use default mapping from the original code
                default_mapping = {
                    'Software Development and Engineering': ['Web Development', 'Mobile App Development'],
                    'Doctor': ['Biology', 'Chemistry', 'Medical Knowledge'],
                    'Lawyer': ['Communication', 'Critical Thinking', 'Law'],
                    'Teacher': ['Communication', 'Subject Knowledge', 'Education'],
                    'Business Owner': ['Leadership', 'Management', 'Communication'],
                    'Scientist': ['Research', 'Analysis', 'Problem Solving'],
                    'Software Engineer': ['Programming', 'Problem Solving', 'Algorithms'],
                    'Government Officer': ['Administration', 'Policy Knowledge'],
                    'Artist': ['Creativity', 'Visual Arts', 'Expression']
                }
                model.set_career_skills(default_mapping)
            
            accuracy = model.train(data_loader.get_merged_data())
            print(f"\n✓ Model training complete!")
        
        elif choice == '3':
            try:
                filepath = input("\nEnter model filepath (press Enter for 'career_model.pkl'): ").strip()
                if not filepath:
                    filepath = 'career_model.pkl'
                model = CareerRecommendationModel()
                model.load_model(filepath)
            except FileNotFoundError:
                print("✗ Model file not found.")
            except Exception as e:
                print(f"✗ Error loading model: {e}")
        
        elif choice == '4':
            if model is None:
                print("\n✗ No model loaded. Please train or load a model first.")
                continue
            
            student_data = get_manual_input()
            
            print("\n" + "=" * 60)
            print("GENERATING RECOMMENDATIONS...")
            print("=" * 60)
            
            recommendations = model.predict(student_data)
            
            print("\n" + "=" * 60)
            print("CAREER RECOMMENDATIONS")
            print("=" * 60)
            for i, rec in enumerate(recommendations, 1):
                print(f"\n{i}. {rec['career']}")
                print(f"   Confidence: {rec['probability']:.2%}")
                print(f"   Key Skills: {', '.join(rec['skills'][:5])}")
            print("\n" + "=" * 60)
        
        elif choice == '5':
            if model is None:
                print("\n✗ No model loaded. Please train or load a model first.")
                continue
            
            test_student = {
                'gender': 'male',
                'part_time_job': False,
                'absence_days': 3,
                'extracurricular_activities': True,
                'weekly_self_study_hours': 25,
                'math_score': 92,
                'history_score': 75,
                'physics_score': 88,
                'chemistry_score': 85,
                'biology_score': 70,
                'english_score': 80,
                'geography_score': 78
            }
            
            print("\nUsing sample student data:")
            print(f"  Gender: {test_student['gender']}")
            print(f"  Math: {test_student['math_score']}, Physics: {test_student['physics_score']}")
            print(f"  Study Hours: {test_student['weekly_self_study_hours']}/week")
            
            recommendations = model.predict(test_student)
            
            print("\n" + "=" * 60)
            print("CAREER RECOMMENDATIONS")
            print("=" * 60)
            for i, rec in enumerate(recommendations, 1):
                print(f"\n{i}. {rec['career']}")
                print(f"   Confidence: {rec['probability']:.2%}")
                print(f"   Key Skills: {', '.join(rec['skills'][:5])}")
            print("\n" + "=" * 60)
        
        elif choice == '6':
            if model is None:
                print("\n✗ No model to save. Please train a model first.")
                continue
            
            filepath = input("\nEnter filepath to save (press Enter for 'career_model.pkl'): ").strip()
            if not filepath:
                filepath = 'career_model.pkl'
            model.save_model(filepath)
        
        elif choice == '7':
            print("\n" + "=" * 60)
            print("LOADED DATA SUMMARY")
            print("=" * 60)
            
            if data_loader.student_data is not None:
                print(f"\n✓ Student Data: {len(data_loader.student_data)} records")
                print(f"   Columns: {len(data_loader.student_data.columns)}")
                
                if 'career_aspiration' in data_loader.student_data.columns:
                    careers = data_loader.student_data['career_aspiration'].value_counts()
                    print(f"\n   Career Distribution:")
                    for career, count in careers.items():
                        print(f"     - {career}: {count}")
            else:
                print("\n✗ No student data loaded")
            
            if data_loader.career_skills_mapping:
                print(f"\n✓ Career-Skills Mapping: {len(data_loader.career_skills_mapping)} careers")
            else:
                print("\n✗ No career-skills mapping loaded")
            
            if data_loader.skills_matrix is not None:
                print(f"\n✓ Skills Matrix: {data_loader.skills_matrix.shape}")
            else:
                print("\n✗ No skills matrix loaded")
        
        elif choice == '8':
            print("\n" + "=" * 60)
            print("Thank you for using Career Recommendation System!")
            print("=" * 60)
            break
        
        else:
            print("\n✗ Invalid choice. Please enter a number between 1 and 8.")


if __name__ == "__main__":
    print("=" * 60)
    print("CAREER RECOMMENDATION AI MODEL")
    print("Multi-File Dataset Support")
    print("=" * 60)
    print("\nThis system supports loading data from multiple files:")
    print("  • Student data (CSV/Excel)")
    print("  • Career-Skills mapping (CSV/Excel)")
    print("  • Skills matrix (CSV/Excel)")
    
    interactive_menu()