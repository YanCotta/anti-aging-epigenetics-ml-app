from django.contrib.auth.models import AbstractUser
from django.db import models
import json

#PLACEHOLDER CODE #1

class User(AbstractUser):
    """Extended user model for authentication"""
    email = models.EmailField(unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']


class UserProfile(models.Model):
    """User demographic and health profile"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    age = models.IntegerField()
    gender = models.CharField(max_length=10, choices=[('M', 'Male'), ('F', 'Female'), ('O', 'Other')])
    weight = models.FloatField(help_text="Weight in kg")
    height = models.FloatField(help_text="Height in cm")
    medical_history = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.username} - Profile"


class GeneticProfile(models.Model):
    """Genetic/epigenetic data for the user"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='genetic_profile')
    raw_data = models.JSONField(help_text="Raw genetic data as JSON")
    processed_features = models.JSONField(blank=True, null=True, help_text="Processed features for ML")
    upload_date = models.DateTimeField(auto_now_add=True)
    file_name = models.CharField(max_length=255)
    file_size = models.IntegerField()

    def __str__(self):
        return f"{self.user.username} - Genetic Profile"


class Habits(models.Model):
    """Lifestyle habits and behaviors"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='habits')
    exercise_frequency = models.IntegerField(help_text="Days per week")
    sleep_hours = models.FloatField(help_text="Average hours per night")
    stress_level = models.IntegerField(choices=[(i, str(i)) for i in range(1, 11)])
    diet_quality = models.IntegerField(choices=[(i, str(i)) for i in range(1, 11)])
    smoking = models.BooleanField(default=False)
    alcohol_consumption = models.IntegerField(help_text="Drinks per week")
    recorded_date = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name_plural = "Habits"
        ordering = ['-recorded_date']

    def __str__(self):
        return f"{self.user.username} - Habits ({self.recorded_date.date()})"


class Prediction(models.Model):
    """ML model predictions"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='predictions')
    biological_age = models.FloatField()
    chronological_age = models.FloatField()
    aging_rate = models.FloatField()
    confidence_score = models.FloatField()
    shap_values = models.JSONField(help_text="SHAP explanation values")
    model_version = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.username} - Prediction ({self.created_at.date()})"
    

"""
PLACEHOLDER CODE #2

from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    age = models.IntegerField(null=True, blank=True, validators=[MinValueValidator(18), MaxValueValidator(120)])
    gender = models.CharField(max_length=10, choices=[('M', 'Male'), ('F', 'Female'), ('Other', 'Other')], null=True, blank=True)

    def __str__(self):
        return f"Profile for {self.user.username}"

class GeneticProfile(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    snp_data = models.JSONField()  # Store SNP dict, e.g., {'SIRT1_rs7896005': 'A/G', ...}
    upload_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Genetic profile for {self.user.username}"

class Habits(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    exercises_per_week = models.IntegerField(validators=[MinValueValidator(0), MaxValueValidator(7)])
    daily_calories = models.IntegerField(validators=[MinValueValidator(1000), MaxValueValidator(5000)])
    alcohol_doses_per_week = models.FloatField(validators=[MinValueValidator(0)])
    years_smoking = models.IntegerField(validators=[MinValueValidator(0)])
    hours_of_sleep = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(24)])
    stress_level = models.IntegerField(validators=[MinValueValidator(1), MaxValueValidator(10)])
    entry_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Habits for {self.user.username}"
    
    """