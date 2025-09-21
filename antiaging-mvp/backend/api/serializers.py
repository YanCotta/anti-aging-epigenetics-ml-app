"""
Django REST Framework Serializers

Provides serialization and validation for API endpoints.
"""

from rest_framework import serializers
from django.contrib.auth.password_validation import validate_password
from .models import User, UserProfile, GeneticProfile, Habits, Prediction


class UserSerializer(serializers.ModelSerializer):
    """User serializer for basic user data"""
    class Meta:
        model = User
        fields = ('id', 'username', 'email', 'first_name', 'last_name', 'created_at')
        read_only_fields = ('id', 'created_at')


class UserRegistrationSerializer(serializers.ModelSerializer):
    """User registration serializer with validation"""
    password = serializers.CharField(write_only=True, validators=[validate_password])
    password_confirm = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = ('username', 'email', 'password', 'password_confirm', 'first_name', 'last_name')

    def validate(self, attrs):
        if attrs['password'] != attrs['password_confirm']:
            raise serializers.ValidationError("Passwords don't match")
        return attrs

    def create(self, validated_data):
        validated_data.pop('password_confirm')
        user = User.objects.create_user(**validated_data)
        return user


class UserProfileSerializer(serializers.ModelSerializer):
    """User profile serializer"""
    user = UserSerializer(read_only=True)

    class Meta:
        model = UserProfile
        fields = '__all__'
        read_only_fields = ('user', 'created_at', 'updated_at')

    def validate_age(self, value):
        if value < 18 or value > 120:
            raise serializers.ValidationError("Age must be between 18 and 120")
        return value

    def validate_weight(self, value):
        if value < 30 or value > 300:
            raise serializers.ValidationError("Weight must be between 30 and 300 kg")
        return value

    def validate_height(self, value):
        if value < 100 or value > 250:
            raise serializers.ValidationError("Height must be between 100 and 250 cm")
        return value


class GeneticProfileSerializer(serializers.ModelSerializer):
    """Genetic profile serializer"""
    user = UserSerializer(read_only=True)

    class Meta:
        model = GeneticProfile
        fields = '__all__'
        read_only_fields = ('user', 'upload_date')

    def to_representation(self, instance):
        """Custom representation to hide sensitive raw data"""
        data = super().to_representation(instance)
        # Don't return raw genetic data in API responses for privacy
        data['raw_data'] = {'status': 'uploaded', 'size': len(str(instance.raw_data))}
        return data


class HabitsSerializer(serializers.ModelSerializer):
    """Habits serializer with validation"""
    user = UserSerializer(read_only=True)

    class Meta:
        model = Habits
        fields = '__all__'
        read_only_fields = ('user', 'recorded_date')

    def validate_exercise_frequency(self, value):
        if value < 0 or value > 7:
            raise serializers.ValidationError("Exercise frequency must be between 0 and 7 days")
        return value

    def validate_sleep_hours(self, value):
        if value < 3 or value > 12:
            raise serializers.ValidationError("Sleep hours must be between 3 and 12")
        return value

    def validate_stress_level(self, value):
        if value < 1 or value > 10:
            raise serializers.ValidationError("Stress level must be between 1 and 10")
        return value

    def validate_diet_quality(self, value):
        if value < 1 or value > 10:
            raise serializers.ValidationError("Diet quality must be between 1 and 10")
        return value

    def validate_alcohol_consumption(self, value):
        if value < 0 or value > 50:
            raise serializers.ValidationError("Alcohol consumption must be between 0 and 50 drinks per week")
        return value


class PredictionSerializer(serializers.ModelSerializer):
    """Prediction serializer"""
    user = UserSerializer(read_only=True)
    aging_difference = serializers.SerializerMethodField()

    class Meta:
        model = Prediction
        fields = '__all__'
        read_only_fields = ('user', 'created_at')

    def get_aging_difference(self, obj):
        """Calculate difference between biological and chronological age"""
        return round(obj.biological_age - obj.chronological_age, 2)


class PredictionSummarySerializer(serializers.ModelSerializer):
    """Simplified prediction serializer for dashboard"""
    aging_difference = serializers.SerializerMethodField()

    class Meta:
        model = Prediction
        fields = ('id', 'biological_age', 'chronological_age', 'aging_rate', 
                 'confidence_score', 'aging_difference', 'created_at')

    def get_aging_difference(self, obj):
        return round(obj.biological_age - obj.chronological_age, 2)
    


#PLACEHOLDER CODE #2

    """

from rest_framework import serializers
from .models import GeneticProfile, Habits

class GeneticProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = GeneticProfile
        fields = '__all__'

class HabitsSerializer(serializers.ModelSerializer):
    class Meta:
        model = Habits
        fields = ['exercises_per_week', 'daily_calories', 'alcohol_doses_per_week', 'years_smoking', 'hours_of_sleep', 'stress_level']

"""