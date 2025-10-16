"""
Django REST Framework Views

Provides API views and endpoints for the Django application.
"""

from rest_framework import status, generics, permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import authenticate
from django.conf import settings
from .models import User, UserProfile, GeneticProfile, Habits, Prediction
from .serializers import (
    UserSerializer, UserProfileSerializer, GeneticProfileSerializer,
    HabitsSerializer, PredictionSerializer, UserRegistrationSerializer
)
from .ml.predict import MLPredictor
import json
import logging

logger = logging.getLogger(__name__)


class UserRegistrationView(generics.CreateAPIView):
    """User registration endpoint"""
    queryset = User.objects.all()
    serializer_class = UserRegistrationSerializer
    permission_classes = [permissions.AllowAny]

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        
        refresh = RefreshToken.for_user(user)
        return Response({
            'user': UserSerializer(user).data,
            'refresh': str(refresh),
            'access': str(refresh.access_token),
        }, status=status.HTTP_201_CREATED)


class LoginView(APIView):
    """User login endpoint"""
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        email = request.data.get('email')
        password = request.data.get('password')
        
        if email and password:
            user = authenticate(username=email, password=password)
            if user:
                refresh = RefreshToken.for_user(user)
                return Response({
                    'user': UserSerializer(user).data,
                    'refresh': str(refresh),
                    'access': str(refresh.access_token),
                })
        
        return Response(
            {'error': 'Invalid credentials'}, 
            status=status.HTTP_401_UNAUTHORIZED
        )


class UserProfileView(generics.RetrieveUpdateCreateAPIView):
    """User profile management"""
    serializer_class = UserProfileSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self):
        profile, created = UserProfile.objects.get_or_create(user=self.request.user)
        return profile


class GeneticUploadView(APIView):
    """Genetic data upload endpoint"""
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        try:
            uploaded_file = request.FILES.get('genetic_file')
            if not uploaded_file:
                return Response(
                    {'error': 'No file provided'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Process the uploaded file (simplified)
            raw_data = self._process_genetic_file(uploaded_file)
            
            genetic_profile, created = GeneticProfile.objects.get_or_create(
                user=request.user,
                defaults={
                    'raw_data': raw_data,
                    'file_name': uploaded_file.name,
                    'file_size': uploaded_file.size
                }
            )
            
            if not created:
                genetic_profile.raw_data = raw_data
                genetic_profile.file_name = uploaded_file.name
                genetic_profile.file_size = uploaded_file.size
                genetic_profile.save()

            serializer = GeneticProfileSerializer(genetic_profile)
            return Response(serializer.data, status=status.HTTP_201_CREATED)

        except Exception as e:
            logger.error(f"Genetic upload error: {str(e)}")
            return Response(
                {'error': 'File processing failed'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def _process_genetic_file(self, uploaded_file):
        """Process uploaded genetic file (placeholder implementation)"""
        # This would contain actual genetic data processing logic
        content = uploaded_file.read().decode('utf-8')
        return {'raw_content': content[:1000]}  # Truncated for demo


class HabitsView(generics.ListCreateAPIView):
    """Habits tracking endpoint"""
    serializer_class = HabitsSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Habits.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class PredictionView(APIView):
    """ML prediction endpoint"""
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        try:
            user = request.user
            
            # Check if user has required data
            if not hasattr(user, 'profile') or not hasattr(user, 'genetic_profile'):
                return Response(
                    {'error': 'Missing required profile or genetic data'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Get latest habits
            latest_habits = user.habits.first()
            if not latest_habits:
                return Response(
                    {'error': 'No habits data found'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Run ML prediction
            predictor = MLPredictor()
            prediction_result = predictor.predict_aging(user)

            # Save prediction
            prediction = Prediction.objects.create(
                user=user,
                biological_age=prediction_result['biological_age'],
                chronological_age=prediction_result['chronological_age'],
                aging_rate=prediction_result['aging_rate'],
                confidence_score=prediction_result['confidence_score'],
                shap_values=prediction_result['shap_values'],
                model_version=prediction_result['model_version']
            )

            serializer = PredictionSerializer(prediction)
            return Response(serializer.data, status=status.HTTP_201_CREATED)

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return Response(
                {'error': 'Prediction failed'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class PredictionHistoryView(generics.ListAPIView):
    """User's prediction history"""
    serializer_class = PredictionSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Prediction.objects.filter(user=self.request.user)


@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def dashboard_data(request):
    """Dashboard summary data"""
    user = request.user
    
    # Get latest prediction
    latest_prediction = user.predictions.first()
    
    # Get habits count
    habits_count = user.habits.count()
    
    # Get profile completion status
    profile_complete = (
        hasattr(user, 'profile') and 
        hasattr(user, 'genetic_profile') and 
        habits_count > 0
    )
    
    return Response({
        'profile_complete': profile_complete,
        'habits_count': habits_count,
        'latest_prediction': PredictionSerializer(latest_prediction).data if latest_prediction else None,
        'user': UserSerializer(user).data
    })

#PLACEHOLDER CODE #2

""" 

from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from .serializers import GeneticProfileSerializer, HabitsSerializer
from .ml.predict import predict_with_explain
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class SignupView(APIView):
    permission_classes = [AllowAny]
    def post(self, request):
        try:
            username = request.data['username']
            password = request.data['password']
            email = request.data.get('email', '')
            if User.objects.filter(username=username).exists():
                return Response({"error": "Username exists"}, status=status.HTTP_400_BAD_REQUEST)
            user = User.objects.create_user(username=username, email=email, password=password)
            refresh = RefreshToken.for_user(user)
            return Response({
                'refresh': str(refresh),
                'access': str(refresh.access_token),
            })
        except Exception as e:
            logger.error(f"Signup error: {e}")
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

class LoginView(APIView):
    permission_classes = [AllowAny]
    def post(self, request):
        try:
            username = request.data['username']
            password = request.data['password']
            user = authenticate(username=username, password=password)
            if user:
                refresh = RefreshToken.for_user(user)
                return Response({
                    'refresh': str(refresh),
                    'access': str(refresh.access_token),
                })
            return Response({"error": "Invalid credentials"}, status=status.HTTP_401_UNAUTHORIZED)
        except Exception as e:
            logger.error(f"Login error: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class UploadGeneticView(APIView):
    permission_classes = [IsAuthenticated]
    def post(self, request):
        try:
            file = request.FILES['file']
            df = pd.read_csv(file)
            # Validate schema (adjust expected_cols to match your SNPs + demographics)
            expected_cols = ['SIRT1_rs7896005', 'FOXO3_rs2802292', 'APOE_rs429358', 'age', 'gender']  # Example; expand
            if set(df.columns) - {'risk'} != set(expected_cols):  # Ignore 'risk' if present
                return Response({"error": "Invalid CSV schema"}, status=status.HTTP_400_BAD_REQUEST)
            snp_data = df.iloc[0].to_dict()  # Assume single row for user
            profile = GeneticProfile.objects.create(user=request.user, snp_data=snp_data)
            serializer = GeneticProfileSerializer(profile)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class SubmitHabitsView(APIView):
    permission_classes = [IsAuthenticated]
    def post(self, request):
        serializer = HabitsSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(user=request.user)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class PredictView(APIView):
    permission_classes = [IsAuthenticated]
    def get(self, request):
        try:
            # Fetch latest data
            genetic = GeneticProfile.objects.filter(user=request.user).latest('upload_date')
            habits = Habits.objects.filter(user=request.user).latest('entry_date')
            # Combine into DF
            data_dict = {**genetic.snp_data, **habits.__dict__}  # Merge; adjust keys
            df = pd.DataFrame([data_dict])
            pred, explanations = predict_with_explain(df)
            return Response({"prediction": pred[0], "explanations": explanations})
        except Exception as e:
            logger.error(f"Predict error: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            """