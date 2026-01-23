"""Django Stack Profile - Expert guidance for code generation.

This profile is injected into the implementation prompt when the user
selects Django as their tech stack. It provides Django 5.x patterns,
Django REST Framework best practices, and modern project structure.
"""

PROFILE_NAME = "django"
PROFILE_VERSION = "1.0"

# Technologies covered by this profile
TECHNOLOGIES = ["django", "drf", "django-rest-framework", "celery"]

# Injected into the system prompt for implementation
SYSTEM_GUIDANCE = """
## Django Expert Guidelines

You are generating Django code. Follow these patterns exactly:

### Project Structure (CRITICAL)

```
project/
├── config/              # Project settings (renamed from project_name/)
│   ├── __init__.py
│   ├── settings/
│   │   ├── __init__.py
│   │   ├── base.py     # Common settings
│   │   ├── local.py    # Development
│   │   └── production.py
│   ├── urls.py
│   └── wsgi.py
├── apps/
│   ├── users/          # Custom user app
│   │   ├── models.py
│   │   ├── views.py
│   │   ├── serializers.py
│   │   ├── urls.py
│   │   └── tests/
│   └── core/           # Shared utilities
├── manage.py
└── requirements/
    ├── base.txt
    ├── local.txt
    └── production.txt
```

### Models (CRITICAL - Django 5.x)

```python
# apps/users/models.py
from django.contrib.auth.models import AbstractUser
from django.db import models

class User(AbstractUser):
    \"\"\"Custom user model - ALWAYS extend AbstractUser.\"\"\"
    email = models.EmailField(unique=True)

    # Use this in settings.py: AUTH_USER_MODEL = 'users.User'

    class Meta:
        ordering = ['-date_joined']

class Profile(models.Model):
    \"\"\"User profile with additional data.\"\"\"
    user = models.OneToOneField(
        'users.User',
        on_delete=models.CASCADE,
        related_name='profile'
    )
    bio = models.TextField(blank=True)
    avatar = models.ImageField(upload_to='avatars/', blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Profile'
        verbose_name_plural = 'Profiles'

    def __str__(self):
        return f"Profile of {self.user.username}"
```

### Django REST Framework Views

```python
# apps/users/views.py
from rest_framework import viewsets, permissions, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.shortcuts import get_object_or_404

from .models import User
from .serializers import UserSerializer, UserCreateSerializer

class UserViewSet(viewsets.ModelViewSet):
    \"\"\"
    API endpoint for users.

    list: Get all users
    create: Register new user
    retrieve: Get user by ID
    update: Update user (owner only)
    destroy: Delete user (admin only)
    \"\"\"
    queryset = User.objects.all()
    permission_classes = [permissions.IsAuthenticated]

    def get_serializer_class(self):
        if self.action == 'create':
            return UserCreateSerializer
        return UserSerializer

    def get_permissions(self):
        if self.action == 'create':
            return [permissions.AllowAny()]
        if self.action == 'destroy':
            return [permissions.IsAdminUser()]
        return super().get_permissions()

    @action(detail=False, methods=['get'])
    def me(self, request):
        \"\"\"Get current user profile.\"\"\"
        serializer = self.get_serializer(request.user)
        return Response(serializer.data)
```

### Serializers

```python
# apps/users/serializers.py
from rest_framework import serializers
from django.contrib.auth.password_validation import validate_password
from .models import User, Profile

class ProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = Profile
        fields = ['bio', 'avatar']

class UserSerializer(serializers.ModelSerializer):
    profile = ProfileSerializer(read_only=True)

    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name', 'profile']
        read_only_fields = ['id']

class UserCreateSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, validators=[validate_password])
    password_confirm = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = ['username', 'email', 'password', 'password_confirm']

    def validate(self, data):
        if data['password'] != data['password_confirm']:
            raise serializers.ValidationError({'password_confirm': 'Passwords must match'})
        return data

    def create(self, validated_data):
        validated_data.pop('password_confirm')
        user = User.objects.create_user(**validated_data)
        Profile.objects.create(user=user)  # Create profile automatically
        return user
```

### URL Configuration

```python
# apps/users/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import UserViewSet

router = DefaultRouter()
router.register('users', UserViewSet)

urlpatterns = [
    path('', include(router.urls)),
]

# config/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('apps.users.urls')),
    path('api/auth/', include('dj_rest_auth.urls')),  # Login, logout, password reset
]
```

### Settings Best Practices

```python
# config/settings/base.py
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent

SECRET_KEY = os.getenv('SECRET_KEY')
DEBUG = False

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    # Third party
    'rest_framework',
    'corsheaders',
    # Local apps
    'apps.users',
    'apps.core',
]

AUTH_USER_MODEL = 'users.User'

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20,
}
```

### Signals (Use Sparingly)

```python
# apps/users/signals.py
from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import User, Profile

@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    \"\"\"Create profile when user is created.\"\"\"
    if created:
        Profile.objects.create(user=instance)
```

### Celery Tasks

```python
# apps/core/tasks.py
from celery import shared_task
from django.core.mail import send_mail

@shared_task
def send_welcome_email(user_id: int):
    \"\"\"Send welcome email to new user.\"\"\"
    from apps.users.models import User
    user = User.objects.get(id=user_id)
    send_mail(
        subject='Welcome!',
        message=f'Welcome to our platform, {user.username}!',
        from_email='noreply@example.com',
        recipient_list=[user.email],
    )
```

### Common Mistakes to Avoid

1. **Not creating custom User model** - ALWAYS extend AbstractUser before first migration
2. **Using function-based views for APIs** - Use ViewSets for REST APIs
3. **Putting logic in views** - Use services or model methods
4. **Not using transactions** - Use `@transaction.atomic` for multi-model operations
5. **Hardcoding settings** - Use environment variables via python-dotenv
6. **Not using select_related/prefetch_related** - N+1 query problem

```python
# WRONG: N+1 queries
for user in User.objects.all():
    print(user.profile.bio)

# CORRECT: Single query with join
for user in User.objects.select_related('profile').all():
    print(user.profile.bio)
```

### Required Dependencies

```
# requirements/base.txt
Django>=5.0
djangorestframework>=3.14.0
django-cors-headers>=4.3.0
djangorestframework-simplejwt>=5.3.0
python-dotenv>=1.0.0
Pillow>=10.1.0
```
"""

DEPENDENCIES = [
    "Django>=5.0",
    "djangorestframework>=3.14.0",
    "django-cors-headers>=4.3.0",
    "djangorestframework-simplejwt>=5.3.0",
    "python-dotenv>=1.0.0",
]

OPTIONAL_DEPENDENCIES = {
    "database": ["psycopg2-binary>=2.9.0"],
    "celery": ["celery>=5.3.0", "redis>=5.0.0"],
    "testing": ["pytest-django>=4.7.0", "factory-boy>=3.3.0"],
    "storage": ["django-storages>=1.14.0", "boto3>=1.33.0"],
}

TRIGGER_KEYWORDS = [
    "django",
    "drf",
    "django rest",
    "django-rest-framework",
    "django api",
]


def should_apply(tech_stack: list[str] | None, prompt: str) -> bool:
    """Determine if this profile should be applied."""
    prompt_lower = prompt.lower()

    # Check explicit tech stack selection
    if tech_stack:
        tech_lower = [t.lower() for t in tech_stack]
        if any(kw in tech_lower for kw in ["django", "drf"]):
            return True

    # Check prompt keywords
    return any(kw in prompt_lower for kw in TRIGGER_KEYWORDS)


def get_guidance() -> str:
    """Get the full guidance text to inject into prompts."""
    return SYSTEM_GUIDANCE


def get_dependencies(features: list[str] | None = None) -> list[str]:
    """Get recommended dependencies."""
    deps = DEPENDENCIES.copy()

    if features:
        for feature in features:
            if feature in OPTIONAL_DEPENDENCIES:
                deps.extend(OPTIONAL_DEPENDENCIES[feature])

    return deps
