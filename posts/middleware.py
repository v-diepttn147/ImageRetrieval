from django.contrib.auth import login
from django.contrib.auth.models import User

class AutoLoginMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Auto-login only if anonymous and accessing /create/
        if request.path.startswith('/create/') and not request.user.is_authenticated:
            try:
                user = User.objects.get(username='gau147')
                user.backend = 'django.contrib.auth.backends.ModelBackend'
                login(request, user)
            except User.DoesNotExist:
                pass

        return self.get_response(request)
