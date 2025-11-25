from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views

urlpatterns = [
    # page
    path("", views.index, name="home"),
    path("start-chat", views.chatbot_view, name="chat"),
    path("api/chat", views.chat_asistance, name="chat-stream")

]

# if settings.DEBUG:
#     urlpatterns += static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS)