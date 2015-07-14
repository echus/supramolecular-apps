from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^fit$', views.fit, name="bindfit"),
]
