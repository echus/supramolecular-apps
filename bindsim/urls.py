from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^nmr/1to1$', views.nmr_1to1, name="nmr_1to1"),
    url(r'^nmr/1to2$', views.nmr_1to2, name="nmr_1to2"),
]
