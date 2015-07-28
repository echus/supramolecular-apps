from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^fit$', views.FitterView.as_view(), name="bindfit_fit"),
    url(r'^upload$', views.UploadView.as_view(), name="bindfit_upload"),
]
