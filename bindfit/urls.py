from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^fit$',     views.FitterView.as_view(),        name="bindfit_fit"),
    url(r'^list$',    views.FitterListView.as_view(),    name="bindfit_list"),
    url(r'^options$', views.FitterOptionsView.as_view(), name="bindfit_options"),
    url(r'^labels$',  views.FitterLabelsView.as_view(),  name="bindfit_labels"),
    url(r'^upload$',  views.UploadView.as_view(),        name="bindfit_upload"),
]
