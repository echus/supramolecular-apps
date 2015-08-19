from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^fit$',     views.FitView.as_view(),        name="bindfit_fit"),
    url(r'^list$',    views.FitListView.as_view(),    name="bindfit_list"),
    url(r'^options$', views.FitOptionsView.as_view(), name="bindfit_options"),
    url(r'^labels$',  views.FitLabelsView.as_view(),  name="bindfit_labels"),
    url(r'^export$',  views.FitExportView.as_view(),  name="bindfit_export"),
    url(r'^upload$',  views.UploadView.as_view(),     name="bindfit_upload"),
]
