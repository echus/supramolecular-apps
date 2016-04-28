from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^fit$',      views.FitView.as_view(),         name="bindfit_fit"),
    url(r'^fit/extras/mc$',
                       views.FitMonteCarloView.as_view(),
                       name="bindfit_fit_save"),
    url(r'^fit/save$', views.FitSaveView.as_view(),     name="bindfit_fit_save"),
    url(r'^edit$',     views.FitEditEmailView.as_view(),name="bindfit_edit"),
    url(r'^search$',   views.FitSearchView.as_view(),   name="bindfit_search"),
    url(r'^search/email$',   
                       views.FitSearchEmailView.as_view(),   
                       name="bindfit_search_email"),
    url(r'^search/id/(?P<id>[0-9a-zA-Z-]+)$', 
                       views.FitRetrieveView.as_view(), 
                       name="bindfit_search_id"),
    url(r'^list$',     views.FitListView.as_view(),     name="bindfit_list"),
    url(r'^options$',  views.FitOptionsView.as_view(),  name="bindfit_options"),
    url(r'^labels$',   views.FitLabelsView.as_view(),   name="bindfit_labels"),
    url(r'^export$',   views.FitExportView.as_view(),   name="bindfit_export"),
    url(r'^upload$',   views.UploadDataView.as_view(),  name="bindfit_upload"),
]
