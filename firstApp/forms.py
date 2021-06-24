from django import forms

class inputForm(forms.Form):
    title = forms.CharField(widget=forms.TextInput(
            attrs={'class' : 'form-control', 'placeholder': 'Title'}))
    news = forms.CharField(widget=forms.Textarea(
            attrs={'id' : 'newsbox', 'class' : 'form-control', 'placeholder': 'News', })) 