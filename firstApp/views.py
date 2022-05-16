from django.shortcuts import render
from .forms import inputForm
import pickle

# Create your views here.
def index(request):
    model = pickle.load(open('./fake_news_classifier_svc_pipe.sav','rb'))
    form = inputForm()
    pred = [2]
    if request.method == 'POST':
        form = inputForm(request.POST)

        if form.is_valid():
            title = form.cleaned_data['title']
            news = form.cleaned_data['news']
            
            pred = model.predict([title + news])

    return render(request,'firstApp/index.html',{'form':form, 'output':pred[0]})

def about(request):
    return render(request, 'firstApp/about.html')

def notebook(request):
    return render(request, 'firstApp/notebook.html')
